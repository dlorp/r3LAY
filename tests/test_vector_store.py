"""Tests for r3lay.core.vector_store module.

Covers:
- VectorStoreBase abstract interface verification
- NumpyFallbackStore: add, search, remove, save/load, count, from_numpy, get_id
- FAISSVectorStore: same coverage + IVF rebuild, score clamping, thread safety
- create_vector_store factory function
- Migration: from_numpy with real vectors, verify search works after migration
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from r3lay.core.vector_store import (
    NumpyFallbackStore,
    VectorStoreBase,
    create_vector_store,
)

# Reproducible random generator
RNG = np.random.default_rng(42)


def _make_vectors(n: int, dim: int = 64) -> np.ndarray:
    """Generate reproducible random vectors."""
    return RNG.standard_normal((n, dim)).astype(np.float32)


def _make_query(dim: int = 64) -> np.ndarray:
    """Generate a reproducible random query vector."""
    return RNG.standard_normal(dim).astype(np.float32)


# ============================================================================
# VectorStoreBase Abstract Interface Tests
# ============================================================================


class TestVectorStoreBase:
    """Tests for VectorStoreBase abstract interface."""

    def test_cannot_instantiate_directly(self):
        """VectorStoreBase cannot be instantiated — it is abstract."""
        with pytest.raises(TypeError, match="abstract"):
            VectorStoreBase()  # type: ignore[abstract]

    def test_subclass_must_implement_all_methods(self):
        """A subclass missing abstract methods cannot be instantiated."""

        class IncompleteStore(VectorStoreBase):
            pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteStore()  # type: ignore[abstract]

    def test_default_get_id_returns_none(self):
        """Base get_id method returns None by default."""
        # NumpyFallbackStore overrides get_id, so test the base via direct call
        assert VectorStoreBase.get_id(None, 0) is None  # type: ignore[arg-type]


# ============================================================================
# NumpyFallbackStore Tests
# ============================================================================


class TestNumpyFallbackStoreInit:
    """Tests for NumpyFallbackStore initialization."""

    def test_init_no_persist_path(self):
        """Store initializes with no persistence configured."""
        store = NumpyFallbackStore(dimension=64)
        assert store.count == 0
        assert store._persist_path is None

    def test_init_with_persist_path(self, tmp_path: Path):
        """Store initializes with persistence directory."""
        store = NumpyFallbackStore(dimension=64, persist_path=tmp_path)
        assert store.count == 0
        assert store._persist_path == tmp_path

    def test_init_loads_existing_data(self, tmp_path: Path):
        """Store loads persisted data on init if available."""
        # Save data first
        store1 = NumpyFallbackStore(dimension=64, persist_path=tmp_path)
        vecs = _make_vectors(5)
        store1.add(vecs)
        store1.save()

        # New store should load automatically
        store2 = NumpyFallbackStore(dimension=64, persist_path=tmp_path)
        assert store2.count == 5


class TestNumpyFallbackStoreAdd:
    """Tests for NumpyFallbackStore.add()."""

    def test_add_vectors_without_ids(self):
        """Add vectors with auto-generated string IDs."""
        store = NumpyFallbackStore(dimension=64)
        vecs = _make_vectors(3)
        store.add(vecs)
        assert store.count == 3
        # Auto-generated IDs are stringified indices
        assert store.get_id(0) == "0"
        assert store.get_id(1) == "1"
        assert store.get_id(2) == "2"

    def test_add_vectors_with_ids(self):
        """Add vectors with explicit chunk IDs."""
        store = NumpyFallbackStore(dimension=64)
        vecs = _make_vectors(2)
        store.add(vecs, ids=["chunk_a", "chunk_b"])
        assert store.count == 2
        assert store.get_id(0) == "chunk_a"
        assert store.get_id(1) == "chunk_b"

    def test_add_multiple_batches(self):
        """Multiple add calls accumulate vectors."""
        store = NumpyFallbackStore(dimension=64)
        store.add(_make_vectors(3))
        store.add(_make_vectors(2))
        assert store.count == 5

    def test_add_wrong_dimensionality_raises(self):
        """Adding vectors with wrong dimension raises ValueError."""
        store = NumpyFallbackStore(dimension=64)
        wrong_dim = _make_vectors(3, dim=128)
        with pytest.raises(ValueError, match="Expected vectors of shape"):
            store.add(wrong_dim)

    def test_add_1d_array_raises(self):
        """Adding a 1D array raises ValueError."""
        store = NumpyFallbackStore(dimension=64)
        with pytest.raises(ValueError, match="Expected vectors of shape"):
            store.add(RNG.standard_normal(64).astype(np.float32))


class TestNumpyFallbackStoreSearch:
    """Tests for NumpyFallbackStore.search()."""

    def test_search_empty_store(self):
        """Searching an empty store returns empty list."""
        store = NumpyFallbackStore(dimension=64)
        query = _make_query()
        results = store.search(query, k=5)
        assert results == []

    def test_search_returns_correct_top_k(self):
        """Search returns at most k results."""
        store = NumpyFallbackStore(dimension=64)
        store.add(_make_vectors(20))
        query = _make_query()
        results = store.search(query, k=5)
        assert len(results) == 5

    def test_search_k_exceeds_count(self):
        """Search with k > count returns all available vectors."""
        store = NumpyFallbackStore(dimension=64)
        store.add(_make_vectors(3))
        query = _make_query()
        results = store.search(query, k=10)
        assert len(results) == 3

    def test_search_cosine_similarity_correctness(self):
        """Known similar vector should rank highest."""
        store = NumpyFallbackStore(dimension=64)

        # Create a known query and a vector identical to it
        query = RNG.standard_normal(64).astype(np.float32)
        identical = query.copy()
        orthogonal = np.zeros(64, dtype=np.float32)
        orthogonal[0] = 1.0  # Likely different direction

        # Add: random noise, then the identical vector, then orthogonal
        noise = _make_vectors(5)
        all_vecs = np.vstack([noise, identical.reshape(1, -1), orthogonal.reshape(1, -1)])
        store.add(all_vecs)

        results = store.search(query, k=1)
        # The identical vector at index 5 should be the best match
        assert results[0][0] == 5
        # Score should be very close to 1.0
        assert results[0][1] > 0.99

    def test_search_scores_sorted_descending(self):
        """Results are sorted by score in descending order."""
        store = NumpyFallbackStore(dimension=64)
        store.add(_make_vectors(10))
        query = _make_query()
        results = store.search(query, k=10)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_scores_non_negative(self):
        """All scores are clamped to >= 0."""
        store = NumpyFallbackStore(dimension=64)
        store.add(_make_vectors(10))
        query = _make_query()
        results = store.search(query, k=10)
        for _, score in results:
            assert score >= 0.0


class TestNumpyFallbackStoreRemove:
    """Tests for NumpyFallbackStore.remove()."""

    def test_remove_by_index(self):
        """Removing a vector decreases count and updates ID mapping."""
        store = NumpyFallbackStore(dimension=64)
        store.add(_make_vectors(5), ids=["a", "b", "c", "d", "e"])
        assert store.count == 5

        store.remove([1, 3])  # Remove "b" and "d"
        assert store.count == 3
        # Remaining IDs should be remapped to new positions
        assert store.get_id(0) == "a"
        assert store.get_id(1) == "c"
        assert store.get_id(2) == "e"

    def test_remove_empty_list(self):
        """Removing empty list is a no-op."""
        store = NumpyFallbackStore(dimension=64)
        store.add(_make_vectors(3))
        store.remove([])
        assert store.count == 3

    def test_remove_all_vectors(self):
        """Removing all vectors results in empty store."""
        store = NumpyFallbackStore(dimension=64)
        store.add(_make_vectors(3))
        store.remove([0, 1, 2])
        assert store.count == 0

    def test_remove_out_of_range_index(self):
        """Out-of-range indices are silently ignored."""
        store = NumpyFallbackStore(dimension=64)
        store.add(_make_vectors(3))
        store.remove([0, 99])  # 99 is out of range
        assert store.count == 2

    def test_remove_from_empty_store(self):
        """Removing from empty store is a no-op."""
        store = NumpyFallbackStore(dimension=64)
        store.remove([0, 1])
        assert store.count == 0


class TestNumpyFallbackStorePersistence:
    """Tests for NumpyFallbackStore save/load cycle."""

    def test_save_load_roundtrip(self, tmp_path: Path):
        """Vectors and ID mapping survive save/load cycle."""
        store1 = NumpyFallbackStore(dimension=64, persist_path=tmp_path)
        vecs = _make_vectors(5)
        store1.add(vecs, ids=["x", "y", "z", "w", "v"])
        store1.save()

        store2 = NumpyFallbackStore(dimension=64, persist_path=tmp_path)
        assert store2.count == 5
        assert store2.get_id(0) == "x"
        assert store2.get_id(4) == "v"

    def test_save_no_persist_path_raises(self):
        """Saving without persist_path raises RuntimeError."""
        store = NumpyFallbackStore(dimension=64)
        with pytest.raises(RuntimeError, match="no persist_path"):
            store.save()

    def test_load_no_persist_path_returns_false(self):
        """Load returns False when no persist_path is set."""
        store = NumpyFallbackStore(dimension=64)
        assert store.load() is False

    def test_load_nonexistent_path_returns_false(self, tmp_path: Path):
        """Load returns False when files don't exist."""
        store = NumpyFallbackStore(dimension=64, persist_path=tmp_path / "nonexistent")
        assert store.load() is False

    def test_save_empty_store(self, tmp_path: Path):
        """Saving an empty store removes the vectors file if it existed."""
        store = NumpyFallbackStore(dimension=64, persist_path=tmp_path)
        store.save()
        # ID map should exist, vectors file should not
        assert (tmp_path / "numpy_id_map.json").exists()
        assert not (tmp_path / "vectors.npy").exists()

    def test_load_without_id_map_generates_sequential_ids(self, tmp_path: Path):
        """Loading vectors without an ID map generates sequential IDs."""
        store1 = NumpyFallbackStore(dimension=64, persist_path=tmp_path)
        store1.add(_make_vectors(3))
        store1.save()

        # Remove the ID map file
        (tmp_path / "numpy_id_map.json").unlink()

        store2 = NumpyFallbackStore(dimension=64, persist_path=tmp_path)
        assert store2.count == 3
        assert store2.get_id(0) == "0"
        assert store2.get_id(2) == "2"

    def test_search_after_load(self, tmp_path: Path):
        """Search works correctly on a loaded store."""
        query = RNG.standard_normal(64).astype(np.float32)

        store1 = NumpyFallbackStore(dimension=64, persist_path=tmp_path)
        vecs = np.vstack([_make_vectors(4), query.reshape(1, -1)])
        store1.add(vecs)
        store1.save()

        store2 = NumpyFallbackStore(dimension=64, persist_path=tmp_path)
        results = store2.search(query, k=1)
        assert results[0][0] == 4  # The query vector itself
        assert results[0][1] > 0.99

    def test_load_corrupted_file_returns_false(self, tmp_path: Path):
        """Loading a corrupted file returns False and resets state."""
        vectors_file = tmp_path / "vectors.npy"
        vectors_file.write_text("not a numpy file")

        store = NumpyFallbackStore(dimension=64)
        store._persist_path = tmp_path
        result = store.load()
        assert result is False
        assert store.count == 0


class TestNumpyFallbackStoreCount:
    """Tests for NumpyFallbackStore.count property."""

    def test_count_empty(self):
        """Empty store has count 0."""
        store = NumpyFallbackStore(dimension=64)
        assert store.count == 0

    def test_count_after_add(self):
        """Count reflects added vectors."""
        store = NumpyFallbackStore(dimension=64)
        store.add(_make_vectors(7))
        assert store.count == 7

    def test_count_after_remove(self):
        """Count decreases after removal."""
        store = NumpyFallbackStore(dimension=64)
        store.add(_make_vectors(5))
        store.remove([0, 2])
        assert store.count == 3


class TestNumpyFallbackStoreFromNumpy:
    """Tests for NumpyFallbackStore.from_numpy() classmethod."""

    def test_from_numpy_basic(self):
        """Create store from existing numpy array."""
        vecs = _make_vectors(10)
        store = NumpyFallbackStore.from_numpy(vecs)
        assert store.count == 10
        assert isinstance(store, NumpyFallbackStore)

    def test_from_numpy_with_persist_path(self, tmp_path: Path):
        """from_numpy accepts persist_path."""
        vecs = _make_vectors(5)
        store = NumpyFallbackStore.from_numpy(vecs, persist_path=tmp_path)
        assert store.count == 5
        assert store._persist_path == tmp_path

    def test_from_numpy_invalid_shape_raises(self):
        """1D array raises ValueError."""
        with pytest.raises(ValueError, match="Expected 2D array"):
            NumpyFallbackStore.from_numpy(RNG.standard_normal(64).astype(np.float32))


class TestNumpyFallbackStoreGetId:
    """Tests for NumpyFallbackStore.get_id()."""

    def test_get_id_valid_index(self):
        """get_id returns the mapped chunk ID."""
        store = NumpyFallbackStore(dimension=64)
        store.add(_make_vectors(3), ids=["alpha", "beta", "gamma"])
        assert store.get_id(1) == "beta"

    def test_get_id_invalid_index(self):
        """get_id returns None for unmapped index."""
        store = NumpyFallbackStore(dimension=64)
        store.add(_make_vectors(3))
        assert store.get_id(99) is None


class TestNumpyFallbackStoreRebuildIndex:
    """Tests for NumpyFallbackStore.rebuild_index()."""

    def test_rebuild_index_is_noop(self):
        """rebuild_index does nothing for NumPy store."""
        store = NumpyFallbackStore(dimension=64)
        store.add(_make_vectors(5))
        count_before = store.count
        store.rebuild_index()
        assert store.count == count_before


# ============================================================================
# FAISSVectorStore Tests (conditional)
# ============================================================================

faiss = pytest.importorskip("faiss", reason="faiss-cpu not installed")

from r3lay.core.vector_store import FAISSVectorStore  # noqa: E402


class TestFAISSVectorStoreInit:
    """Tests for FAISSVectorStore initialization."""

    def test_init_no_persist_path(self):
        """Store initializes with no persistence configured."""
        store = FAISSVectorStore(dimension=64)
        assert store.count == 0
        assert store._persist_path is None

    def test_init_with_persist_path(self, tmp_path: Path):
        """Store initializes with persistence directory."""
        store = FAISSVectorStore(dimension=64, persist_path=tmp_path)
        assert store.count == 0
        assert store._persist_path == tmp_path

    def test_init_loads_existing_data(self, tmp_path: Path):
        """Store loads persisted data on init."""
        store1 = FAISSVectorStore(dimension=64, persist_path=tmp_path)
        store1.add(_make_vectors(5))
        store1.save()

        store2 = FAISSVectorStore(dimension=64, persist_path=tmp_path)
        assert store2.count == 5


class TestFAISSVectorStoreAdd:
    """Tests for FAISSVectorStore.add()."""

    def test_add_vectors_without_ids(self):
        """Add vectors with auto-generated string IDs."""
        store = FAISSVectorStore(dimension=64)
        store.add(_make_vectors(3))
        assert store.count == 3
        assert store.get_id(0) == "0"
        assert store.get_id(2) == "2"

    def test_add_vectors_with_ids(self):
        """Add vectors with explicit chunk IDs."""
        store = FAISSVectorStore(dimension=64)
        store.add(_make_vectors(2), ids=["chunk_a", "chunk_b"])
        assert store.count == 2
        assert store.get_id(0) == "chunk_a"
        assert store.get_id(1) == "chunk_b"

    def test_add_multiple_batches(self):
        """Multiple add calls accumulate vectors."""
        store = FAISSVectorStore(dimension=64)
        store.add(_make_vectors(3))
        store.add(_make_vectors(2))
        assert store.count == 5

    def test_add_wrong_dimensionality_raises(self):
        """Adding vectors with wrong dimension raises ValueError."""
        store = FAISSVectorStore(dimension=64)
        with pytest.raises(ValueError, match="Expected vectors of shape"):
            store.add(_make_vectors(3, dim=128))

    def test_add_1d_array_raises(self):
        """Adding a 1D array raises ValueError."""
        store = FAISSVectorStore(dimension=64)
        with pytest.raises(ValueError, match="Expected vectors of shape"):
            store.add(RNG.standard_normal(64).astype(np.float32))


class TestFAISSVectorStoreSearch:
    """Tests for FAISSVectorStore.search()."""

    def test_search_empty_store(self):
        """Searching an empty store returns empty list."""
        store = FAISSVectorStore(dimension=64)
        results = store.search(_make_query(), k=5)
        assert results == []

    def test_search_returns_correct_top_k(self):
        """Search returns at most k results."""
        store = FAISSVectorStore(dimension=64)
        store.add(_make_vectors(20))
        results = store.search(_make_query(), k=5)
        assert len(results) == 5

    def test_search_k_exceeds_count(self):
        """Search with k > count returns all available vectors."""
        store = FAISSVectorStore(dimension=64)
        store.add(_make_vectors(3))
        results = store.search(_make_query(), k=10)
        assert len(results) == 3

    def test_search_cosine_similarity_correctness(self):
        """Known similar vector should rank highest."""
        store = FAISSVectorStore(dimension=64)
        query = RNG.standard_normal(64).astype(np.float32)
        noise = _make_vectors(5)
        all_vecs = np.vstack([noise, query.reshape(1, -1)])
        store.add(all_vecs)

        results = store.search(query, k=1)
        assert results[0][0] == 5
        assert results[0][1] > 0.99

    def test_search_scores_sorted_descending(self):
        """Results are sorted by score in descending order."""
        store = FAISSVectorStore(dimension=64)
        store.add(_make_vectors(10))
        results = store.search(_make_query(), k=10)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_scores_clamped_0_to_1(self):
        """All scores are clamped to [0, 1] range."""
        store = FAISSVectorStore(dimension=64)
        store.add(_make_vectors(10))
        results = store.search(_make_query(), k=10)
        for _, score in results:
            assert 0.0 <= score <= 1.0


class TestFAISSVectorStoreRemove:
    """Tests for FAISSVectorStore.remove()."""

    def test_remove_by_index(self):
        """Removing vectors decreases count and updates ID mapping."""
        store = FAISSVectorStore(dimension=64)
        store.add(_make_vectors(5), ids=["a", "b", "c", "d", "e"])
        store.remove([1, 3])
        assert store.count == 3
        assert store.get_id(0) == "a"
        assert store.get_id(1) == "c"
        assert store.get_id(2) == "e"

    def test_remove_empty_list(self):
        """Removing empty list is a no-op."""
        store = FAISSVectorStore(dimension=64)
        store.add(_make_vectors(3))
        store.remove([])
        assert store.count == 3

    def test_remove_all_vectors(self):
        """Removing all vectors results in empty store."""
        store = FAISSVectorStore(dimension=64)
        store.add(_make_vectors(3))
        store.remove([0, 1, 2])
        assert store.count == 0

    def test_remove_from_empty_store(self):
        """Removing from empty store is a no-op."""
        store = FAISSVectorStore(dimension=64)
        store.remove([0, 1])
        assert store.count == 0


class TestFAISSVectorStorePersistence:
    """Tests for FAISSVectorStore save/load cycle."""

    def test_save_load_roundtrip(self, tmp_path: Path):
        """Vectors and ID mapping survive save/load cycle."""
        store1 = FAISSVectorStore(dimension=64, persist_path=tmp_path)
        store1.add(_make_vectors(5), ids=["a", "b", "c", "d", "e"])
        store1.save()

        store2 = FAISSVectorStore(dimension=64, persist_path=tmp_path)
        assert store2.count == 5
        assert store2.get_id(0) == "a"
        assert store2.get_id(4) == "e"

    def test_save_no_persist_path_raises(self):
        """Saving without persist_path raises RuntimeError."""
        store = FAISSVectorStore(dimension=64)
        with pytest.raises(RuntimeError, match="no persist_path"):
            store.save()

    def test_load_no_persist_path_returns_false(self):
        """Load returns False when no persist_path is set."""
        store = FAISSVectorStore(dimension=64)
        assert store.load() is False

    def test_load_nonexistent_path_returns_false(self, tmp_path: Path):
        """Load returns False when index file doesn't exist."""
        store = FAISSVectorStore(dimension=64, persist_path=tmp_path / "nonexistent")
        assert store.load() is False

    def test_search_after_load(self, tmp_path: Path):
        """Search works correctly on a loaded store."""
        query = RNG.standard_normal(64).astype(np.float32)

        store1 = FAISSVectorStore(dimension=64, persist_path=tmp_path)
        vecs = np.vstack([_make_vectors(4), query.reshape(1, -1)])
        store1.add(vecs)
        store1.save()

        store2 = FAISSVectorStore(dimension=64, persist_path=tmp_path)
        results = store2.search(query, k=1)
        assert results[0][0] == 4
        assert results[0][1] > 0.99

    def test_load_without_id_map_generates_sequential_ids(self, tmp_path: Path):
        """Loading without ID map file generates sequential IDs."""
        store1 = FAISSVectorStore(dimension=64, persist_path=tmp_path)
        store1.add(_make_vectors(3))
        store1.save()

        (tmp_path / "faiss_id_map.json").unlink()

        store2 = FAISSVectorStore(dimension=64, persist_path=tmp_path)
        assert store2.count == 3
        assert store2.get_id(0) == "0"
        assert store2.get_id(2) == "2"

    def test_load_corrupted_index_returns_false(self, tmp_path: Path):
        """Loading a corrupted index file returns False and resets state."""
        index_file = tmp_path / "faiss.index"
        index_file.write_text("not a faiss index")

        store = FAISSVectorStore(dimension=64)
        store._persist_path = tmp_path
        result = store.load()
        assert result is False
        assert store.count == 0


class TestFAISSVectorStoreRebuild:
    """Tests for FAISSVectorStore.rebuild_index() and IVF transition."""

    def test_rebuild_below_threshold_stays_flat(self):
        """Rebuild below IVF_THRESHOLD keeps flat index."""
        store = FAISSVectorStore(dimension=64)
        store.add(_make_vectors(100))
        store.rebuild_index()
        assert store._is_ivf is False
        assert store.count == 100

    def test_rebuild_empty_store_is_noop(self):
        """Rebuild on empty store does nothing."""
        store = FAISSVectorStore(dimension=64)
        store.rebuild_index()
        assert store.count == 0

    def test_ivf_rebuild_when_threshold_exceeded(self):
        """Index switches to IVF when count >= IVF_THRESHOLD."""
        store = FAISSVectorStore(dimension=64)

        # Use a small threshold to avoid creating 100k vectors
        with patch("r3lay.core.vector_store.IVF_THRESHOLD", 50):
            store.add(_make_vectors(60))
            store.rebuild_index()
            # The rebuild reads the patched module-level constant, but the method
            # uses the local IVF_THRESHOLD. Let's check the internal rebuild logic.

        # Since patch only changes the module-level constant and rebuild_index
        # reads it at call time, we need to patch during the rebuild call.
        store2 = FAISSVectorStore(dimension=64)
        store2.add(_make_vectors(60))
        with patch.object(
            type(store2),
            "rebuild_index",
            wraps=store2.rebuild_index,
        ):
            # Direct approach: monkey-patch the threshold in the module
            import r3lay.core.vector_store as vs_mod

            original_threshold = vs_mod.IVF_THRESHOLD
            try:
                vs_mod.IVF_THRESHOLD = 50
                store2.rebuild_index()
                assert store2._is_ivf is True
                assert store2.count == 60
                # Search should still work on IVF index
                results = store2.search(_make_query(), k=5)
                assert len(results) == 5
            finally:
                vs_mod.IVF_THRESHOLD = original_threshold

    def test_rebuild_preserves_search_quality(self):
        """Rebuild does not degrade search results for flat index."""
        store = FAISSVectorStore(dimension=64)
        query = RNG.standard_normal(64).astype(np.float32)
        vecs = np.vstack([_make_vectors(9), query.reshape(1, -1)])
        store.add(vecs)

        results_before = store.search(query, k=1)
        store.rebuild_index()
        results_after = store.search(query, k=1)

        assert results_before[0][0] == results_after[0][0]


class TestFAISSVectorStoreThreadSafety:
    """Tests for FAISSVectorStore thread safety under concurrent adds."""

    def test_concurrent_adds_do_not_corrupt(self):
        """Concurrent add calls produce correct total count."""
        store = FAISSVectorStore(dimension=64)
        batches = [_make_vectors(10) for _ in range(8)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(store.add, batch) for batch in batches]
            concurrent.futures.wait(futures)
            # Check no exceptions
            for f in futures:
                f.result()

        assert store.count == 80


class TestFAISSVectorStoreFromNumpy:
    """Tests for FAISSVectorStore.from_numpy() classmethod."""

    def test_from_numpy_basic(self):
        """Create FAISS store from numpy array."""
        vecs = _make_vectors(10)
        store = FAISSVectorStore.from_numpy(vecs)
        assert store.count == 10
        assert isinstance(store, FAISSVectorStore)

    def test_from_numpy_with_persist_path(self, tmp_path: Path):
        """from_numpy accepts persist_path."""
        vecs = _make_vectors(5)
        store = FAISSVectorStore.from_numpy(vecs, persist_path=tmp_path)
        assert store.count == 5

    def test_from_numpy_invalid_shape_raises(self):
        """1D array raises ValueError."""
        with pytest.raises(ValueError, match="Expected 2D array"):
            FAISSVectorStore.from_numpy(RNG.standard_normal(64).astype(np.float32))


class TestFAISSVectorStoreGetId:
    """Tests for FAISSVectorStore.get_id()."""

    def test_get_id_valid(self):
        """get_id returns mapped chunk ID."""
        store = FAISSVectorStore(dimension=64)
        store.add(_make_vectors(2), ids=["alpha", "beta"])
        assert store.get_id(0) == "alpha"
        assert store.get_id(1) == "beta"

    def test_get_id_invalid(self):
        """get_id returns None for unmapped index."""
        store = FAISSVectorStore(dimension=64)
        store.add(_make_vectors(2))
        assert store.get_id(99) is None


# ============================================================================
# create_vector_store Factory Tests
# ============================================================================


class TestCreateVectorStore:
    """Tests for create_vector_store factory function."""

    def test_returns_faiss_when_available(self):
        """Factory returns FAISSVectorStore when faiss is available."""
        with patch("r3lay.core.vector_store.FAISS_AVAILABLE", True):
            store = create_vector_store(dimension=64)
            assert isinstance(store, FAISSVectorStore)

    def test_returns_numpy_when_faiss_unavailable(self):
        """Factory returns NumpyFallbackStore when faiss is not available."""
        with patch("r3lay.core.vector_store.FAISS_AVAILABLE", False):
            store = create_vector_store(dimension=64)
            assert isinstance(store, NumpyFallbackStore)

    def test_force_numpy_always_returns_numpy(self):
        """force_numpy=True returns NumpyFallbackStore regardless of faiss."""
        with patch("r3lay.core.vector_store.FAISS_AVAILABLE", True):
            store = create_vector_store(dimension=64, force_numpy=True)
            assert isinstance(store, NumpyFallbackStore)

    def test_persist_path_forwarded(self, tmp_path: Path):
        """persist_path is forwarded to the created store."""
        store = create_vector_store(dimension=64, persist_path=tmp_path, force_numpy=True)
        assert store._persist_path == tmp_path


# ============================================================================
# Migration Tests
# ============================================================================


class TestMigration:
    """Tests for from_numpy migration with real vector search."""

    def test_numpy_to_numpy_migration_preserves_search(self):
        """Migrating NumPy vectors to NumpyFallbackStore preserves search quality."""
        query = RNG.standard_normal(64).astype(np.float32)
        vecs = np.vstack([_make_vectors(9), query.reshape(1, -1)])

        store = NumpyFallbackStore.from_numpy(vecs)
        results = store.search(query, k=1)
        assert results[0][0] == 9
        assert results[0][1] > 0.99

    def test_numpy_to_faiss_migration_preserves_search(self):
        """Migrating NumPy vectors to FAISSVectorStore preserves search quality."""
        query = RNG.standard_normal(64).astype(np.float32)
        vecs = np.vstack([_make_vectors(9), query.reshape(1, -1)])

        store = FAISSVectorStore.from_numpy(vecs)
        results = store.search(query, k=1)
        assert results[0][0] == 9
        assert results[0][1] > 0.99

    def test_migration_with_persistence_roundtrip(self, tmp_path: Path):
        """Migrated store can be saved and loaded with correct search results."""
        query = RNG.standard_normal(64).astype(np.float32)
        vecs = np.vstack([_make_vectors(9), query.reshape(1, -1)])

        store1 = FAISSVectorStore.from_numpy(vecs, persist_path=tmp_path)
        store1.save()

        store2 = FAISSVectorStore(dimension=64, persist_path=tmp_path)
        results = store2.search(query, k=1)
        assert results[0][0] == 9
        assert results[0][1] > 0.99
