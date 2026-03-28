"""Tests for adaptive retrieval strategy classification."""

from __future__ import annotations

import pytest

from r3lay.core.index import RetrievalStrategy, classify_query


class TestRetrievalStrategy:
    """Test the RetrievalStrategy enum."""

    def test_enum_values(self) -> None:
        assert RetrievalStrategy.NO_RETRIEVAL.value == "no_retrieval"
        assert RetrievalStrategy.BM25_ONLY.value == "bm25_only"
        assert RetrievalStrategy.HYBRID.value == "hybrid"
        assert RetrievalStrategy.HYBRID_RERANK.value == "hybrid_rerank"

    def test_all_strategies_exist(self) -> None:
        assert len(RetrievalStrategy) == 4


class TestClassifyQuery:
    """Test query classification heuristics."""

    # --- NO_RETRIEVAL: greetings and short non-queries ---

    @pytest.mark.parametrize(
        "query",
        [
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "ok",
            "okay",
            "sure",
            "yes",
            "no",
            "bye",
            "goodbye",
            "Hi there",
            "Hello!",
        ],
    )
    def test_greetings_skip_retrieval(self, query: str) -> None:
        result = classify_query(query, has_hybrid=True, has_reranker=True)
        assert result == RetrievalStrategy.NO_RETRIEVAL

    def test_long_greeting_not_skipped(self) -> None:
        # "hello how are you" is 4 words, not a simple greeting
        result = classify_query("hello how are you", has_hybrid=True)
        assert result != RetrievalStrategy.NO_RETRIEVAL

    # --- BM25_ONLY: short keywords without hybrid ---

    def test_short_keyword_no_hybrid(self) -> None:
        result = classify_query("python logging", has_hybrid=False)
        assert result == RetrievalStrategy.BM25_ONLY

    def test_short_keyword_with_hybrid(self) -> None:
        # Short keyword queries use BM25-only even when hybrid is available
        result = classify_query("python logging", has_hybrid=True)
        assert result == RetrievalStrategy.BM25_ONLY

    def test_question_forces_hybrid_when_available(self) -> None:
        result = classify_query("what is logging?", has_hybrid=True)
        assert result == RetrievalStrategy.HYBRID

    def test_no_hybrid_always_bm25(self) -> None:
        result = classify_query("explain the logging architecture in detail", has_hybrid=False)
        assert result == RetrievalStrategy.BM25_ONLY

    # --- HYBRID: standard queries with vector search ---

    def test_standard_question_hybrid(self) -> None:
        result = classify_query("how does authentication work", has_hybrid=True)
        assert result == RetrievalStrategy.HYBRID

    def test_medium_query_hybrid(self) -> None:
        result = classify_query("find the database connection", has_hybrid=True)
        assert result == RetrievalStrategy.HYBRID

    # --- HYBRID_RERANK: complex queries ---

    def test_long_query_triggers_rerank(self) -> None:
        result = classify_query(
            "how does the hybrid search index combine BM25 and vector results using RRF",
            has_hybrid=True,
            has_reranker=True,
        )
        assert result == RetrievalStrategy.HYBRID_RERANK

    def test_code_tokens_trigger_rerank(self) -> None:
        result = classify_query("find getUserProfile method", has_hybrid=True, has_reranker=True)
        assert result == RetrievalStrategy.HYBRID_RERANK

    def test_snake_case_triggers_rerank(self) -> None:
        result = classify_query(
            "where is _load_from_disk called", has_hybrid=True, has_reranker=True
        )
        assert result == RetrievalStrategy.HYBRID_RERANK

    def test_question_with_enough_words_triggers_rerank(self) -> None:
        result = classify_query(
            "what is the purpose of the token budget",
            has_hybrid=True,
            has_reranker=True,
        )
        assert result == RetrievalStrategy.HYBRID_RERANK

    def test_complex_query_falls_back_to_hybrid_without_reranker(self) -> None:
        result = classify_query(
            "how does the hybrid search index combine BM25 and vector results using RRF",
            has_hybrid=True,
            has_reranker=False,
        )
        assert result == RetrievalStrategy.HYBRID

    # --- Edge cases ---

    def test_empty_query(self) -> None:
        result = classify_query("", has_hybrid=True)
        # Empty string: 1 word, no question, no code tokens -> BM25_ONLY
        assert result == RetrievalStrategy.BM25_ONLY

    def test_single_word_non_greeting(self) -> None:
        result = classify_query("python", has_hybrid=False)
        assert result == RetrievalStrategy.BM25_ONLY

    def test_whitespace_only(self) -> None:
        result = classify_query("   ", has_hybrid=True)
        # "   ".strip() = "", 1 word, no question, no code tokens -> BM25_ONLY
        assert result == RetrievalStrategy.BM25_ONLY

    def test_question_mark_triggers_question_detection(self) -> None:
        # "is this correct?" has a question mark
        result = classify_query("is this correct?", has_hybrid=True)
        assert result == RetrievalStrategy.HYBRID

    def test_all_capabilities_available(self) -> None:
        # Complex query with all capabilities should use HYBRID_RERANK
        result = classify_query(
            "explain how the retrospective revision loop detects contradictions",
            has_hybrid=True,
            has_reranker=True,
        )
        assert result == RetrievalStrategy.HYBRID_RERANK

    def test_defaults_no_capabilities(self) -> None:
        # With no hybrid and no reranker, should be BM25_ONLY
        result = classify_query("complex search query with many words")
        assert result == RetrievalStrategy.BM25_ONLY
