"""Index panel - RAG index status and controls.

Phase D: Added vision embedding support for PDF/image indexing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Label, Static

if TYPE_CHECKING:
    from ...core import R3LayState

logger = logging.getLogger(__name__)


class IndexPanel(Vertical):
    """Panel for RAG index status and controls.

    Displays:
    - Knowledge base header
    - Index statistics (chunks, collection, hybrid status)
    - Progress indicator during indexing
    - Reindex and Clear buttons

    Features:
    - Lazy loads embedder only when reindexing
    - Falls back to BM25-only if embedding dependencies unavailable
    - Shows hybrid status in stats

    Keybindings:
    - Ctrl+R triggers reindexing from anywhere in the app
    """

    DEFAULT_CSS = """
    IndexPanel {
        width: 100%;
        height: 100%;
        padding: 1;
    }
    #index-header { height: 1; text-style: bold; }
    #index-stats { height: auto; padding: 1; background: $surface-darken-1; margin: 1 0; }
    #index-progress { height: auto; color: $primary; margin-bottom: 1; }
    #button-row { height: 3; width: 100%; }
    #reindex-button { width: 1fr; margin-right: 1; }
    #clear-button { width: 1fr; }
    """

    def __init__(self, state: "R3LayState"):
        super().__init__()
        self.state = state
        self._is_indexing = False
        self._embedder_loaded = False

    def compose(self) -> ComposeResult:
        yield Label("Knowledge Base", id="index-header")
        yield Static("Not indexed", id="index-stats")
        yield Static("", id="index-progress")
        with Horizontal(id="button-row"):
            yield Button("Reindex", id="reindex-button", variant="primary")
            yield Button("Clear", id="clear-button", variant="warning")

    def on_mount(self) -> None:
        self._refresh_stats()

    def _refresh_stats(self) -> None:
        """Update stats display from current index state."""
        from ...core import embeddings_available, vision_embeddings_available, pdf_extraction_available

        stats_widget = self.query_one("#index-stats", Static)
        if self.state.index is None:
            # Check if embeddings could be available
            text_embed = "yes" if embeddings_available() else "no"
            vision_embed = "yes" if vision_embeddings_available() else "no"
            pdf_extract = "yes" if pdf_extraction_available() else "no"
            stats_widget.update(
                f"Index not initialized\n"
                f"Click Reindex to build\n"
                f"Text embed: {text_embed} | Vision: {vision_embed} | PDF: {pdf_extract}"
            )
            return
        try:
            stats = self.state.index.get_stats()

            # Get chunk counts (text vs image)
            text_count = stats.get("count", 0)
            image_count = stats.get("image_count", 0)

            # Build chunk count string
            if image_count > 0:
                chunk_str = f"Chunks: {text_count} (text) + {image_count} (images)"
            else:
                chunk_str = f"Chunks: {text_count}"

            # Determine hybrid status string
            if stats["hybrid_enabled"]:
                hybrid_status = "Enabled"
            elif stats["vectors_count"] > 0:
                hybrid_status = "Vectors ready (embedder not loaded)"
            elif embeddings_available():
                hybrid_status = "Disabled (no vectors)"
            else:
                hybrid_status = "Unavailable (no dependencies)"

            # Build stats display
            lines = [
                chunk_str,
                f"Collection: {stats['collection']}",
                f"Hybrid: {hybrid_status}",
            ]

            # Add vector details if available
            text_vectors = stats.get("vectors_count", 0)
            image_vectors = stats.get("image_vectors_count", 0)
            text_dim = stats.get("embedding_dim", 0)
            image_dim = stats.get("image_embedding_dim", 0)

            if text_vectors > 0 or image_vectors > 0:
                vector_parts = []
                if text_vectors > 0:
                    vector_parts.append(f"{text_vectors} text (dim={text_dim})")
                if image_vectors > 0:
                    vector_parts.append(f"{image_vectors} image (dim={image_dim})")
                lines.append(f"Vectors: {' + '.join(vector_parts)}")

            stats_widget.update("\n".join(lines))
        except Exception as e:
            stats_widget.update(f"Error: {e}")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "reindex-button":
            if not self._is_indexing:
                await self._do_reindex_sync()
        elif event.button.id == "clear-button":
            if self.state.index is not None:
                self.state.index.clear()
                self._refresh_stats()
                self.app.notify("Index cleared")

    def _update_progress(self, text: str) -> None:
        """Update progress indicator."""
        self.query_one("#index-progress", Static).update(text)

    def _update_button(self, label: str, disabled: bool) -> None:
        """Update reindex button state."""
        btn = self.query_one("#reindex-button", Button)
        btn.label = label
        btn.disabled = disabled

    async def _do_reindex_sync(self) -> None:
        """Reindex with optional embedding generation.

        Runs chunking on main thread, then generates embeddings if available.
        Uses refresh() to update UI between operations.

        Steps:
        1. Initialize index
        2. Load and chunk documents + collect image paths (BM25 indexing)
        3. If text embeddings available, load embedder and generate vectors
        4. If images found and vision embedder available, generate image embeddings
        """
        from ...core import embeddings_available, vision_embeddings_available, pdf_extraction_available
        from ...core.index import DocumentLoader, LoadResult

        self._is_indexing = True
        self._update_button("Indexing...", True)

        try:
            # Initialize index
            self._update_progress("Initializing index...")
            self.refresh()

            idx = self.state.init_index()
            idx.clear()

            # Load documents from project directory (text + images + PDFs)
            self._update_progress("Loading documents...")
            self.refresh()

            loader = DocumentLoader(index=idx)  # Pass index for PDF extraction
            result: LoadResult = loader.load_directory_with_images(
                self.state.project_path,
                recursive=True,
                include_pdfs=pdf_extraction_available(),
            )

            # Add text chunks to index (BM25 indexing)
            self._update_progress(f"Indexing {len(result.chunks)} text chunks...")
            self.refresh()

            count = idx.add_chunks(result.chunks)

            # Generate text embeddings if available
            text_embeddings_generated = False
            if count > 0 and embeddings_available():
                self._update_progress("Loading text embedding model...")
                self.refresh()

                try:
                    # Lazy load embedder
                    embedder = await self.state.init_embedder()

                    if embedder is not None:
                        # Attach embedder to index
                        idx.text_embedder = embedder
                        self._embedder_loaded = True

                        self._update_progress(f"Generating text embeddings for {count} chunks...")
                        self.refresh()

                        # Generate embeddings
                        num_embeddings = await idx.generate_embeddings()
                        text_embeddings_generated = True

                        logger.info(f"Generated {num_embeddings} text embeddings")

                except Exception as e:
                    # Log but don't fail - BM25 still works
                    logger.warning(f"Text embedding generation failed: {e}")
                    self._update_progress(f"Text embeddings failed: {e}")
                    self.refresh()
                    import asyncio
                    await asyncio.sleep(1)  # Show error briefly

            # Generate vision embeddings for images if available
            image_count = len(result.image_paths)
            vision_embeddings_generated = False

            if image_count > 0:
                self._update_progress(f"Found {image_count} images...")
                self.refresh()

                try:
                    # Try to get vision embedder from state or load it
                    vision_embedder = self.state.vision_embedder
                    if vision_embedder is None:
                        vision_embedder_config = self._get_vision_embedder_config()
                        if vision_embeddings_available():
                            self._update_progress("Loading vision embedding model...")
                            self.refresh()
                            vision_embedder = await self.state.init_vision_embedder(
                                model_name=vision_embedder_config
                            )

                    if vision_embedder is not None and vision_embedder.is_loaded:
                        # Attach vision embedder to index
                        idx.vision_embedder = vision_embedder

                        self._update_progress(f"Embedding {image_count} images...")
                        self.refresh()

                        # Actually call add_images() to generate embeddings
                        images_added = await idx.add_images(
                            result.image_paths,
                            result.image_metadata,
                        )
                        vision_embeddings_generated = True
                        logger.info(f"Added {images_added} images with vision embeddings")
                    else:
                        self._update_progress(f"Found {image_count} images (vision embedder not available)")
                        self.refresh()
                        logger.info(f"Found {image_count} images but no vision embedder available")

                except Exception as e:
                    logger.warning(f"Vision embedding failed: {e}")
                    self._update_progress(f"Vision embeddings failed: {e}")
                    self.refresh()
                    import asyncio
                    await asyncio.sleep(1)

            # Final status
            status_parts = []
            if text_embeddings_generated:
                status_parts.append(f"{count} text chunks with embeddings")
            else:
                status_parts.append(f"{count} text chunks (BM25 only)")

            if image_count > 0:
                if vision_embeddings_generated:
                    status_parts.append(f"{image_count} images embedded")
                else:
                    status_parts.append(f"{image_count} images (no embeddings)")

            final_status = "Indexed: " + ", ".join(status_parts)
            self._update_progress(final_status)

            if text_embeddings_generated or vision_embeddings_generated:
                self.app.notify(f"Indexed {count} chunks (hybrid search enabled)")
            else:
                self.app.notify(f"Indexed {count} chunks")

        except Exception as e:
            logger.exception("Reindex failed")
            self._update_progress(f"Error: {e}")
        finally:
            self._is_indexing = False
            self._update_button("Reindex", False)
            self._refresh_stats()
            self._refresh_welcome_message()

    def _get_vision_embedder_config(self) -> str | None:
        """Get the configured vision embedder model name from app config."""
        try:
            app = self.app
            if hasattr(app, "config") and hasattr(app.config, "model_roles"):
                return app.config.model_roles.vision_embedder
        except Exception:
            pass
        return None

    def _refresh_welcome_message(self) -> None:
        """Refresh the welcome message in ResponsePane to reflect index state."""
        try:
            response_pane = self.screen.query_one("ResponsePane")
            if hasattr(response_pane, "refresh_welcome"):
                response_pane.refresh_welcome()
        except Exception:
            pass  # Best effort - ResponsePane may not exist


__all__ = ["IndexPanel"]
