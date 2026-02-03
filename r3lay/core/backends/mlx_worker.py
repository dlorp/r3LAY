#!/usr/bin/env python3
"""MLX worker process for isolated model inference.

This module runs as a standalone subprocess to completely isolate mlx-lm's
terminal interactions from Textual's TUI.

Communication happens via JSON lines over stdin/stdout:
- stdin: JSON commands (one per line)
- stdout: JSON responses (one per line)
- stderr: redirected to /dev/null

Commands:
  {"cmd": "load", "path": "/path/to/model", "is_vision": false}
  {"cmd": "generate", "messages": [...], "max_tokens": 512, "temperature": 0.7, "images": []}
  {"cmd": "stop"}
  {"cmd": "unload"}

Responses:
  {"type": "loaded", "success": true, "is_vlm": false}
  {"type": "loaded", "success": false, "error": "message"}
  {"type": "token", "text": "..."}
  {"type": "done"}
  {"type": "error", "message": "..."}

Usage:
    python -m r3lay.core.backends.mlx_worker

Supports both text models (mlx-lm) and vision models (mlx-vlm).
"""

from __future__ import annotations

import gc
import json
import os
import sys
from typing import Any


def setup_isolation() -> None:
    """Set up environment isolation before any imports.

    MUST be called before importing mlx_lm, transformers, etc.
    """
    # Disable terminal features
    os.environ["TERM"] = "dumb"
    os.environ["NO_COLOR"] = "1"
    os.environ["FORCE_COLOR"] = "0"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["MLX_SHOW_PROGRESS"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # Redirect stderr to /dev/null (keep stdout for JSON responses)
    sys.stderr = open(os.devnull, "w")


def send_response(response: dict) -> None:
    """Send a JSON response to stdout."""
    print(json.dumps(response), flush=True)


def format_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Format messages using tokenizer's chat template."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    # Fallback format
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"System: {content}\n")
        elif role == "user":
            parts.append(f"User: {content}\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}\n")
    parts.append("Assistant: ")
    return "".join(parts)


def main() -> None:
    """Main worker loop."""
    # Set up isolation FIRST
    setup_isolation()

    # Text model state (mlx-lm)
    model = None
    tokenizer = None
    stream_generate = None
    stop_requested = False

    # Vision model state (mlx-vlm)
    is_vlm = False
    vlm_generate = None
    vlm_processor = None

    # Read commands from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            send_response({"type": "error", "message": f"Invalid JSON: {e}"})
            continue

        cmd_type = cmd.get("cmd")

        if cmd_type == "load":
            model_path = cmd.get("path")
            is_vision = cmd.get("is_vision", False)  # Hint from backend

            try:
                if is_vision:
                    # Try loading as VLM using mlx-vlm
                    try:
                        from mlx_vlm import generate as vlm_gen
                        from mlx_vlm import load as vlm_load

                        model, vlm_processor = vlm_load(model_path)
                        vlm_generate = vlm_gen
                        is_vlm = True
                        tokenizer = None  # VLMs use processor instead
                        stream_generate = None
                        send_response({"type": "loaded", "success": True, "is_vlm": True})
                    except ImportError:
                        send_response(
                            {
                                "type": "loaded",
                                "success": False,
                                "error": "mlx-vlm not installed for vision models",
                            }
                        )
                        continue
                else:
                    # Load as text model using mlx-lm
                    from mlx_lm import load
                    from mlx_lm import stream_generate as sg

                    stream_generate = sg
                    is_vlm = False
                    vlm_generate = None
                    vlm_processor = None

                    model, tokenizer = load(model_path)
                    send_response({"type": "loaded", "success": True, "is_vlm": False})

            except ImportError as e:
                send_response(
                    {"type": "loaded", "success": False, "error": f"mlx-lm not installed: {e}"}
                )
            except Exception as e:
                send_response({"type": "loaded", "success": False, "error": f"Failed to load: {e}"})

        elif cmd_type == "generate":
            if model is None:
                send_response({"type": "error", "message": "Model not loaded"})
                continue

            messages = cmd.get("messages", [])
            max_tokens = cmd.get("max_tokens", 512)
            temperature = cmd.get("temperature", 0.7)
            images = cmd.get("images", [])  # List of image paths for VLMs
            stop_requested = False

            try:
                if is_vlm and vlm_generate is not None:
                    # VLM generation with images
                    from PIL import Image

                    # Get the last user message for the prompt
                    prompt = ""
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            prompt = msg.get("content", "")
                            break

                    # Load images
                    loaded_images = []
                    for img_path in images:
                        try:
                            img = Image.open(img_path)
                            loaded_images.append(img)
                        except Exception as e:
                            send_response(
                                {
                                    "type": "error",
                                    "message": f"Failed to load image {img_path}: {e}",
                                }
                            )
                            continue

                    # Generate with VLM (mlx-vlm uses different API)
                    # Note: mlx-vlm's generate is not streaming, so we send output at once
                    if loaded_images:
                        output = vlm_generate(
                            model,
                            vlm_processor,
                            loaded_images[0],  # mlx-vlm takes single image
                            prompt,
                            max_tokens=max_tokens,
                            temp=temperature,
                        )
                        send_response({"type": "token", "text": output})
                    else:
                        # No images provided, just use the prompt
                        output = vlm_generate(
                            model,
                            vlm_processor,
                            prompt,
                            max_tokens=max_tokens,
                            temp=temperature,
                        )
                        send_response({"type": "token", "text": output})

                    send_response({"type": "done"})
                else:
                    # Text-only generation using mlx-lm
                    if tokenizer is None:
                        send_response({"type": "error", "message": "Tokenizer not loaded"})
                        continue

                    # Format messages
                    prompt = format_messages(tokenizer, messages)

                    # Build generation kwargs
                    gen_kwargs = {"prompt": prompt, "max_tokens": max_tokens}

                    # Try new mlx-lm API (>= 0.30)
                    try:
                        from mlx_lm.sample_utils import make_sampler

                        gen_kwargs["sampler"] = make_sampler(temperature)
                    except ImportError:
                        gen_kwargs["temp"] = temperature

                    # Stream tokens
                    if stream_generate is None:
                        send_response({"type": "error", "message": "stream_generate not loaded"})
                        continue
                    for response in stream_generate(model, tokenizer, **gen_kwargs):
                        if stop_requested:
                            break

                        # Handle different mlx-lm response formats
                        if hasattr(response, "text"):
                            text = response.text
                        elif isinstance(response, tuple) and len(response) >= 1:
                            text = response[0]
                        else:
                            text = str(response)

                        if text:
                            send_response({"type": "token", "text": text})

                    send_response({"type": "done"})

            except Exception as e:
                send_response({"type": "error", "message": f"Generation error: {e}"})

        elif cmd_type == "stop":
            stop_requested = True

        elif cmd_type == "unload":
            # Cleanup and exit
            if model is not None:
                try:
                    import mlx.core as mx

                    # Clean up model
                    del model

                    # Clean up text model resources
                    if tokenizer is not None:
                        del tokenizer

                    # Clean up VLM resources
                    if vlm_processor is not None:
                        del vlm_processor

                    gc.collect()

                    # Clear Metal cache with force sync
                    if hasattr(mx.metal, "clear_cache"):
                        mx.metal.clear_cache()
                        mx.eval(mx.zeros(1))  # Force sync
                        mx.metal.clear_cache()
                    elif hasattr(mx, "clear_cache"):
                        mx.clear_cache()
                except Exception:
                    pass
            break

        else:
            send_response({"type": "error", "message": f"Unknown command: {cmd_type}"})


if __name__ == "__main__":
    main()
