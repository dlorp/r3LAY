#!/usr/bin/env python3
"""MLX worker process for isolated model inference.

Runs as a standalone subprocess. Communicates via JSON lines over stdin/stdout.

Commands:
  {"cmd": "load", "path": "/path/to/model", "is_vision": false}
  {"cmd": "generate", "messages": [...], "max_tokens": 512, "temperature": 0.7}
  {"cmd": "stop"}
  {"cmd": "unload"}

Responses:
  {"type": "loaded", "success": true}
  {"type": "token", "text": "..."}
  {"type": "done"}
  {"type": "error", "message": "..."}
"""

from __future__ import annotations

import gc
import json
import os
import sys
from typing import Any


def setup_isolation() -> None:
    """Set up environment isolation before any imports."""
    os.environ["TERM"] = "dumb"
    os.environ["NO_COLOR"] = "1"
    os.environ["FORCE_COLOR"] = "0"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["MLX_SHOW_PROGRESS"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    sys.stderr = open(os.devnull, "w")


def send_response(response: dict) -> None:
    print(json.dumps(response), flush=True)


def format_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
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
    setup_isolation()

    model = None
    tokenizer = None
    stream_generate = None
    stop_requested = False
    is_vlm = False
    vlm_generate = None
    vlm_processor = None

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
            is_vision = cmd.get("is_vision", False)

            try:
                if is_vision:
                    try:
                        from mlx_vlm import generate as vlm_gen
                        from mlx_vlm import load as vlm_load

                        model, vlm_processor = vlm_load(model_path)
                        vlm_generate = vlm_gen
                        is_vlm = True
                        tokenizer = None
                        stream_generate = None
                        send_response({"type": "loaded", "success": True, "is_vlm": True})
                    except ImportError:
                        send_response(
                            {
                                "type": "loaded",
                                "success": False,
                                "error": "mlx-vlm not installed",
                            }
                        )
                        continue
                else:
                    from mlx_lm import load
                    from mlx_lm import stream_generate as sg

                    stream_generate = sg
                    is_vlm = False
                    vlm_generate = None
                    vlm_processor = None
                    model, tokenizer = load(model_path)
                    send_response({"type": "loaded", "success": True, "is_vlm": False})
            except ImportError as e:
                send_response({"type": "loaded", "success": False, "error": str(e)})
            except Exception as e:
                send_response({"type": "loaded", "success": False, "error": str(e)})

        elif cmd_type == "generate":
            if model is None:
                send_response({"type": "error", "message": "Model not loaded"})
                continue

            messages = cmd.get("messages", [])
            max_tokens = cmd.get("max_tokens", 512)
            temperature = cmd.get("temperature", 0.7)
            images = cmd.get("images", [])
            stop_requested = False

            try:
                if is_vlm and vlm_generate is not None:
                    from PIL import Image

                    prompt = ""
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            prompt = msg.get("content", "")
                            break
                    loaded_images = []
                    for img_path in images:
                        try:
                            loaded_images.append(Image.open(img_path))
                        except Exception as e:
                            send_response({"type": "error", "message": f"Image load failed: {e}"})
                    if loaded_images:
                        output = vlm_generate(
                            model,
                            vlm_processor,
                            loaded_images[0],
                            prompt,
                            max_tokens=max_tokens,
                            temp=temperature,
                        )
                    else:
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
                    if tokenizer is None or stream_generate is None:
                        send_response({"type": "error", "message": "Tokenizer not loaded"})
                        continue
                    prompt = format_messages(tokenizer, messages)
                    gen_kwargs: dict[str, Any] = {"prompt": prompt, "max_tokens": max_tokens}
                    try:
                        from mlx_lm.sample_utils import make_sampler

                        gen_kwargs["sampler"] = make_sampler(temperature)
                    except ImportError:
                        gen_kwargs["temp"] = temperature

                    for response in stream_generate(model, tokenizer, **gen_kwargs):
                        if stop_requested:
                            break
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
            if model is not None:
                try:
                    import mlx.core as mx

                    del model
                    if tokenizer is not None:
                        del tokenizer
                    if vlm_processor is not None:
                        del vlm_processor
                    gc.collect()
                    if hasattr(mx.metal, "clear_cache"):
                        mx.metal.clear_cache()
                        mx.eval(mx.zeros(1))
                        mx.metal.clear_cache()
                except Exception:
                    pass
            break

        else:
            send_response({"type": "error", "message": f"Unknown command: {cmd_type}"})


if __name__ == "__main__":
    main()
