# Troubleshooting Guide

Common issues and their solutions when using r3LAY.

## Model Loading Issues

### "Out of memory" or OOM Errors

**Symptoms**: Model fails to load, process killed, Metal/CUDA errors

**Solutions**:

1. **Use smaller models**: 7B models at 4-bit quantization fit in ~8GB
   - Recommended: Qwen2.5-7B-Instruct-4bit (text)
   - Recommended: Qwen2-VL-7B-4bit (vision)

2. **Close memory-intensive applications**

3. **Check memory pressure**:
   - macOS: Activity Monitor → Memory tab
   - Linux: `free -h` or `htop`
   - NVIDIA: `nvidia-smi`

4. **For Apple Silicon**: The MLX backend uses unified memory. If you see Metal errors, reduce model size.

5. **For NVIDIA**: Ensure VRAM is sufficient:
   ```bash
   nvidia-smi
   ```

### "Model not found" Error

**Solutions**:

1. **Verify model path exists**:
   ```bash
   ls ~/.cache/huggingface/hub/
   ```

2. **Check Ollama models**:
   ```bash
   ollama list
   ```

3. **Re-download model**:
   ```bash
   ollama pull llama3.1:7b
   ```

### Model Loading Hangs

**Solutions**:

1. **Check disk I/O**: Large models take time to load from disk
2. **Verify sufficient RAM**: Model must fit in memory during load
3. **Kill and restart**: Press Ctrl+C, restart r3LAY

## Network Issues

### SearXNG Unavailable

**Symptoms**: Web search fails, `/research` shows "web search unavailable"

**Solutions**:

1. **Check if SearXNG is running**:
   ```bash
   curl http://localhost:8080/healthz
   ```

2. **Start SearXNG**:
   ```bash
   docker compose --profile search up -d searxng
   ```

3. **r3LAY continues without web search**: Uses local index only for research

4. **Check SearXNG logs**:
   ```bash
   docker compose logs searxng
   ```

### Ollama Connection Refused

**Symptoms**: "Cannot connect to Ollama" error

**Solutions**:

1. **Start Ollama**:
   ```bash
   ollama serve
   ```

2. **Verify Ollama is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

3. **Check port binding**: Ensure nothing else uses port 11434:
   ```bash
   lsof -i :11434
   ```

4. **Docker users**: Verify `host.docker.internal` resolves (see Docker docs)

### Connection Timeouts

**Solutions**:

1. **Increase timeout**: Large models may need more time
2. **Check network connectivity**
3. **Verify server isn't overloaded**

## File and Data Issues

### Corrupted Index

**Symptoms**: Search returns errors, index panel shows warnings

**Solutions**:

1. **Delete and rebuild index**:
   ```bash
   rm -rf /path/to/project/.r3lay/index/
   ```
   Then reindex using `/reindex` command or Ctrl+R

2. **Check disk space**: Index operations need free space

### Session Won't Load

**Symptoms**: Error loading previous session

**Solutions**:

1. **Check JSON validity**:
   ```bash
   python -m json.tool /project/.r3lay/sessions/<session-id>.json
   ```

2. **Delete corrupt session** and start fresh

3. **Check file permissions**

### Axioms Not Saving

**Solutions**:

1. **Check write permissions** on project directory
2. **Verify disk space**
3. **Check axioms directory exists**:
   ```bash
   ls -la /project/axioms/
   ```

## UI Issues

### Terminal Rendering Broken

**Symptoms**: Garbled text, missing borders, wrong colors

**Solutions**:

1. **Ensure terminal supports 256 colors**:
   ```bash
   echo $TERM
   # Should be xterm-256color or similar
   ```

2. **Set TERM variable**:
   ```bash
   export TERM=xterm-256color
   python -m r3lay.app /project
   ```

3. **Try different terminal emulator**:
   - Recommended: iTerm2 (macOS), Alacritty, Kitty
   - Works: Terminal.app, GNOME Terminal, Windows Terminal
   - Problematic: Some older terminals

4. **Check terminal size**: Minimum recommended 80x24

### Keybindings Not Working

**Solutions**:

1. **Check for conflicting bindings**:
   - tmux/screen may capture some keys
   - Terminal emulator may override keys

2. **Ensure input focus**: Click in the input area or use Tab

3. **Common conflicts**:
   - Ctrl+Q: May be captured by terminal
   - Ctrl+R: tmux uses this for history search

### Slow UI Updates

**Solutions**:

1. **Reduce terminal size**: Larger terminals require more rendering
2. **Check CPU usage**: Heavy model inference can slow UI
3. **Use faster model**: Smaller models respond faster

## Research and Axioms

### Research Taking Too Long

**Solutions**:

1. **Cancel with Escape**: Press Escape to stop current research
2. **Use more specific queries**: Broad queries generate many cycles
3. **Check convergence**: Research stops when axiom generation decreases

### Contradictions Not Resolving

**Solutions**:

1. **Manual review**: Some contradictions need human judgment
2. **Use `/axioms --disputed`** to see disputed axioms
3. **Resolve manually** with `/dispute` command

### Axioms Not Appearing in Chat

**Solutions**:

1. **Validate axioms first**: Only validated axioms appear in context
2. **Check axiom count**: Large axiom sets may be truncated
3. **Verify axiom categories**: Some categories may be filtered

## Docker-Specific Issues

See [Docker Deployment Guide](docker.md) for Docker-specific troubleshooting.

## Getting Help

If you're still stuck:

1. **Check logs**: Enable debug logging with `--debug` flag
2. **Report issues**: https://github.com/dlorp/r3LAY/issues
3. **Include**: OS version, Python version, model being used, full error message
