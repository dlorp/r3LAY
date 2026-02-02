# Contributing to r3LAY

Thanks for your interest in contributing! 🎉

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/r3LAY.git`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Create a branch: `git checkout -b feature/your-feature`

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/
```

## Pull Requests

- Keep PRs focused — one feature or fix per PR
- Add tests for new functionality
- Update docs if needed
- Follow existing code style

## Code Style

- Python 3.11+ type hints
- Use `ruff` for linting and formatting
- Docstrings for public functions

## Reporting Issues

- Check existing issues first
- Include steps to reproduce
- Include Python version and OS

## Questions?

Open a discussion or issue — we're happy to help!
