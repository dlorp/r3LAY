# Contributing to r3LAY

Thank you for your interest in contributing to r3LAY! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** and clone it locally
2. **Set up the development environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
   pip install -e ".[dev]"
   ```
3. **Run tests** to ensure everything works:
   ```bash
   pytest tests/
   ```

## Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the code style guidelines

3. Write or update tests as needed

4. Run the linter and formatter:
   ```bash
   ruff check --fix .
   ruff format .
   ```

5. Run tests:
   ```bash
   pytest tests/
   ```

6. Commit your changes with a clear message:
   ```bash
   git commit -m "feat: add your feature description"
   ```

7. Push and create a pull request

## Code Style

- We use **ruff** for linting and formatting
- Follow PEP 8 conventions
- Use type hints where practical
- Write docstrings for public functions and classes

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

## Pull Requests

- Keep PRs focused on a single change
- Include tests for new functionality
- Update documentation as needed
- Ensure CI passes before requesting review

## Reporting Issues

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version)

## Questions?

Feel free to open an issue for questions or discussions about the project.
