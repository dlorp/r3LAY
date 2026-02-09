#!/bin/bash
# Version consistency check - ensures all version references match pyproject.toml
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Extract version from pyproject.toml (source of truth)
PYPROJECT_VERSION=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)

if [ -z "$PYPROJECT_VERSION" ]; then
    echo "❌ Error: Could not extract version from pyproject.toml"
    exit 1
fi

echo "📦 Checking version consistency..."
echo "   Source of truth (pyproject.toml): $PYPROJECT_VERSION"
echo ""

ERRORS=0

# Check README.md badge
README_VERSION=$(grep -o 'version-[0-9]\+\.[0-9]\+\.[0-9]\+' README.md | head -1 | cut -d'-' -f2)
if [ "$README_VERSION" != "$PYPROJECT_VERSION" ]; then
    echo "❌ README.md badge mismatch: $README_VERSION (expected $PYPROJECT_VERSION)"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ README.md badge: $README_VERSION"
fi

# Check r3lay/__init__.py
INIT_VERSION=$(grep '__version__ = ' r3lay/__init__.py | cut -d'"' -f2)
if [ "$INIT_VERSION" != "$PYPROJECT_VERSION" ]; then
    echo "❌ r3lay/__init__.py mismatch: $INIT_VERSION (expected $PYPROJECT_VERSION)"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ r3lay/__init__.py: $INIT_VERSION"
fi

# Check starting_docs/__init__.py
STARTING_VERSION=$(grep '__version__ = ' starting_docs/__init__.py | cut -d'"' -f2)
if [ "$STARTING_VERSION" != "$PYPROJECT_VERSION" ]; then
    echo "❌ starting_docs/__init__.py mismatch: $STARTING_VERSION (expected $PYPROJECT_VERSION)"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ starting_docs/__init__.py: $STARTING_VERSION"
fi

echo ""
if [ $ERRORS -gt 0 ]; then
    echo "❌ Version consistency check FAILED: $ERRORS mismatch(es) found"
    echo ""
    echo "To fix, update all versions to match pyproject.toml:"
    echo "  - README.md badge"
    echo "  - r3lay/__init__.py"
    echo "  - starting_docs/__init__.py"
    exit 1
else
    echo "✅ All version references are consistent: $PYPROJECT_VERSION"
    exit 0
fi
