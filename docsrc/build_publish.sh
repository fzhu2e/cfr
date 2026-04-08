#!/usr/bin/env bash
# Build both documentation backends and assemble the output in docs/
# Run from the repository root: bash docsrc/build_publish.sh

set -e

# Clean previous builds
rm -rf docs/v2024 docs/v2026

# Build v2024 docs
sphinx-build docsrc/v2024 docs/v2024

# Build v2026 docs
sphinx-build docsrc/v2026 docs/v2026

# Copy landing page
cp docsrc/landing/index.html docs/index.html

# Ensure GitHub Pages compatibility
touch docs/.nojekyll

# Stage for commit
git add docs/
