#!/usr/bin/env bash
# Build both documentation backends and assemble the output in _site/
# Run from the repository root: bash docsrc/build_publish.sh
#
# The built site is placed in _site/ for local preview.
# Deployment to gh-pages is handled by GitHub Actions (.github/workflows/docs.yml).

set -e

# Clean previous builds
rm -rf _site

# Build v2024 docs
sphinx-build docsrc/v2024 _site/v2024

# Build v2026 docs
sphinx-build docsrc/v2026 _site/v2026

# Copy landing page and version switcher config
cp docsrc/landing/index.html _site/index.html
cp docsrc/versions.json _site/versions.json

# Ensure GitHub Pages compatibility
touch _site/.nojekyll

echo ""
echo "Site built in _site/. To preview locally:"
echo "  python -m http.server -d _site 8000"
