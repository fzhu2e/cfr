#!/usr/bin/env bash
# Build both documentation backends and assemble the output in _site/
# Run from the repository root: bash docsrc/build_publish.sh
#
# Usage:
#   bash docsrc/build_publish.sh          # build only
#   bash docsrc/build_publish.sh --deploy  # build and deploy to gh-pages

set -e

DEPLOY=false
if [ "$1" = "--deploy" ]; then
    DEPLOY=true
fi

# Clean previous builds
rm -rf _site

# Build v2024 docs
sphinx-build docsrc/v2024 _site/v2024

# Build v2026 docs
sphinx-build docsrc/v2026 _site/v2026

# Copy landing page
cp docsrc/landing/index.html _site/index.html

# Ensure GitHub Pages compatibility
touch _site/.nojekyll

echo ""
echo "Site built in _site/."

if [ "$DEPLOY" = true ]; then
    echo "Deploying to gh-pages..."
    ghp-import -n -p -f _site
    echo "Deployed to gh-pages."
else
    echo "To preview locally:"
    echo "  python -m http.server -d _site 8000"
    echo ""
    echo "To deploy to gh-pages:"
    echo "  bash docsrc/build_publish.sh --deploy"
fi
