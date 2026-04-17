#!/usr/bin/env bash
# Build both documentation backends and assemble the output in _site/
# Run from the repository root: bash docsrc/build.sh
#
# Usage:
#   bash docsrc/build.sh              # build all versions
#   bash docsrc/build.sh v2024        # rebuild v2024 only
#   bash docsrc/build.sh v2026        # rebuild v2026 only
#   bash docsrc/build.sh --deploy     # build all and deploy to gh-pages
#   bash docsrc/build.sh --push       # deploy only (skip build)

set -e

ACTION="build"
VERSION="all"

for arg in "$@"; do
    case "$arg" in
        --deploy) ACTION="build-and-deploy" ;;
        --push)   ACTION="push" ;;
        v2024|v2026) VERSION="$arg" ;;
    esac
done

if [ "$ACTION" != "push" ]; then
    if [ "$VERSION" = "all" ]; then
        rm -rf _site
    fi

    if [ "$VERSION" = "all" ] || [ "$VERSION" = "v2024" ]; then
        rm -rf _site/v2024
        sphinx-build docsrc/v2024 _site/v2024
    fi

    if [ "$VERSION" = "all" ] || [ "$VERSION" = "v2026" ]; then
        rm -rf _site/v2026
        sphinx-build docsrc/v2026 _site/v2026
    fi

    # Copy landing page
    cp docsrc/landing/index.html _site/index.html

    # Ensure GitHub Pages compatibility
    touch _site/.nojekyll

    echo ""
    echo "Site built in _site/."
fi

if [ "$ACTION" = "build-and-deploy" ] || [ "$ACTION" = "push" ]; then
    if [ ! -d _site ]; then
        echo "Error: _site/ does not exist. Run without --push first to build."
        exit 1
    fi
    echo "Deploying to gh-pages..."
    ghp-import -n -p -f _site
    echo "Deployed to gh-pages."
else
    echo "To preview locally:"
    echo "  python -m http.server -d _site 8000"
    echo ""
    echo "To deploy to gh-pages:"
    echo "  bash docsrc/build.sh --deploy"
    echo "  bash docsrc/build.sh --push   # skip build"
fi
