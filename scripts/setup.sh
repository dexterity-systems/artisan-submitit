#!/usr/bin/env bash
# One-time, idempotent contributor setup after installing the Pixi environment.

set -euo pipefail

if command -v pre-commit >/dev/null 2>&1; then
    echo "setup: installing pre-commit hooks..."
    pre-commit install >/dev/null
fi

echo "setup: done."
