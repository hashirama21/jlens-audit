#!/usr/bin/env bash
set -e
MODELS=${MODELS:-/workspace/models}
LENSES=${LENSES:-/workspace/lenses}
mkdir -p "$MODELS" "$LENSES"
huggingface-cli download Qwen/Qwen3.6-27B --local-dir "$MODELS/qwen3.6-27b"
# NOTE: one --include with several patterns. Repeating the flag (--include A --include B) lets the
# last occurrence override the first (argparse nargs="*") -> only README.md would be downloaded.
huggingface-cli download camilablank/workspace-lenses --include "qwen3.6-27b/*" "README.md" --local-dir "$LENSES"
echo "OK. Lire $LENSES/README.md AVANT de toucher src/lens.py"
