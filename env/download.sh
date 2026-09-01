#!/usr/bin/env bash
set -e
MODELS=${MODELS:-/workspace/models}
LENSES=${LENSES:-/workspace/lenses}
mkdir -p "$MODELS" "$LENSES"
hf download Qwen/Qwen3.6-27B --local-dir "$MODELS/qwen3.6-27b"
# NOTE: one --include with several patterns. Repeating the flag (--include A --include B) lets the
# last occurrence override the first (argparse nargs="*") -> only README.md would be downloaded.
# Restrict to j-lens/r-lens (+README): pulling "qwen3.6-27b/*" would also drag the ~18 GB
# template-lens/ we never use (~7 GB instead of ~25 GB).
hf download camilablank/workspace-lenses \
  --include "qwen3.6-27b/j-lens/*" "qwen3.6-27b/r-lens/*" "README.md" \
  --local-dir "$LENSES"
echo "OK. Read $LENSES/README.md BEFORE touching src/lens.py"
