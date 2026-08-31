#!/usr/bin/env bash
# Persistent kernel for the agent (recipe: Neel's application doc, "persistent Jupyter Kernel" section)
set -euo pipefail
TOKEN="${JUPYTER_TOKEN:?set JUPYTER_TOKEN to a non-empty secret before launching (do not expose 0.0.0.0 without one)}"
cd "$(dirname "$0")/.."
jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --NotebookApp.token="$TOKEN" --notebook-dir=notebooks
# Then in the Claude Code config: jupyter-mcp-server with URL http://<pod>:8888 and this token.