# notebooks/

`main.ipynb` is the persistent kernel shared with the agent (via jupyter-mcp-server, see env/jupyter.sh).

Cell 1 — SETUP (run once, never re-run without asking):
```python
import sys; sys.path.insert(0, "..")
from src.load_model import load, layers, get_resid
from src.lens import load_all
tok, model = load()
lenses = load_all()          # after resolving the # ADAPTER points in src/lens.py
print(len(layers()), "layers scanned:", layers())
```
Cell 2 — smoke test:
```python
from src.validate import smoke; smoke(lenses)
```
Then: each experiment in its own cell, results -> results/, figures -> figs/. Nothing important lives only in the notebook.
