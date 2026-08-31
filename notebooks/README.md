# notebooks/

`main.ipynb` est le kernel persistant partagé avec l'agent (via jupyter-mcp-server, voir env/jupyter.sh).

Cellule 1 — SETUP (à exécuter une fois, ne jamais réexécuter sans demander) :
```python
import sys; sys.path.insert(0, "..")
from src.load_model import load, layers, get_resid
from src.lens import load_all
tok, model = load()
lenses = load_all()          # après avoir résolu les # ADAPTER de src/lens.py
print(len(layers()), "couches scannées :", layers())
```
Cellule 2 — smoke test :
```python
from src.validate import smoke; smoke(lenses)
```
Ensuite : chaque expérience dans sa cellule, résultats -> results/, figures -> figs/. Rien d'important ne vit seulement dans le notebook.
