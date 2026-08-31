"""The three instruments under a common interface: readout(h, layer, k) -> top-k tokens.

    logit : final_norm(h) @ W_U^T
    jlens : final_norm(J_L @ h) @ W_U^T   with J_L the J-lens Jacobian of layer L
    rlens : same with the R-lens matrix

Format resolved against lenses/README.md (camilablank/workspace-lenses):
  - ONE torch file per lens: <LENS_DIR>/{j-lens,r-lens}/lens.pt, a dict with keys
    ['J', 'n_prompts', 'source_layers', 'd_model', 'provenance']. J is the stack of
    per-layer Jacobians; row i corresponds to source_layers[i].
  - readout is softmax(W_U . norm(J_L . h)): J_L is (d, d) applied to h (NOT (vocab, d)),
    there is NO bias, and the final norm is NOT folded in (we apply self.final_norm here).
  - the anchor row at source_layers == TARGET_LAYER is exactly I, so at L=TARGET_LAYER
    both lenses degenerate to the logit lens (see src/validate.py identity_check).
The go/no-go test (src/validate.py identity_check) must pass before going any further.
"""
import torch

from .load_model import load, unembed_parts, layers
from .config import LENS_DIR

# One lens.pt per kind, loaded once: (J stack, {source_layer -> row index}, full dict).
_STACKS: dict = {}
_SUBDIR = {"jlens": "j-lens", "rlens": "r-lens"}


def _stack(kind: str):
    if kind not in _STACKS:
        d = torch.load(LENS_DIR / _SUBDIR[kind] / "lens.pt", map_location="cpu", weights_only=False)
        idx = {int(sl): i for i, sl in enumerate(d["source_layers"])}
        _STACKS[kind] = (d["J"], idx, d)
    return _STACKS[kind]


def _load_layer_map(kind: str, L: int) -> torch.Tensor:
    J, idx, _ = _stack(kind)
    if L not in idx:
        raise KeyError(f"layer {L} absent from {kind} source_layers (skip_first=4); available: {sorted(idx)}")
    return J[idx[L]]


class Lens:
    def __init__(self, kind: str):
        assert kind in ("logit", "jlens", "rlens")
        self.kind = kind
        self.tok, self.model = load()
        self.W_U, self.final_norm = unembed_parts()
        self.maps = {}
        if kind != "logit":
            for L in layers():
                self.maps[L] = _load_layer_map(kind, L).to(self.model.device, torch.bfloat16)

    @torch.no_grad()
    def logits(self, H_L: torch.Tensor, L: int) -> torch.Tensor:
        """H_L : (seq, d) -> (seq, vocab). Batched per layer for speed."""
        if self.kind == "logit":
            z = self.final_norm(H_L)
        else:
            z = H_L @ self.maps[L].T   # J_L . h per row (norm not folded into J)
            z = self.final_norm(z)     # explicit norm() from the readout formula
        return z @ self.W_U.T

    @torch.no_grad()
    def readout_all(self, H_L: torch.Tensor, L: int, k: int = 10):
        """(seq, d) -> list (seq) of lists of k decoded tokens."""
        top = torch.topk(self.logits(H_L, L), k, dim=-1).indices.cpu()
        return [[self.tok.decode(int(t)) for t in row] for row in top]

    @torch.no_grad()
    def readout(self, h: torch.Tensor, L: int, k: int = 10):
        return self.readout_all(h[None], L, k)[0]


def load_all():
    return {kind: Lens(kind) for kind in ("jlens", "rlens", "logit")}