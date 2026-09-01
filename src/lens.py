"""The three instruments under a common interface: readout(h, layer, k) -> top-k tokens.

    logit : final_norm(h) @ W_U^T
    jlens : final_norm(J_L @ h) @ W_U^T   with J_L the J-lens Jacobian of layer L
    rlens : same with the R-lens matrix

Format resolved by inspecting the real lens.pt (camilablank/workspace-lenses):
  - ONE torch file per lens: <LENS_DIR>/{j-lens,r-lens}/lens.pt, a dict with keys
    ['J', 'n_prompts', 'source_layers', 'd_model', 'provenance']. J is a dict
    {layer_number: Jacobian (d, d)} in float16, keyed 0..TARGET_LAYER (63 layers) —
    NOT a stack. skip_first in provenance is a fitting parameter, not a row exclusion.
  - readout is softmax(W_U . norm(J_L . h)): J_L is (d, d) applied to h (NOT (vocab, d)),
    there is NO bias, and the final norm is NOT folded in (we apply self.final_norm here).
  - J[TARGET_LAYER] is exactly I, so at L=TARGET_LAYER both lenses degenerate to the
    logit lens (see src/validate.py identity_check).
J is fp16 on disk; we cast to bf16 to match the residual stream at compute time.
Run `python -m src.lens` to inspect the files (keys, dtype, provenance, identity anchor).
"""
import torch

from .load_model import load, unembed_parts, layers
from .config import LENS_DIR, TARGET_LAYER

# One lens.pt per kind, loaded once. Tuple: (J dict {layer: matrix}, same J so layers()
# can iterate the keys, full lens dict).
_STACKS: dict = {}
_SUBDIR = {"jlens": "j-lens", "rlens": "r-lens"}


def _stack(kind: str):
    if kind not in _STACKS:
        d = torch.load(LENS_DIR / _SUBDIR[kind] / "lens.pt", map_location="cpu", weights_only=False)
        _STACKS[kind] = (d["J"], d["J"], d)   # J is keyed directly by layer number
    return _STACKS[kind]


def _load_layer_map(kind: str, L: int) -> torch.Tensor:
    J, _, _ = _stack(kind)
    if L not in J:
        raise KeyError(f"layer {L} absent from {kind} J (keys {min(J)}..{max(J)})")
    return J[L].to(torch.bfloat16)   # fp16 on disk -> bf16 to match the residual at compute time


class Lens:
    def __init__(self, kind: str):
        assert kind in ("logit", "jlens", "rlens")
        self.kind = kind
        self.tok, self.model = load()
        self.W_U, self.final_norm = unembed_parts()
        self.maps = {}
        if kind != "logit":
            for L in layers():
                self.maps[L] = _load_layer_map(kind, L).to(self.model.device)  # already bf16

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


def inspect(kind: str = "jlens"):
    """Print the lens.pt structure and check the identity anchor. Run on the pod (needs the
    file + torch): `python -m src.lens` inspects both lenses. Closes the checkpoint/provenance
    blind spot that neither identity_check nor orientation_check cover."""
    J, _, d = _stack(kind)
    keys = sorted(J)
    print(f"{kind}: {len(J)} layers, keys {keys[:3]}..{keys[-3:]}, dtype {J[keys[0]].dtype}")
    print("provenance:", d.get("provenance"))
    A = J[TARGET_LAYER].float()
    ok = torch.allclose(A, torch.eye(A.shape[0]), atol=1e-3)
    print(f"anchor J[{TARGET_LAYER}] == I ? {ok} (diag mean {A.diagonal().mean().item():.4f})")


if __name__ == "__main__":
    import sys
    for k in (sys.argv[1:] or ["jlens", "rlens"]):
        inspect(k)
        print()