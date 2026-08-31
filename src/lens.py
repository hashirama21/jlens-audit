"""The three instruments under a common interface: readout(h, layer, k) -> top-k tokens.

    logit : final_norm(h) @ W_U^T
    jlens : final_norm(M_L @ h) @ W_U^T   with M_L the J-lens matrix of layer L
    rlens : same with the R-lens matrix

# ADAPTER (mandatory, read lenses/README.md of the camilablank/workspace-lenses repo):
  - file name and format per layer (safetensors? .pt? one file per layer or a single file?)
  - is the matrix (d, d) applied to h, or already (vocab, d) (in which case W_U is included)?
  - is there a bias? is the final norm already folded into the matrix?
  - does lens layer L correspond to hidden_states[L+1] (output of block L)?
The conformity test (src/validate.py --smoke) must pass before going any further.
"""
import torch
from safetensors.torch import load_file

from .load_model import load, unembed_parts, layers
from .config import LENS_DIR


def _load_layer_map(kind: str, L: int) -> torch.Tensor:
    # ADAPTER: path and key per README. Plausible candidates below, to be corrected.
    cands = [LENS_DIR / f"{kind}_layer{L}.safetensors", LENS_DIR / kind / f"layer_{L}.safetensors"]
    for c in cands:
        if c.exists():
            sd = load_file(str(c))
            key = "weight" if "weight" in sd else next(iter(sd))
            return sd[key]
    raise FileNotFoundError(f"Lens {kind} layer {L} not found — read lenses/README.md (candidates: {cands})")


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
            z = H_L @ self.maps[L].T   # ADAPTER: confirm matrix orientation and whether norm is folded in
            z = self.final_norm(z)     # ADAPTER: drop this line if the lens already includes the final norm
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