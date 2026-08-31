"""Single model load. Run inside the SETUP cell of the persistent kernel, or imported by
the scripts (which reload — acceptable for background runs).

`load_tok()` returns the tokenizer alone (no GPU, no 27B weights) so corpus tooling and
tests can run on a laptop."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import MODEL_DIR, LAYER_STRIDE

_state = {}


def load_tok():
    """Tokenizer only — cheap, no GPU. Use this anywhere the full model is not needed."""
    if "tok" not in _state:
        _state["tok"] = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    return _state["tok"]


def load(dtype=torch.bfloat16, device_map="cuda"):
    if "model" in _state:
        return _state["tok"], _state["model"]
    tok = load_tok()
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=dtype, device_map=device_map)
    model.eval()
    _state["model"] = model
    return tok, model


def layers():
    _, model = load()
    n = model.config.num_hidden_layers
    return list(range(0, n, LAYER_STRIDE))


def unembed_parts():
    """W_U and the final norm. # ADAPTER if the architecture names them differently (model.model.norm / lm_head)."""
    _, model = load()
    return model.lm_head.weight, model.model.norm


@torch.no_grad()
def get_resid(text, layer_list=None):
    """dict layer -> (seq, d) residual stream (bf16, on GPU), plus the input token ids.
    Convention: hidden_states[0] = embeddings, hidden_states[L+1] = output of block L.  # ADAPTER if needed."""
    tok, model = load()
    layer_list = layer_list or layers()
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model(**ids, output_hidden_states=True)
    return {L: out.hidden_states[L + 1][0] for L in layer_list}, ids["input_ids"][0]
