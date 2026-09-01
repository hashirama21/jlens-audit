"""Single model load. Run inside the SETUP cell of the persistent kernel, or imported by
the scripts (which reload — acceptable for background runs).

`load_tok()` returns the tokenizer alone (no GPU, no 27B weights) so corpus tooling and
tests can run on a laptop."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import MODEL_DIR, LAYER_STRIDE, N_LAYERS, TARGET_LAYER, SKIP_FIRST, USE_CHAT_TEMPLATE

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
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), dtype=dtype, device_map=device_map)
    model.eval()
    _state["model"] = model
    return tok, model


def layers():
    """Scan grid, gated by the lens source_layers (skip_first=4 removes layers 0..3),
    with TARGET_LAYER always included (identity anchor, the key test). Falls back to the
    naive stride grid when the lens files are not downloaded yet, so the SETUP cell can
    report a grid before the lenses are loaded."""
    try:
        from .lens import _stack
        src = set(_stack("jlens")[1])
    except FileNotFoundError:
        grid = list(range(SKIP_FIRST, N_LAYERS, LAYER_STRIDE))
        return sorted(set(grid + [TARGET_LAYER]))
    grid = [L for L in range(0, N_LAYERS, LAYER_STRIDE) if L in src]
    if TARGET_LAYER not in grid and TARGET_LAYER in src:
        grid.append(TARGET_LAYER)
    return sorted(grid)


def unembed_parts():
    """W_U (lm_head) and the final norm (model.model.norm)."""
    _, model = load()
    return model.lm_head.weight, model.model.norm


def render_input(tok, content, *, add_generation_prompt=False):
    """The exact string fed to the model: chat-templated (special tokens included) or raw.
    Single source of truth for the input framing (USE_CHAT_TEMPLATE)."""
    if USE_CHAT_TEMPLATE:
        return tok.apply_chat_template([{"role": "user", "content": content}],
                                       add_generation_prompt=add_generation_prompt, tokenize=False)
    return content


def _add_special():
    # Under the template the rendered string already carries the special tokens, so we must
    # not add them again (matches apply_chat_template(tokenize=True)); raw mode keeps the default.
    return not USE_CHAT_TEMPLATE


def to_input_ids(tok, content, *, add_generation_prompt=False):
    """Input ids (1, seq) for `content`, so scan, capability probe and span share one framing."""
    rendered = render_input(tok, content, add_generation_prompt=add_generation_prompt)
    return tok(rendered, return_tensors="pt", add_special_tokens=_add_special())["input_ids"]


def content_span(tok, content):
    """(lo, hi) token indices covering `content` inside the framed sequence, so the scan can
    skip the template scaffolding while keeping absolute positions (span/evidence stay aligned).
    Full range in raw mode, or when the content cannot be located verbatim."""
    rendered = render_input(tok, content)
    seq = len(tok(rendered, add_special_tokens=_add_special())["input_ids"])
    if not USE_CHAT_TEMPLATE:
        return 0, seq
    a = rendered.find(content)
    if a < 0:
        return 0, seq
    b = a + len(content)
    off = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    lo = next((i for i, (s, e) in enumerate(off) if e > a), 0)
    hi = next((i + 1 for i in range(len(off) - 1, -1, -1) if off[i][0] < b), seq)
    return lo, hi


@torch.no_grad()
def get_resid(text, layer_list=None):
    """dict layer -> (seq, d) residual stream (bf16, on GPU), plus the input token ids.
    Convention: hidden_states[0] = embeddings, hidden_states[L+1] = output of block L."""
    tok, model = load()
    layer_list = layer_list or layers()
    ids = to_input_ids(tok, text).to(model.device)
    out = model(input_ids=ids, output_hidden_states=True)
    return {L: out.hidden_states[L + 1][0] for L in layer_list}, ids[0]
