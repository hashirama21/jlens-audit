"""Central configuration. Every design choice lives here, commented."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("MODELS", "/workspace/models")) / "qwen3.6-27b"
LENS_DIR = Path(os.environ.get("LENSES", "/workspace/lenses")) / "qwen3.6-27b"

DATA = ROOT / "data"
SCANS = ROOT / "scans"
JUDGE_PROMPTS = ROOT / "judge" / "prompts"
JUDGE_OUT = ROOT / "judge" / "outputs"
RESULTS = ROOT / "results"
FIGS = ROOT / "figs"
for p in (DATA, SCANS, JUDGE_OUT, RESULTS, FIGS):
    p.mkdir(parents=True, exist_ok=True)

# --- Instruments ---
INSTRUMENTS = ["jlens", "rlens", "logit"]

# --- Model architecture (Qwen3.6-27B; confirmed against lenses/README.md) ---
N_LAYERS = 64             # transformer blocks (hidden_states has N_LAYERS + 1 entries).
D_MODEL = 5120            # residual width; matches the lens J stack (d_model in lens.pt).
TARGET_LAYER = 62         # lens anchor row = identity here (n_layers - 2); J_62 == I -> degenerates to logit lens.
SKIP_FIRST = 0            # J covers layers 0..62; skip_first=4 in provenance is a fitting
                          # parameter, NOT a row exclusion. Verified in lens.pt (63 keys, 0..62).

USE_CHAT_TEMPLATE = True      # feed texts through the chat template (deployment framing).
ADD_GENERATION_PROMPT = True  # deployment framing (model about to answer); content_span still
                              # excludes the assistant prompt, and causal attention leaves the
                              # content-position residuals unchanged either way.
ENABLE_THINKING = False       # Qwen3 template: no <think> block (would break capability YES/NO parsing).

# --- Scan ---
LAYER_STRIDE = 8          # scan 1 layer out of LAYER_STRIDE over the 64 blocks (J keys gate the grid).
TOP_K = 3                 # top-k tokens per (position, layer); the smoke test shows the concept in
                          # the top-3, so this cuts ~40% of judge context at little cost (note in limits).

# --- Corpus ---
FAMILIES = ["injection", "bug", "false_premise", "conflict"]
PAIRS_PER_FAMILY = 6
MAX_DIFF_FRAC = 0.10      # the clean twin differs from the anomalous version by < 10% of tokens.

# --- Judges ---
# OpenRouter model ids, verified against openrouter.ai/api/v1/models (2026-08). Env-overridable.
JUDGES = {
    "judgeA": os.environ.get("JUDGE_A", "anthropic/claude-sonnet-4.6"),  # was 4.5 (superseded)
    "judgeB": os.environ.get("JUDGE_B", "openai/gpt-5-mini"),            # still current
}
JUDGE_TEMPERATURE = 0.0
PROMPT_VERSIONS = ["v1", "v2"]

# --- Corpus generator ---
# A THIRD family, distinct from both judges (Anthropic judgeA, OpenAI judgeB), to avoid a
# self-grading confound. Deliberately not a Qwen: that is the family under audit (Qwen3.6-27B).
# Flash, not Pro: gemini-2.5-pro is a reasoning model whose thinking tokens consumed the output
# budget and truncated the JSON (finish_reason=length, ~few hundred chars even at max_tokens=3000).
# Generating clean code + a one-line mutation needs a fast structured generator, not deep reasoning.
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "google/gemini-2.5-flash")
GENERATOR_TEMPERATURE = 0.9   # variety at generation time; the judge stays at temperature 0.

# --- Metrics ---
N_BOOT = 1000
FPR_TARGET = 0.05
SEED = 0