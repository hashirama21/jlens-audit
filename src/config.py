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

# --- Scan ---
LAYER_STRIDE = 4          # scan 1 layer out of LAYER_STRIDE; on a ~48-layer 27B this is ~12 layers.
TOP_K = 10                # top-k tokens per (position, layer); drop to 5 if the judge context saturates.

# --- Corpus ---
FAMILIES = ["injection", "bug", "false_premise", "conflict"]
PAIRS_PER_FAMILY = 10
MAX_DIFF_FRAC = 0.10      # the clean twin differs from the anomalous version by < 10% of tokens.

# --- Judges ---
JUDGES = {
    # logical name -> OpenRouter model id. Confirm at run time (pricing, availability).
    "judgeA": os.environ.get("JUDGE_A", "anthropic/claude-sonnet-4.5"),
    "judgeB": os.environ.get("JUDGE_B", "openai/gpt-5-mini"),
}
JUDGE_TEMPERATURE = 0.0
PROMPT_VERSIONS = ["v1", "v2"]

# --- Corpus generator (kept DISTINCT from the judges to avoid a self-grading confound) ---
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "anthropic/claude-opus-4.1")
GENERATOR_TEMPERATURE = 0.9   # variety at generation time; the judge stays at temperature 0.

# --- Metrics ---
N_BOOT = 1000
FPR_TARGET = 0.05
SEED = 0