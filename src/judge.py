"""LLM call harness: OpenRouter, forced JSON, retry, disk cache.

`complete(model, ...)` is the single entry point (also used by the corpus generator).
`call(judge, ...)` is the thin judge-facing wrapper at temperature 0.

Cache key includes the model id AND temperature, so changing JUDGE_A mid-project can never
silently serve an old model's answers. Errors are never cached, so a re-run retries them."""
import json
import hashlib
import os
import time
import re

from .config import JUDGES, JUDGE_TEMPERATURE, JUDGE_PROMPTS, JUDGE_OUT

_client = None


def client():
    global _client
    if _client is None:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set — `export OPENROUTER_API_KEY=...` "
                               "before running generation or judge calls.")
        from openai import OpenAI
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=90)
    return _client


CACHE = JUDGE_OUT / "_cache"
CACHE.mkdir(exist_ok=True)


def prompt(name: str, content: str) -> str:
    return (JUDGE_PROMPTS / f"{name}.txt").read_text().replace("{CONTENT}", content)


def _key(*parts) -> str:
    return hashlib.sha1("||".join(map(str, parts)).encode()).hexdigest()


def _parse_json(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def complete(model: str, text: str, *, temperature: float, want_json: bool = True,
             max_tokens: int = 600, retries: int = 3) -> dict | str:
    """One completion. Successful results are cached; errors are not (so re-runs retry).
    Network/API failures are retried with backoff; a JSON parse failure at temperature 0 is
    deterministic, so it is NOT retried (that would just burn identical calls)."""
    cache_file = CACHE / (_key(model, temperature, want_json, max_tokens, text) + ".json")
    if cache_file.exists():
        return json.load(open(cache_file))

    cli = client()   # fail fast on a missing key (config error) rather than retry it 3x
    last = None
    for attempt in range(retries):
        try:
            r = cli.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "user", "content": text}])
            raw = r.choices[0].message.content or ""
            print(f"[completion] model={model} finish_reason={r.choices[0].finish_reason} "
                  f"chars={len(raw)} max_tokens={max_tokens}")
        except Exception as e:  # network / API error -> retry with backoff
            last = e
            print(f"[api] {model} attempt {attempt + 1}/{retries}: {type(e).__name__}: {e}")
            time.sleep(2 ** attempt)
            continue

        if not want_json:
            if raw.strip():                       # never cache an empty response as a success
                json.dump(raw, open(cache_file, "w"))
            return raw
        parsed = _parse_json(raw)
        if parsed is None:  # deterministic at temp 0 -> do not retry, surface as error (uncached)
            return {"anomaly": False, "confidence": 0.0,  # full raw so raw_fail_*.txt shows the cut-off
                    "description": f"JSON parse failed (finish_reason={r.choices[0].finish_reason})",
                    "evidence": [], "_raw": raw, "_error": True}
        parsed.setdefault("anomaly", False)
        parsed.setdefault("confidence", 0.0)
        parsed["_raw"] = raw
        json.dump(parsed, open(cache_file, "w"), ensure_ascii=False)
        return parsed

    return {"anomaly": False, "confidence": 0.0, "description": f"API error: {last}",
            "evidence": [], "_error": True}


def call(judge: str, text: str, want_json: bool = True, max_tokens: int = 1200) -> dict | str:
    """Judge-facing wrapper: resolves the logical judge name and pins temperature to 0."""
    return complete(JUDGES[judge], text, temperature=JUDGE_TEMPERATURE,
                    want_json=want_json, max_tokens=max_tokens)