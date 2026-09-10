#!/usr/bin/env python3
"""General input-token counter across model families.

    count_tokens(text, model) -> TokenResult

Design choice (per requirements): LOCAL-ONLY, no vendor API calls, no keys.
Consequence you must keep in mind:

  * OpenAI (tiktoken) and Qwen (HuggingFace) have REAL local tokenizers -> exact.
  * Anthropic (Claude) ships NO local tokenizer for any 3+/4/5 model, and Google
    (Gemini) ships only partial local coverage. Under the local-only rule we
    therefore APPROXIMATE both with a BPE proxy (tiktoken o200k_base) times a
    tunable per-family calibration factor. These are estimates, flagged exact=False.
    If you need the true Claude/Gemini count, call their count_tokens() HTTP APIs
    instead -- that is out of scope here by design.

Unknown / future versions (e.g. "gpt-6", "gemini-3.5", "qwen-3.8", "claude-sonnet-5"):
FAMILY FALLBACK -- resolve by family to the best-available shipped tokenizer and
flag exact=False when the specific version could not be matched.

Backends are lazy-imported, so the tool runs with whatever subset is installed:
  tiktoken            -> OpenAI (exact) + Claude/Gemini proxy
  transformers        -> Qwen (exact, one-time HF tokenizer download, then cached)
If a backend lib is missing it falls back to a chars-per-token heuristic (flagged).

CLI:
  python temp/tokenlen.py -m gpt-6 "some text"
  python temp/tokenlen.py -m claude-sonnet-5 --file notes.md
  echo hi | python temp/tokenlen.py -m qwen-3.8 --stdin
  python temp/tokenlen.py --list
"""
from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass, field


# ---- family resolution -------------------------------------------------------

FAMILIES = ("openai", "anthropic", "google", "qwen")

# substrings -> family. First match wins; order matters (check specific first).
_FAMILY_HINTS = [
    ("qwen", "qwen"),
    ("claude", "anthropic"), ("sonnet", "anthropic"),
    ("opus", "anthropic"), ("haiku", "anthropic"), ("anthropic", "anthropic"),
    ("gemini", "google"), ("gemma", "google"), ("google", "google"), ("palm", "google"),
    ("gpt", "openai"), ("openai", "openai"), ("davinci", "openai"),
    ("o1", "openai"), ("o3", "openai"), ("o4", "openai"),  # reasoning-model names
]

# Best-available SHIPPED local tokenizer per family, used as the fallback when the
# exact requested version isn't recognized by the backend.
_OPENAI_FALLBACK_ENCODING = "o200k_base"     # GPT-4o / o-series family encoding
_QWEN_FALLBACK_MODEL = "Qwen/Qwen2.5-7B"     # public Qwen tokenizer on HF

# Proxy used for the no-local-tokenizer families (Anthropic, Google). We count with
# a real BPE (o200k_base) and scale. 1.0 = report the raw proxy count. These are
# honest placeholders you can calibrate against ground-truth API counts on your own
# corpus; we do NOT invent precise factors we can't back.
_CALIBRATION = {"anthropic": 1.0, "google": 1.0}

_CHARS_PER_TOKEN = 4.0   # last-resort heuristic when no tokenizer lib is available


def resolve_family(model: str) -> str:
    m = model.lower()
    for hint, fam in _FAMILY_HINTS:
        if hint in m:
            return fam
    raise ValueError(
        f"cannot resolve a model family from {model!r}; "
        f"expected a name containing one of: gpt/openai, claude/sonnet/opus/haiku, "
        f"gemini/gemma, qwen"
    )


# ---- result ------------------------------------------------------------------

@dataclass
class TokenResult:
    tokens: int
    model: str
    family: str
    method: str          # e.g. "tiktoken:o200k_base", "hf:Qwen/Qwen2.5-7B", "heuristic:chars/4"
    exact: bool          # True only when the model's real tokenizer was used
    notes: list = field(default_factory=list)

    def __str__(self):
        flag = "exact" if self.exact else "APPROX"
        s = f"{self.tokens} tokens  [{flag}]  {self.model} ({self.family}) via {self.method}"
        for n in self.notes:
            s += f"\n  note: {n}"
        return s


# ---- backends (lazy) ---------------------------------------------------------

def _tiktoken_encoding(model: str | None):
    """Return (encoding, name, matched_exact). matched_exact=False => fell back."""
    import tiktoken
    if model is not None:
        try:
            enc = tiktoken.encoding_for_model(model)
            return enc, enc.name, True
        except KeyError:
            pass
    return tiktoken.get_encoding(_OPENAI_FALLBACK_ENCODING), _OPENAI_FALLBACK_ENCODING, False


def _count_openai(text: str, model: str) -> TokenResult:
    notes = []
    try:
        enc, name, matched = _tiktoken_encoding(model)
    except ImportError:
        return _count_heuristic(text, model, "openai",
                                ["tiktoken not installed; run `uv pip install tiktoken`"])
    if not matched:
        notes.append(f"'{model}' not in tiktoken's model table; used family fallback "
                     f"encoding {name}. Count is an approximation for this version.")
    return TokenResult(len(enc.encode(text)), model, "openai",
                       f"tiktoken:{name}", exact=matched, notes=notes)


def _count_qwen(text: str, model: str) -> TokenResult:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return _count_heuristic(text, model, "qwen",
                                ["transformers not installed; run `uv pip install transformers`"])
    notes = []
    # Try the requested repo id verbatim, then the family fallback.
    candidates = []
    if "/" in model:
        candidates.append((model, True))
    candidates.append((_QWEN_FALLBACK_MODEL, False))
    last_err = None
    for repo, matched in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
        except Exception as e:            # network/unknown-repo/etc.
            last_err = e
            continue
        if not matched:
            notes.append(f"'{model}' not resolvable as an HF repo; used family fallback "
                         f"tokenizer {repo}. Exact only if it matches your Qwen version.")
        ids = tok.encode(text, add_special_tokens=False)
        return TokenResult(len(ids), model, "qwen", f"hf:{repo}", exact=matched, notes=notes)
    return _count_heuristic(text, model, "qwen",
                            [f"could not load any Qwen tokenizer ({last_err}); "
                             f"first use needs network to fetch {_QWEN_FALLBACK_MODEL}"])


def _count_proxy(text: str, model: str, family: str) -> TokenResult:
    """Anthropic / Google: no (full) local tokenizer -> BPE proxy * calibration."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding(_OPENAI_FALLBACK_ENCODING)
    except ImportError:
        return _count_heuristic(text, model, family,
                                ["tiktoken not installed; run `uv pip install tiktoken`"])
    raw = len(enc.encode(text))
    factor = _CALIBRATION.get(family, 1.0)
    est = int(round(raw * factor))
    who = "Anthropic" if family == "anthropic" else "Google"
    return TokenResult(
        est, model, family,
        f"proxy:{_OPENAI_FALLBACK_ENCODING}*{factor}", exact=False,
        notes=[f"{who} ships no full local tokenizer; this is a BPE proxy estimate "
               f"(raw {raw} x{factor}). For the true count use the vendor count_tokens API, "
               f"or calibrate _CALIBRATION['{family}'] against it on your corpus."],
    )


def _count_heuristic(text: str, model: str, family: str, extra_notes) -> TokenResult:
    est = int(round(len(text) / _CHARS_PER_TOKEN))
    notes = list(extra_notes) + [f"pure chars/{_CHARS_PER_TOKEN:g} heuristic; rough."]
    return TokenResult(est, model, family, f"heuristic:chars/{_CHARS_PER_TOKEN:g}",
                       exact=False, notes=notes)


# ---- front end ---------------------------------------------------------------

def count_tokens(text: str, model: str) -> TokenResult:
    """Count input tokens for `text` under `model`, using that model's tokenizer
    when a local one exists (OpenAI, Qwen) and a flagged approximation otherwise
    (Anthropic, Google). Returns a TokenResult; `.tokens` is the integer count."""
    family = resolve_family(model)
    if family == "openai":
        return _count_openai(text, model)
    if family == "qwen":
        return _count_qwen(text, model)
    return _count_proxy(text, model, family)   # anthropic / google


# ---- CLI ---------------------------------------------------------------------

def _read_input(args) -> str:
    if args.stdin:
        return sys.stdin.read()
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            return f.read()
    if args.text:
        return " ".join(args.text)
    # default: read stdin if piped, else error
    if not sys.stdin.isatty():
        return sys.stdin.read()
    sys.exit("no input: pass TEXT, --file PATH, or --stdin")


def main(argv=None):
    p = argparse.ArgumentParser(description="Count input tokens across model families (local-only).")
    p.add_argument("-m", "--model", help="model name, e.g. gpt-6, claude-sonnet-5, gemini-3.5, qwen-3.8")
    p.add_argument("text", nargs="*", help="text to count (or use --file/--stdin)")
    p.add_argument("--file", help="read input text from this file")
    p.add_argument("--stdin", action="store_true", help="read input text from stdin")
    p.add_argument("--count-only", action="store_true", help="print just the integer")
    p.add_argument("--list", action="store_true", help="show family routing and exit")
    args = p.parse_args(argv)

    if args.list:
        print("family routing (substring match, first wins):")
        for hint, fam in _FAMILY_HINTS:
            print(f"  *{hint}* -> {fam}")
        print("\nexactness:")
        print("  openai  -> tiktoken           (exact; family fallback -> o200k_base if version unknown)")
        print("  qwen    -> HF AutoTokenizer   (exact; family fallback -> Qwen/Qwen2.5-7B if version unknown)")
        print("  anthropic/google -> BPE proxy (APPROX by design: no full local tokenizer)")
        return 0

    if not args.model:
        p.error("the following argument is required: -m/--model")

    text = _read_input(args)
    try:
        res = count_tokens(text, args.model)
    except ValueError as e:
        sys.exit(str(e))
    if args.count_only:
        print(res.tokens)
    else:
        print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
