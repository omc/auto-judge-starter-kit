#!/usr/bin/env python3
"""Cheap status/harvest for pairwise OpenRouter batches -- no report loading, no judge.

Reads the openrouter_batch_*.json state files under the cache dir, queries each
batch's status, and (by default) harvests any completed ones into the prompt cache.
Use it to poll progress between the decoupled submit and the final scoring run.

  # show status + harvest completed batches (one pass):
  python -m judges.bonsai_judge.pairwise_status
  # just show status, don't touch the cache:
  python -m judges.bonsai_judge.pairwise_status --no-harvest
  # block until everything finishes:
  python -m judges.bonsai_judge.pairwise_status --wait

Reads endpoint/model/cache_dir from the env (OPENAI_BASE_URL/MODEL/API_KEY, CACHE_DIR),
same as the judge. Prints one line per batch and an overall ready/not-ready verdict.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

from minima_llm import MinimaLlmConfig, OpenAIMinimaLlm
from .openrouter_batch import OpenRouterBatch


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.environ.get("CACHE_DIR"),
                    help="dir holding openrouter_batch_*.json (default: $CACHE_DIR)")
    ap.add_argument("--no-harvest", action="store_true",
                    help="only report status; do not write results to cache")
    ap.add_argument("--wait", action="store_true", help="block until all batches finish")
    args = ap.parse_args(argv)

    cfg = MinimaLlmConfig.from_env()
    cache_dir = args.cache_dir or cfg.cache_dir
    if not cache_dir:
        sys.exit("no cache dir: set CACHE_DIR or --cache-dir")
    backend = OpenAIMinimaLlm(cfg.with_model(cfg.model))
    cache = backend._ensure_cache()

    state_files = sorted(glob.glob(str(Path(cache_dir) / "openrouter_batch_*.json")))
    if not state_files:
        sys.exit(f"no batch state files under {cache_dir}")

    all_ready = True
    tot = {"chunks": 0, "completed": 0, "failed": 0, "pending": 0, "terminal": 0}
    for sf in state_files:
        prefix = Path(sf).name[len("openrouter_batch_"):-len(".json")]
        orb = OpenRouterBatch(cfg, cache, state_dir=Path(cache_dir), prefix=prefix)
        if args.no_harvest:
            # inspect only: read state + query statuses without populating cache
            import json as _json
            st = _json.loads(Path(sf).read_text())
            line = []
            for c in st.get("chunks", []):
                try:
                    s = orb._status(c["batch_id"]).get("status", "?")
                except Exception as e:
                    s = f"err:{e}"
                line.append(f"c{c['index']}={s}")
                if s not in ("completed", "failed", "expired", "cancelled"):
                    all_ready = False
            print(f"{prefix}: " + ", ".join(line))
        else:
            ready, hv = orb.harvest(wait=args.wait)
            all_ready = all_ready and ready
            for k in tot:
                tot[k] += hv.get(k, 0)
            print(f"{prefix}: ready={ready} {hv}")

    print(f"\nOVERALL: {'READY -> re-run the judge to score' if all_ready else 'NOT READY yet'}")
    if not args.no_harvest:
        print(f"  totals: {tot}")
    return 0 if all_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
