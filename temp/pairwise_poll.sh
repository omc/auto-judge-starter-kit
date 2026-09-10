#!/usr/bin/env bash
# Pairwise self-completion poller. Usage: pairwise_poll.sh [variant]  (default: scale)
# Each run invokes the judge, which: drips more batches up to the 20k in-flight cap,
# harvests completed ones into the cache, and scores IFF every comparison is ready.
# Idempotent; resubmits nothing (dedup + dup-guard). rc==0 from the judge == scored.
set -u
cd /Users/max/projects/bonsai-auto-judge || exit 2
VARIANT="${1:-scale}"
DONE="output-pairwise/DONE_${VARIANT}"
[ -f "$DONE" ] && { echo "ALREADY_DONE"; exit 0; }
# Never overlap: if a run for this variant is already going, do nothing. This lets a
# cron safety-net resume after a kill without double-spending on concurrent live calls.
if pgrep -f "workflow.pairwise.yml --variant ${VARIANT} " >/dev/null 2>&1; then
  echo "ALREADY_RUNNING"; exit 0
fi

set -a; source ./.env; set +a
source .venv/bin/activate 2>/dev/null
export PYTHONUNBUFFERED=1

# One drip+harvest(+score-if-ready) pass. Judge exits 0 only when it scored;
# it raises SystemExit (non-zero) with a NOT-READY message otherwise.
auto-judge run --workflow judges/bonsai_judge/workflow.pairwise.yml --variant "$VARIANT" \
  --rag-responses data/rag26/runs/generation/ \
  --rag-topics data/rag26/topics/trec_rag_2026_queries.jsonl \
  --out-dir ./output-pairwise/ > "output-pairwise/score_${VARIANT}.log" 2>&1
rc=$?
# surface the judge's own progress/summary line(s)
grep -iE "created batch|adopted|deferring|group [0-9]+:|valid /|NOT READY" \
  "output-pairwise/score_${VARIANT}.log" | tail -4

if [ "$rc" -eq 0 ]; then
  touch "$DONE"; echo "SCORED_OK"
else
  echo "STILL_PENDING"
fi
