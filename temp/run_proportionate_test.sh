#!/usr/bin/env bash
# Depth-adaptivity test (decision 1, temp/WINDOW_STUDY_PLAN.md): the same report pairs judged
# under the original (broad) request and a narrowed one-sub-question request
# (temp/narrow_topics_<ds>.jsonl), with the better and proportionate mirror pairs in one request.
# PERMITTED topics only: the variant's dev_topics limit it to the window (all 10 topics).
# One dataset per call, so ragtime26 (confirmation) is not run before the rag26 decisions
# are recorded. Budget guard as in run_window_suite.sh (skip unless remaining >= est + floor).
# Usage: bash temp/run_proportionate_test.sh rag26|ragtime26
set -u
DS=${1:?usage: run_proportionate_test.sh rag26|ragtime26}
FLOOR_USD=${FLOOR_USD:-10}
cd "$(dirname "$0")/.."
set -a; source ./.env; set +a
source .venv/bin/activate
export PYTHONUNBUFFERED=1
WF=judges/bonsai_judge/workflow.pairwise_jev.yml
case $DS in
  rag26)     RESP=data/rag26/runs/generation/;   TOPICS=data/rag26/topics/trec_rag_2026_queries.jsonl;                EST=3.6 ;;
  ragtime26) RESP=data/ragtime26/runs/repgen/;   TOPICS=data/ragtime26/topics/topics.all.2026.v0625-fix.request.jsonl; EST=1.3 ;;
  *) echo "unknown dataset $DS"; exit 2 ;;
esac
remaining() { curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENAI_API_KEY" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['limit_remaining'])"; }
step() { local name=$1; shift
  local rem; rem=$(remaining)
  if ! python3 -c "import sys; sys.exit(0 if float('$rem') >= $EST + $FLOOR_USD else 1)"; then
    echo "[$(date +%H:%M:%S)] SKIPPED $name: remaining \$$rem < estimate \$$EST + floor \$$FLOOR_USD"; return; fi
  echo "[$(date +%H:%M:%S)] start $name (est \$$EST, remaining \$$rem)"
  if "$@" > temp/win_prop_$name.log 2>&1; then echo "[$(date +%H:%M:%S)] done $name"
  else echo "[$(date +%H:%M:%S)] FAILED $name (see temp/win_prop_$name.log)"; fi; }
step ${DS}_broad  auto-judge run --workflow $WF --variant noul_proportionate_test --rag-responses $RESP \
     --rag-topics $TOPICS                          --out-dir ./output-win-$DS-prop-broad/
step ${DS}_narrow auto-judge run --workflow $WF --variant noul_proportionate_test --rag-responses $RESP \
     --rag-topics temp/narrow_topics_$DS.jsonl     --out-dir ./output-win-$DS-prop-narrow/
echo "[$(date +%H:%M:%S)] DONE ($DS) | remaining \$$(remaining)"
echo "analyse: JEV_DATASET=$DS python temp/jev_proportionate_compare.py \\"
echo "  output-win-$DS-prop-broad/bonsai_pairwise_jev.pairwise/comparisons.jsonl output-win-$DS-prop-narrow/bonsai_pairwise_jev.pairwise/comparisons.jsonl"
