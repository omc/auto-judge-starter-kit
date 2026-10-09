#!/usr/bin/env bash
# Jev study suite on PERMITTED topics only (data policy, CLAUDE.md), budget-capped.
#   permitted window = first 10 topics of each topics file (rag26 rag2026-0..9, ragtime26 2000..2009)
#   dev variants carry `dev_topics` (judge refuses anything else); probes read temp/jev_dataset.py.
# Budget guard: before each paid step, the OpenRouter key's remaining balance must cover the
# step's estimate plus FLOOR_USD; otherwise the step is SKIPPED (reported). Estimates are
# conservative (measured $/call x calls, minus known cache hits).
# Not re-run here (no decision depends on them; see temp/WINDOW_STUDY_PLAN.md): instruction
# checks, Noul-better pilot, Choice determinism, Choice-only padding.
# The submitted form's slot bias / ties / spread / length / Gemini agreement come at $0 from
# the full runs' window topics: python temp/jev_full_analysis.py <full-run out-dir> <gemini>.
# Usage: bash temp/run_window_suite.sh rag26|ragtime26
set -u
DS=${1:?usage: run_window_suite.sh rag26|ragtime26}
FLOOR_USD=${FLOOR_USD:-10}
cd "$(dirname "$0")/.."
set -a; source ./.env; set +a
source .venv/bin/activate
export JEV_DATASET=$DS PYTHONUNBUFFERED=1
case $DS in
  rag26)     R="--rag-responses data/rag26/runs/generation/ --rag-topics data/rag26/topics/trec_rag_2026_queries.jsonl" ;;
  ragtime26) R="--rag-responses data/ragtime26/runs/repgen/ --rag-topics data/ragtime26/topics/topics.all.2026.v0625-fix.request.jsonl" ;;
  *) echo "unknown dataset $DS"; exit 2 ;;
esac
# conservative per-step estimates in USD (bash-3 compatible lookup)
cost() { case "$DS:$1" in
  rag26:prompts) echo 3.5;; rag26:choice) echo 5.0;; rag26:ident) echo 0.4;; rag26:determ) echo 0.8;;
  rag26:padnoul) echo 2.0;; rag26:gemini) echo 6.5;; rag26:padgem) echo 2.5;;
  ragtime26:prompts) echo 1.2;; ragtime26:choice) echo 2.0;; ragtime26:ident) echo 0.3;; ragtime26:determ) echo 0.3;;
  ragtime26:padnoul) echo 2.0;; ragtime26:gemini) echo 0;; ragtime26:padgem) echo 3.5;;
  *) echo 999;; esac; }
remaining() { curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENAI_API_KEY" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['limit_remaining'])"; }
WF=judges/bonsai_judge/workflow.pairwise_jev.yml
O=output-win-$DS
step() { local name=$1 key=$2; shift 2
  local rem; rem=$(remaining)
  if ! python3 -c "import sys; sys.exit(0 if float('$rem') >= $(cost $key) + $FLOOR_USD else 1)"; then
    echo "[$(date +%H:%M:%S)] SKIPPED $name: remaining \$$rem < estimate \$$(cost $key) + floor \$$FLOOR_USD"; return; fi
  echo "[$(date +%H:%M:%S)] start $name (est \$$(cost $key), remaining \$$rem)"
  if "$@" > temp/win_${DS}_$name.log 2>&1; then echo "[$(date +%H:%M:%S)] done $name"
  else echo "[$(date +%H:%M:%S)] FAILED $name (see temp/win_${DS}_$name.log)"; fi; }
# decision 1 (adapts?) runs separately: bash temp/run_proportionate_test.sh $DS
step prompts      prompts auto-judge run --workflow $WF --variant prompts $R --out-dir ./$O-jev-prompts/   # Choice + wording variants; padding targets come from here
step choice_pilot choice  auto-judge run --workflow $WF --variant pilot   $R --out-dir ./$O-jev/            # decision 5: ranking vs Choice, both orientations
step identical    ident   bash -c 'python temp/jev_probes.py identical && python temp/jev_probes.py identical --question noul_a_proportionate && python temp/jev_probes.py identical --question noul_b_proportionate'   # decision 2: ties
step determinism  determ  python temp/jev_probes.py determinism --question mirror                               # decision 5: mirror flips
step padding_noul padnoul python temp/jev_padding.py run --judge noul                                            # decision 4: submitted form
if [ "$DS" = rag26 ]; then
  step gemini     gemini  auto-judge run --workflow judges/bonsai_judge/workflow.pairwise.yml --variant dev $R --out-dir ./$O-gemini/   # rag2026-0..6 cached; 7..9 new
fi
step padding_gem  padgem  python temp/jev_padding.py run --judge gemini                                         # comparison only (first 3 topics)
echo "[$(date +%H:%M:%S)] SUITE COMPLETE ($DS) | remaining \$$(remaining)"
