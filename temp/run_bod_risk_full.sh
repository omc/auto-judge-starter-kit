#!/usr/bin/env bash
# Full BOD x Hallucination Risk runs (every topic -- the submission). No new Jev calls expected
# (BOD grades are cached); spaCy tags every cited document. Usage: bash temp/run_bod_risk_full.sh
set -u
cd "$(dirname "$0")/.."
set -a; source ./.env; set +a
source .venv/bin/activate
export PYTHONUNBUFFERED=1
WF=judges/bonsai_judge/workflow.bod_risk.yml
for ds in ragtime26 rag26; do   # reuse the submitted BOD rubric byte for byte
  mkdir -p output-full-$ds-bodrisk
  cp -n output-full-$ds-bod/bonsai_bod.nuggets.jsonl output-full-$ds-bodrisk/bonsai_bod_risk.nuggets.jsonl
done
auto-judge run --workflow $WF --variant full --rag-responses data/ragtime26/runs/repgen/ \
  --rag-topics data/ragtime26/topics/topics.all.2026.v0625-fix.request.jsonl \
  --out-dir ./output-full-ragtime26-bodrisk/ > temp/bod_risk_full_ragtime26.log 2>&1; echo "ragtime26 exit $?"
grep -E "\[bod" temp/bod_risk_full_ragtime26.log | grep -v tagged
auto-judge run --workflow $WF --variant full --rag-responses data/rag26/runs/generation/ \
  --rag-topics data/rag26/topics/trec_rag_2026_queries.jsonl \
  --out-dir ./output-full-rag26-bodrisk/ > temp/bod_risk_full_rag26.log 2>&1; echo "rag26 exit $?"
grep -E "\[bod" temp/bod_risk_full_rag26.log | grep -v tagged
