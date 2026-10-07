#!/usr/bin/env bash
# Noul probe suite: instruction check (judge), determinism (2 passes), padding.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source ./.env; set +a
source .venv/bin/activate
auto-judge run --workflow judges/bonsai_judge/workflow.pairwise_jev.yml --variant noul_instruction_check \
  --rag-responses data/rag26/runs/generation/ --rag-topics data/rag26/topics/trec_rag_2026_queries.jsonl \
  --out-dir ./output-pairwise-jev-noul-icheck/ > temp/jev_noul_icheck.log 2>&1
echo "icheck done"
python temp/jev_probes.py determinism --topic rag2026-100 --question mirror > temp/jev_noul_determinism.log 2>&1
echo "determinism done"
python temp/jev_padding.py run --judge noul > temp/jev_noul_padding.log 2>&1
echo "padding done"
