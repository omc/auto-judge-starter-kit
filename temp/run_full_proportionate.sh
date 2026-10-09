#!/usr/bin/env bash
# Full Jev mirrored-Noul (proportionate) tournaments: ragtime26 first, then rag26.
# Resumable: re-run this script; completed calls are served from the cache.
cd "$(dirname "$0")/.."
set -a; source ./.env; set +a
source .venv/bin/activate
export PYTHONUNBUFFERED=1   # live progress in the logs
WF=judges/bonsai_judge/workflow.pairwise_jev.yml
run() { local name=$1; shift; echo "[$(date '+%F %T')] start $name"
        if auto-judge run --workflow $WF --variant full_noul_proportionate "$@" > temp/full_prop_$name.log 2>&1
        then echo "[$(date '+%F %T')] done $name"; else echo "[$(date '+%F %T')] FAILED $name (exit $?, see temp/full_prop_$name.log)"; fi; }
run ragtime26 --rag-responses data/ragtime26/runs/repgen/ \
    --rag-topics data/ragtime26/topics/topics.all.2026.v0625-fix.request.jsonl --out-dir ./output-ragtime26-jev-full-prop/
run rag26 --rag-responses data/rag26/runs/generation/ \
    --rag-topics data/rag26/topics/trec_rag_2026_queries.jsonl --out-dir ./output-pairwise-jev-full-prop/
echo "[$(date '+%F %T')] ALL DONE"
