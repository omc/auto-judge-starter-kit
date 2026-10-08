#!/usr/bin/env bash
# Depth-adaptivity test: same report pairs judged under the original (broad) request
# and a narrowed one-sub-question request; better / appropriate / proportionate pairs.
cd "$(dirname "$0")/.."
set -a; source ./.env; set +a
source .venv/bin/activate
WF=judges/bonsai_judge/workflow.pairwise_jev.yml
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] start $name"
         if "$@" > temp/prop_$name.log 2>&1; then echo "[$(date +%H:%M:%S)] done $name"
         else echo "[$(date +%H:%M:%S)] FAILED $name (see temp/prop_$name.log)"; fi; }
R26="--rag-responses data/rag26/runs/generation/ --topic rag2026-1 --topic rag2026-107"
RT26="--rag-responses data/ragtime26/runs/repgen/ --topic 2041 --topic 2046"
step rag26_broad      auto-judge run --workflow $WF --variant noul_proportionate_test $R26 --rag-topics data/rag26/topics/trec_rag_2026_queries.jsonl --out-dir ./output-prop-rag26-broad/
step rag26_narrow     auto-judge run --workflow $WF --variant noul_proportionate_test $R26 --rag-topics temp/narrow_topics_rag26.jsonl --out-dir ./output-prop-rag26-narrow/
step ragtime26_broad  auto-judge run --workflow $WF --variant noul_proportionate_test $RT26 --rag-topics data/ragtime26/topics/topics.all.2026.v0625-fix.request.jsonl --out-dir ./output-prop-ragtime26-broad/
step ragtime26_narrow auto-judge run --workflow $WF --variant noul_proportionate_test $RT26 --rag-topics temp/narrow_topics_ragtime26.jsonl --out-dir ./output-prop-ragtime26-narrow/
echo "[$(date +%H:%M:%S)] SUITE COMPLETE"
