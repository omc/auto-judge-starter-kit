#!/bin/bash
auto-judge run \
    --workflow judges/bert_leader/workflow.ragtime.yml \
    --rag-responses /c/dev/trec/2026-ragtime/runs/repgen \
    --rag-topics /c/dev/trec/2026-ragtime/topics/topics.all.2026.v0625-fix.request.jsonl \
    --out-dir ./output-bert-ragtime/
