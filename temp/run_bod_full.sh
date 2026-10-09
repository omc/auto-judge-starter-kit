#!/usr/bin/env bash
# Full BOD runs (every topic -- the submission). Usage: bash temp/run_bod_full.sh
set -u
cd "$(dirname "$0")/.."
set -a; source ./.env; set +a
source .venv/bin/activate
export PYTHONUNBUFFERED=1
WF=judges/bonsai_judge/workflow.bod.yml
remaining() { curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENAI_API_KEY" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['limit_remaining'])"; }
echo "start | remaining \$$(remaining)"
auto-judge run --workflow $WF --variant full --rag-responses data/ragtime26/runs/repgen/ \
  --rag-topics data/ragtime26/topics/topics.all.2026.v0625-fix.request.jsonl \
  --out-dir ./output-full-ragtime26-bod/ > temp/bod_full_ragtime26.log 2>&1; echo "ragtime26 exit $?"
grep "\[bod\]" temp/bod_full_ragtime26.log
auto-judge run --workflow $WF --variant full --rag-responses data/rag26/runs/generation/ \
  --rag-topics data/rag26/topics/trec_rag_2026_queries.jsonl \
  --out-dir ./output-full-rag26-bod/ > temp/bod_full_rag26.log 2>&1; echo "rag26 exit $?"
grep "\[bod\]" temp/bod_full_rag26.log
echo "done | remaining \$$(remaining)"
