#!/usr/bin/env bash
# Replicate the full Jev study (Choice + Noul + Gemini) on ragtime26 repgen.
# Topics: pilot 2000; 3-topic runs 2000-2002; determinism 2003; off-topic source 2050.
# Each step logs to temp/rt26_<step>.log; a failed step is reported and the suite continues.
cd "$(dirname "$0")/.."
set -a; source ./.env; set +a
source .venv/bin/activate
export JEV_DATASET=ragtime26
R="--rag-responses data/ragtime26/runs/repgen/ --rag-topics data/ragtime26/topics/topics.all.2026.v0625-fix.request.jsonl"
WF=judges/bonsai_judge/workflow.pairwise_jev.yml
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] start $name"
         if "$@" > temp/rt26_$name.log 2>&1; then echo "[$(date +%H:%M:%S)] done $name"
         else echo "[$(date +%H:%M:%S)] FAILED $name (see temp/rt26_$name.log)"; fi; }
step choice_pilot   auto-judge run --workflow $WF --variant pilot $R --out-dir ./output-ragtime26-jev/
step prompts        auto-judge run --workflow $WF --variant prompts $R --out-dir ./output-ragtime26-jev-prompts/
step icheck         auto-judge run --workflow $WF --variant instruction_check $R --out-dir ./output-ragtime26-jev-icheck/
step noul_pilot     auto-judge run --workflow $WF --variant noul_pilot $R --out-dir ./output-ragtime26-jev-noul/
step noul_icheck    auto-judge run --workflow $WF --variant noul_instruction_check $R --out-dir ./output-ragtime26-jev-noul-icheck/
step gemini         auto-judge run --workflow judges/bonsai_judge/workflow.pairwise.yml --variant full -J max_topics=3 $R --out-dir ./output-ragtime26-gemini/
step identical      python temp/jev_probes.py identical
step identical_na   python temp/jev_probes.py identical --question noul_a_better
step identical_nb   python temp/jev_probes.py identical --question noul_b_better
step determinism    python temp/jev_probes.py determinism
step determinism_m  python temp/jev_probes.py determinism --question mirror
step padding_jev    python temp/jev_padding.py run --judge jev
step padding_noul   python temp/jev_padding.py run --judge noul
step padding_gemini python temp/jev_padding.py run --judge gemini
echo "[$(date +%H:%M:%S)] SUITE COMPLETE"
