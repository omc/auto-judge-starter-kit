# HANDOFF — RAG26 Pairwise Summary Judge (full run)

Status as of 2026-09-11. Goal: an LLM-judged **pairwise summary tournament** over all
119 RAG26 topics for **one model**, producing a win-rate + Bradley-Terry leaderboard.

## TL;DR — where it stands
- **78 / 119 topics scored** (partial), from **249,419 cached comparisons** (1 invalid — parsing is essentially perfect).
- **~$149 actually spent** (OpenRouter billing, authoritative) getting to ~64% of the full plan.
- Stopped at a **key spend-limit wall** (403) on the last ~41 topics. Nothing lost — the disk cache holds all completed calls; resuming re-uses them.
- Partial leaderboard + artifacts already written (see below). Top of the board is stable and matches the 1-topic pilot: teams **T569** (lars, ariel, edith, carmen) and **T300** (luca, hana, xavi) dominate.

## To FINISH the full 119 topics later
1. **Raise the OpenRouter key's total limit by ≥ ~$95** above current usage (see cost note — $40 is NOT enough; earlier estimates were low).
2. Resume (cache-skips the ~242k done, calls only the remaining ~138k, then scores):
   ```bash
   set -a; source ./.env; set +a
   auto-judge run --workflow judges/bonsai_judge/workflow.pairwise.yml --variant full \
     --rag-responses data/rag26/runs/generation/ \
     --rag-topics data/rag26/topics/trec_rag_2026_queries.jsonl --out-dir ./output-pairwise/
   ```
   or `bash temp/pairwise_poll.sh full` (guarded/idempotent; marks `output-pairwise/DONE_full` on success).
3. **Expect ~2.5h** at the measured ~14.4 req/s (48 concurrency). It scores automatically when every call is cached.
4. Any time, re-score whatever is cached (no API calls): `python3 temp/pairwise_score_partial.py`.

## COST REALITY (read before spending — prior estimates were wrong)
- **Measured actual ≈ $0.00058 per comparison** (reconciled against the $149 billing), **~1.9× my token-proxy math**. Root causes: (a) real prompt ≈ **1,997 Gemini tokens/call** not the 1,800 gpt-6 answer-proxy — the +~200 is the instruction/query/"Summary A/B" scaffolding I omitted; (b) `gemini-3.5-flash-lite` almost certainly bills higher than the assumed $0.15/$1.25 (that was a *2.5*-flash-lite figure) plus OpenRouter markup; (c) wasted duplicate batches during debugging.
- **Remaining ~138k calls → ~$80–85** (provision **~$95** for the right-skewed input tail; p95 ≈ 2,910 tok).
- **Full run all-in ≈ $235** for this one model. **VERIFY the model's real OpenRouter per-token price before committing** — do not trust the nominal $0.15/$1.25.

## Hard constraints learned (OpenRouter)
- The env model `google/gemini-3.5-flash-lite:batch` — the **`:batch` suffix is batch-ONLY**; realtime calls with it **404**. Realtime uses the plain slug (`.replace(":batch","")`, done automatically in `submit_mode=live`).
- **Batch API caps: 5,000 requests/batch (413 over), 20,000 in-flight requests/entity (429 over).** Batch latency was **~10–12h per wave** → the batch tier CANNOT meet a next-day deadline for 700k calls (~35 waves ≈ weeks). Realtime (~14 req/s) is the only fast path.
- **No batch cancel endpoint** exists (POST/DELETE/PATCH all 404). Submitted batches run to completion.
- **Per-key "total limit"** is separate from account credit; hitting it → `403 Key limit exceeded` on *everything* (batch and realtime).
- Harness `run_in_background` tasks get **reaped** (sometimes within minutes). Use **`nohup … & disown`** for durability; a **watchdog cron** running the guarded poller auto-resumes after a kill (pgrep guard prevents double-spend).

## Methodology (state this when reporting)
- **Single-direction**: each cross-team pair judged **once**, with A/B orientation chosen by a deterministic hash of `(topic, runA, runB)` so neither slot is systematically favored (position bias randomized, not cancelled). This halves volume vs the ordered `N*(N-1)` design — the concession made to hit the deadline/budget. `direction: ordered` restores the full both-directions design.
- Cross-team only (a team's runs are never compared to each other). Chosen summary = +1 point.
- Ranking = **win-rate** (points / comparisons played; comparable across runs despite unequal opponent counts) + **Bradley-Terry** strength (schedule-corrected; MM fit). Raw wins alone are biased by team size.
- Output constrained to a single `A`/`B` token (max_tokens 8, temp 0); prose is flagged unparseable, not guessed.

## Scale / arithmetic
- 83 runs, 25 teams, 119 topics. Ordered plan = 760,886; **single-direction plan = 380,443**.
- Per-topic ~2,900–3,200 comparisons; groups of 20 topics (`topics_per_batch`) ≈ 60–64k each, 6 groups.
- ~7.5% of comparisons dedupe (identical summaries share a cache key).

## Key files
- `judges/bonsai_judge/pairwise.py` — the judge. Variants driven by `workflow.pairwise.yml`.
  - `_plan_comparisons(..., direction)`, `_make_backend(..., submit_mode, live_*)` (live-tunes: plain slug, 48 concurrency, `max_failures=None`, resumable).
  - Scoring: `_write_overall`, `_write_bradley_terry`, `_score_and_write`.
- `judges/bonsai_judge/openrouter_batch.py` — OpenRouter Batch adapter (submit/harvest split, dup-guard signature index, 20k in-flight throttle). **Not used by the realtime path**, but ready if you ever go batch.
- `judges/bonsai_judge/pairwise_status.py` — cheap batch status/harvest CLI (batch mode only).
- `judges/bonsai_judge/workflow.pairwise.yml` — variants: `pilot` (1 topic, batch), `scale` (5, batch), **`full` (all 119, realtime, single-direction)**. Keep `model`/`max_tokens: 8`/`temperature: 0` FIXED (cache-key inputs).
- `temp/pairwise_poll.sh <variant>` — guarded runner/resumer (used by watchdog cron).
- `temp/pairwise_score_partial.py` — cache-only scorer (no API); writes the partial leaderboard.
- `temp/answer_token_counts.py`, `temp/tokenlen.py` — token accounting (gpt-6/qwen; NOTE: add prompt scaffolding + a Gemini factor for real cost).

## State locations
- **Cache (the resumable asset)**: `cache/minima_llm.db` (sqlite; key = sha256 of model+messages+temp+max_tokens). Live results are under the plain slug; earlier batch results under `:batch` (different keys — they do NOT carry over to realtime).
- **Artifacts**: `output-pairwise/bonsai_pairwise.pairwise/` → `leaderboard_overall.csv`, `bradley_terry.csv`, `run_manifest_partial.json` (flagged PARTIAL 78/119).
- **Excluded topics**: the 41 not-yet-complete are the tail of sorted topic order (rag2026-62 … onward). Not cherry-picked.
- **Env**: `.env` = `OPENAI_BASE_URL` (https://openrouter.ai/api/v1), `OPENAI_MODEL` (…:batch), `OPENAI_API_KEY`, `CACHE_DIR` (./cache). Never echo values.

## Multi-model note
"An entire LLM" here = **one model** (gemini-3.5-flash-lite). For another model: change `OPENAI_MODEL`, run `--variant full` again. Cache keys include the model, so each model is a fresh ~$115–235 run — budget per model accordingly.

## Leftover cleanup
- Several debug/duplicate OpenRouter batches were submitted and billed (couldn't be cancelled). They're done; ignore. No crons remain (pilot/scale/full watchdogs all deleted).
