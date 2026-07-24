---
name: transcript-summarizer
description: Summarize a weekly meeting transcript into a structured progress note. Use when the user wants to summarize, digest, or write up a meeting transcript, or invokes /transcript-summarizer with an ISO date (e.g. 2026-07-24). Reads transcripts/raw/[date].txt and writes transcripts/[date].md.
---

# Transcript summarizer

Turn one weekly meeting transcript into a structured markdown summary. These summaries accumulate over the project and are later read **together** to track progress and source a research paper, so consistency of structure across weeks matters as much as the content of any single week.

## Input

- **Argument:** an ISO date, e.g. `2026-07-24`. If none is given, list `transcripts/raw/` and ask which date to summarize.
- **Source:** `transcripts/raw/<date>.txt`. Lines look like `Speaker Name [MM:SS]: text`. If the file is missing, list what's in `transcripts/raw/` and stop — do not invent content.
- **Output:** `transcripts/<date>-summary.md`. If it already exists, show the user and confirm before overwriting.

## Procedure

1. Read the whole transcript file. Do not summarize from a preview or a partial read — action items and decisions are often at the very end (e.g. "homework").
2. Identify the distinct speakers from the `Name [MM:SS]:` prefixes; those are the attendees.
3. Read for substance, discounting chit-chat, connection issues, and pleasantries. Attribute decisions and action items to the person who owns them.
4. Write `transcripts/<date>.md` using the template below.
5. Report a 2–3 sentence recap and the action-item count. Do not edit the raw transcript.

## Conventions

- **Faithful, not generous.** Only record what was actually said. Mark anything uncertain with "(unclear)" rather than guessing. If the transcript is garbled (auto-transcription errors are common — names, tools, and jargon get mangled), note it and give your best reading. Never fabricate decisions, numbers, or owners.
- **Attribute action items** to a named owner when the transcript makes it clear; use "(unassigned)" otherwise. Include a due/target if stated.
- **Findings** = things the team learned or established as true (about the data, the competition, the tooling, results), distinct from **Decisions** (choices made) and **Action items** (future work).
- **Research-paper hooks:** capture anything methodologically relevant — approaches tried, why, results, dead ends, open research questions. This is what the paper is built from; err toward recording it.
- Keep names as spelled in the transcript. Convert `[MM:SS]` references to plain prose; don't litter the summary with timestamps (a couple for pivotal moments is fine).
- Write in past tense, terse and factual. No filler.

## Output template

```markdown
# Meeting Summary — <date>

**Attendees:** <names>
**Duration:** ~<MM> min (from last timestamp)

## TL;DR

<2–4 sentence overview of what the meeting was about and where the project stands after it.>

## Key discussion notes

- <topic-organized bullets of substantive discussion>

## Decisions

- <decision — owner/rationale if given> <!-- omit section if none -->

## Findings

- <what the team learned or established> <!-- omit section if none -->

## Action items

- [ ] <task> — **<owner>** (<due if stated>)

## Open questions / risks

- <unresolved questions, blockers, risks raised> <!-- omit section if none -->

## Notes for the research paper

- <methodology, approaches, results, dead ends, research questions worth carrying forward>
```

Omit any section that genuinely has no content rather than padding it. Keep **Attendees**, **TL;DR**, **Key discussion notes**, and **Action items** in every summary so weeks stay comparable.

## Nicknames

- Max Irwin: Max
- Nate Day: Nate
- Allison Zadrozny: Alli
