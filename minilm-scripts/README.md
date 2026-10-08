Bert Baseline
---
## Overview
A simple BERT approach than can be run offline. The topic is encoded then all sentences from each run:topic combo.  Normalized dot products are calculated to determine similarity which are then averaged and sorted to generate a leaderboard across all runs.

## Setup
You will need to setup a venv with torch/sentence-transformer + CUDA. See `requirements.txt`.

## Generating Scores
Drop an eval python script at the top level of the appropriate TREC dataset then run each to generate a bank of scores used for leaderboard generation.

## Generating Leaderboard
Update paths in the bash scripts/bert\_leader workflows to point at your local data
