Bert Baseline
---
## Overview
A simple BERT approach than can be run offline. The topic is encoded then all sentences from each run:topic combo.  Normalized dot products are calculated to determine similarity which are then averaged and sorted to generate a leaderboard across all runs.

## Setup
You will need to setup a venv with torch/sentence-transformer + CUDA. See `requirements.txt`.

Drop the script at the top level of the appropriate TREC dataset then run.


