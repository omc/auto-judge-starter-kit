from pathlib import Path
from sentence_transformers import SentenceTransformer
from statistics import mean
import json
import time
import torch

RUNS_FOLDER = Path('runs/generation')
TOPICS_DATASET = 'topics/trec_rag_2026_queries.jsonl'

runs = {}
topics = {}

# Scores keyed by run_id, includes list of mean sims for all topics for summary
scores = {}

# Might fall back to spotify distilibert but we'll see how this performs
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device='cuda',
)

# main brains of this operation
# encodes the topic then does normalized dot product for quicker cosine calc
def evaluate(topic_key, submissions):
    topic_meta = topics[topic_key]

    # Encode the problem statement
    base_embed = model.encode(
            [topic_meta['title']],
            convert_to_tensor=True,
            normalize_embeddings=True
    )

    # Encode response sentences in prep for mean sim
    for s in submissions:
        sentences = [item['text'] for item in s['responses']]
        resp_embed = model.encode(
                sentences,
                convert_to_tensor=True,
                normalize_embeddings=True
        )

        sims = []
        for entry in resp_embed:
            sims.append(torch.dot(base_embed[0], entry).item())

        # Prep array of scores if it doesn't exist yet
        if s['metadata']['run_id'] not in scores:
            scores[s['metadata']['run_id']] = []

        score = mean(sims) if len(sims) > 0 else 0.0
        scores[s['metadata']['run_id']].append(score)
        #print('Result: ', score)


# Topics keyed by request_id to full record
print('Reading in topic metadata')
with open(TOPICS_DATASET, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        topics[record['request_id']] = record

# Read in repgen data, indexed by topic which each teams submission underneath
print('Reading repgen data for ragtime task')
for run in RUNS_FOLDER.iterdir():
    if run.is_file():
        with run.open(encoding='utf-8') as src:
            for line in src:
                results = json.loads(line)

                if results['metadata']['topic_id'] not in runs:
                    runs[results['metadata']['topic_id']] = []

                runs[results['metadata']['topic_id']].append(results)

print('Ready for processing')
total_start_tm = time.perf_counter()

for topic_key in topics:
    start_tm = time.perf_counter()
    submissions = runs[topic_key]
    evaluate(topic_key, submissions)
    elapsed = time.perf_counter() - start_tm
    print(f'Done evaluating topic {topic_key}, {elapsed:.4f}s', flush=True)

total_elapsed = time.perf_counter() - total_start_tm
print()
print(f'Done, total time {total_elapsed:.4f}s')

sorted_scores = sorted(
        scores.items(),
        key=lambda item: mean(item[1]),
        reverse=True
)

print('Final results:')
print('--------------')
for key, values in sorted_scores:
    print(f"{key}: {mean(values):.6f}")
