# OMC at TREC 2026 AutoJudge: Methods and Results (notebook paper draft)

**Authors:** Max Irwin <max@onemorecloud.com>, Nathan Day <nday@eezy.com>, Daniel Worley <dworley@gmail.com>

## Abstract

Decision models offer a novel (and cheaper) alternative to the established LLM-as-a-judge paradigm. Jev, the first decision model publicly released in September 2026, changes the output of a large pre-trained model from the free-form-text of LLMs to a type-safe-selection from a pre-defined set of options. Jev also emits a confidence score for each of its decisions and it offers all of this at an order of magnitude cheaper inference costs. Our submissions explore Jev's performance as a both pair-wise-response judge and one-shot-QREL judge. We also explore a prompt enrichment strategey known as 'bag-of-decisions' leveraging a traditional LLM to build an "enriched" request for Jev grading.

## 1. Introduction

LLM-as-a-judge has made relevance assesement cheaper than ever. While LLM-judges have become more common in industry applications and create compelling output judgements, they still exhibit major drawbacks that limit their functional utility including: positional-bias, self-preference, score-centrality and over-confidence. Methods to address these short comings often involed spending more time  "reasoning", running more permutation-prompts per single assesment, running several foundations LLMs in a judge-panel, all of which add to expense and over-confidence as context is expanded. We explore the utility of a 'bag of decisions' context enrichment step that uses a separate LLM to build a evaluation rubric of specific yes/no questions that are then used to subsequent Jev prompt.

Jev is a decision model that takes advantage of similar web-scale pre-training like an LLM, but instead of generating free-form-text output, it instead produces an decision based on a prompt-defined rubric. Unlike LLM-defined-rubrics Jev's training regime makes hallucination impossible and rubric adherence absolute. Jev supports making three classes of decision: Noul, Choice and Score. All of which are accompanied by a confidence score.

We also propose two smaller model strategies for use as first stage filters. The first is Hallucination-Risk (HR), which works independent of LLM or Jev using spaCy noun-phrase detection to build an index of concepts for each topic that allows foreground/background calculations for a given run. The second we include is miniLM's cosine similarity between a topic and response, which gives us a useful marker for "easy" compared to "hard" examples.


## 2. Approach

Our approaches are aimed at measuring Jev's utility as a judge, at both levels of RAG relevance assesment. We judge at both the response and the citation level to compare the quality of the assesment to the cost. Along the way we explore and try to mitigate Jev's suseptibility to the same LLM issues of: positional-bias, over-confidence and score-centrality. Our run naming convention is confusing and we are sorry about that. All of our work is run via OpenRouter and costs are reported in USD ($).

## 2.1 Pairwise Jev

Pairwise Jev is a run response level judge. Pairwise Jev decides on a preferred response from a provided pair of different run responses. A single prompt asks a Noul question that includes the topic and two run responses [appendix1.] We run a single prompt twice to capture both orderings of the run responses [A,B] and [B,A] and then average the resulting scores to get a single preference score normalized for position-bias.

[How combine to leaderboard metrics]

## 2.1.1 Bag of Decisions rubric enrichment

Bag of decisions is a separate LLM prompt that attempts to enrich the provided topic information  beyond just the prompt-statement and information-need provided. Instead of the singular is this response better style question, BOD resquests a list of 5 - 15 rubric questions an expert would ask to check distinct parts of the repsonse for validity. This list is filtered for near-duplicates and then packaged as individual NOUL questions in a new enriched-prompt for the Pairwise Jev, with the final prompt answer scores averaged together before combining both order-permutation versions.

### 2.2 Judge Jev

Judge-Jev is a QREL level judge, It takes single cited document and the associated topic as input and outputs a relevance score ranging from 0 (not relevant) - 3 (most relevant), following TREC multi-label convention. [cite1] A single JudgeJev prompt asks 5 questions about the document’s relevance, 1 Noul (binary relevance probability scaled by a factor of 3) 2-Choice (fwd/rev single score selection e.g. [0,1,2,3] and [3,2,1,3]), 2-Score (fwd/rev continuous float score estimation e.g. [0-3] and [3-0]). All five of these scores are averaged together to a final QREL estimate for each citation.

These QRELS were summarized in the style of trec_eval [cite] to construct various leaderboard metrics.QRELs were not submitted directly to the track, due to evalution limitations. They are available upon requests of the authors and open to publishing with the proceedings. 


## 3. Submissions

[summary table of runs]

## 3.1 Pairwise Jev datum

## 3.2 Judge Jev datum

Cost: $14.40 USD

There is strong answer correlation across differen question framing within the same prompt.  There are distinc confidence distributions associtate with SCORE and CHOICE answers. 

## 4. Conclusions

After conference