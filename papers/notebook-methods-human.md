# OMC at TREC 2026 AutoJudge: Notebook Methods

**Authors:** Max Irwin <max@onemorecloud.com>, Nathan Day <nday@eezy.com>, Daniel Worley <worleydl@gmail.com>

## Abstract

Decision models offer a novel (and cheaper) alternative to the established LLM-as-a-judge paradigm. Jev, the first decision model publicly released in September 2026, changes the output of a large pre-trained model from the free-form-text of LLMs to a type-safe-selection from a pre-defined set of options. Jev also emits a confidence score for each of its decisions and it offers all of this at an order of magnitude cheaper inference costs. Our submissions explore Jev's performance as a both pair-wise-response judge and one-shot-QREL judge. We also explore a prompt enrichment strategey known as 'bag-of-decisions' leveraging a traditional LLM to build an "enriched" request for Jev grading.

## 1. Introduction

LLM-as-a-judge [@zheng2023judging] has made relevance assesement cheaper than ever. While LLM-judges have become more common in industry applications and create compelling output judgements, they still exhibit major drawbacks that limit their functional utility including: positional-bias [@wang2024large], self-preference [@panickssery2024llm], score-centrality and over-confidence. Methods to address these short comings often involed spending more time "reasoning", running more permutation-prompts per single assesment, running several foundations LLMs in a judge-panel, all of which add to expense and over-confidence as context is expanded. We explore the utility of a 'bag of decisions' context enrichment step that uses a separate LLM to build a evaluation rubric of specific yes/no questions that are then used to subsequent Jev prompt.

Jev is a decision model that takes advantage of similar web-scale pre-training like an LLM, but instead of generating free-form-text output, it instead produces an decision based on a prompt-defined rubric. Unlike LLM-defined-rubrics Jev's training regime makes hallucination impossible and rubric adherence absolute. Jev supports making three classes of decision: Noul, Choice and Score. All of which are accompanied by a confidence score.

We also propose two smaller model strategies to be used alongside larger models. The first is Hallucination-Risk (HR), which works independent of LLM or Jev using spaCy [@honnibal2017spacy] noun-phrase detection to build an index of concepts for each topic that allows foreground/background calculations for a given run. The second we include is miniLM's cosine similarity [@wang2020minilm] between a topic and response, which gives us a useful marker for "easy" compared to "hard" examples.

## 2. Approach

Our approaches are aimed at measuring Jev's utility as a judge, at both levels of RAG relevance assesment. We judge at both the response and the citation level to compare the quality of the assesment to the cost. Along the way we explore and try to mitigate Jev's suseptibility to the same LLM issues of: positional-bias, over-confidence and score-centrality. Our run naming convention is confusing and we are sorry about that. All of our work is run via OpenRouter and costs are reported in USD ($).

## 2.1 Pairwise Jev

Pairwise Jev is a run response level judge. Pairwise Jev decides on a preferred response from a provided pair of different run responses. A single prompt asks a NOUL question that includes the topic and two run responses [appendix1] We run a single prompt twice to capture both orderings of the run responses [A,B] and [B,A] and then average the resulting scores to get a single preference score normalized for position-bias. Bradley-Terry [@bradley1952rank] combines all pair wise comparisons for a given run into a global win/loss score across all topics, where more relevant runs score higher.

### 2.2 Bag of Decisions rubric enrichment

Bag of decisions is a separate LLM prompt that attempts to enrich the provided topic information beyond just the prompt-statement and information-need provided. Instead of the singular is this response better style question, BOD resquests a list of 5 - 15 rubric questions an expert would ask to check distinct parts of the repsonse for validity. This list is filtered for near-duplicates and then packaged as individual NOUL questions in a new enriched-prompt for the Pairwise Jev, with the final prompt answer scores averaged together before combining both order-permutation versions.

### 2.2.1 Hallucination Risk

Hallucination risk measures a lexical deviation from concepts in the source cited documents to the answer summary text. Using a spaCy large model, we gathered all normalized noun phrases from the cited documents, and gathered all normalized noun phrases from the answers. If an answer used a noun phrase not present in the cited source, it counts as a potential hallucination. Hallucination risk was not submitted as its own judge, but instead was used as a discount for BOD. This discount didn't result in significant change to the leaderboard, but did flip the top two results.

## 2.3 Judge Jev

Judge-Jev is a QREL level judge, It takes single cited document and the associated topic as input and outputs a relevance score ranging from 0 (not relevant) - 3 (most relevant), following TREC multi-label convention. A single JudgeJev prompt asks 5 questions about the document’s relevance, 1 Noul (binary relevance probability scaled by a factor of 3) 2-Choice (fwd/rev single score selection e.g. [0,1,2,3] and [3,2,1,3]), 2-Score (fwd/rev continuous float score estimation e.g. [0-3] and [3-0]). All five of these scores are averaged together to a final QREL estimate for each citation.

These QRELS were summarized in the style of trec_eval to construct various leaderboard metrics.QRELs were not submitted directly to the track, due to evalution limitations. They are available upon requests of the authors and open to publishing with the proceedings.

## 3. Submissions

[summary table of runs and costs]

| Item           | run_ids                                                | Cost (USD) |
| -------------- | ------------------------------------------------------ | ---------- |
| jev pairwise   | [omc26_ragtime26_jevpairwise, omc26_rag26_jevpairwise] | $93.84     |
| jev + BOD      | [omc26_ragtime26_jevbod, omc26_rag26_jevbod]           | $1.57      |
| jev + BOD + HR | [omc26_ragtime26_jevbodrisk, omc26_rag26_jevbodrisk]   | $0.00      |
| JudgeJev       | [omc26_ragtime26_judge_jev, omc26_rag26_judge_jev]     | $14.40     |

## 3.1 Pairwise Jev datum

### 3.1.1 Order bias

![Jev Choice questions exhibit clear order bias in the Pairwise Jev Judge. Using a mirrored version of the choies with order inverted removes most of that bias.](figures_ggplot/f03_identical_summaries.pdf){#fig:order-bias width=55%}

### 3.1.2 Pairwise Jev vs BOD Jev and BOD+HR Jev

![Comparison of leaderboards created with different Jev Judges. Hallucination Risk (Grounded) makes very little change compared to Bag of Decisions (BOD)](figures_ggplot/f11_full_leaderboards.pdf){#fig:leaderboards width=100%}

## 3.2 Judge Jev datum

There is strong answer correlation across differen question framing within the same prompt. There are distinc confidence distributions associtate with SCORE and CHOICE answers.

## 4. Conclusions

After conference

## References

::: {#refs}
:::

## 5. Appendix

### 5.1 OpenRouter Prompts

#### 5.1.1 Pairwise Jev

```
{
  "model": "typesafe/jev-1.13",
  "state": {
    "query": "fall leaf color change chlorophyll",
    "problem_statement": "My 5-year-old keeps noticing that leaves turn red, orange, and yellow in the fall. They want to know why trees change color, whether the leaves are dying, and if the tree will be okay. They also asked why some trees stay green all year and whether cold weather causes the color change.",
    "background": "This is for a child who loves collecting leaves on walks and sorting them by color and shape. They like pressing leaves into notebooks, doing craft projects with them, and comparing different tree leaves side-by-side. They are especially fascinated by bright colors and want explanations that connect to what they can directly see outdoors.",
    "summary_a": "Leaves change color in fall because of a green pigment called chlorophyll. When days get shorter and cooler, trees stop making chlorophyll. Without the green, other colors like yellow and orange can show through. Red colors come from sugars trapped in the leaves. The tree will be fine - it drops leaves to save water for winter.",
    "summary_b": "Trees have green stuff in their leaves called chlorophyll. In fall, the green goes away and you see other colors. Yellow and orange were hiding under the green all along. Some trees like pine trees stay green because they have a different kind of leaf."
  },
  "questions": {
    "better_summary": {
      "type": "choice",
      "instructions": "You are a subject matter expert. Two RAG summaries (summary_a and summary_b) were generated for the query. Which summary better satisfies the information need of the query, considering relevance, completeness, and how well its claims are grounded?",
      "criteria": {
        "A": "summary_a better satisfies the information need of the query.",
        "B": "summary_b better satisfies the information need of the query."
      }
    }
  }
}
```

#### 5.1.2 Bag of Decisions Jev

#### LLM Bag of Decisions generation

```
{
  "model": "google/gemini-2.5-flash-lite",
  "messages: [
    {
      "role": "user",
      "content": "You are designing a grading rubric for answers to a user's information need.

                  fall leaf color change chlorophyll
                  My 5-year-old keeps noticing that leaves turn red, orange, and yellow in the fall. ...

                  Write the questions a careful expert would use to grade how well an answer meets this information need. Each question checks one distinct part of what the user needs to know or do.

                  Rules:

                    - Write between 5 and 15 questions. Use as many as the need has genuinely distinct parts: a narrow, single-fact need gets few questions; a broad, multi-part need gets more. Do not pad the list to reach a number.
                    - Each question must be gradable on a scale from "not addressed" to "fully answered with specific detail", so phrase it as "Does the answer ...?" about content, e.g. "Does the answer explain why ...?" or "Does the answer give ...?".
                    - Every question must check substantive content: a fact, explanation, step, or piece of advice the user asked for or clearly needs to act on what they asked.
                    - Use any background or situation described only to decide WHAT content the user needs (for example, a safety concern makes safety advice necessary). Do not write questions that merely check whether the answer acknowledges, mentions, or relates to the user's interests, hobbies, or feelings.
                    - No two questions may check the same thing. Do not ask about tone, reading level, style, length, formatting, or citations.
                    - Do not mention any particular source, product, or answer.

                    Return ONLY a JSON array of strings, one per question, with no other text.
    }
  ]
}

You are designing a grading rubric for answers to a user's information need.

fall leaf color change chlorophyll
My 5-year-old keeps noticing that leaves turn red, orange, and yellow in the fall. ...

Write the questions a careful expert would use to grade how well an answer meets this information need. Each question checks one distinct part of what the user needs to know or do.

Rules:

- Write between 5 and 15 questions. Use as many as the need has genuinely distinct parts: a narrow, single-fact need gets few questions; a broad, multi-part need gets more. Do not pad the list to reach a number.
- Each question must be gradable on a scale from "not addressed" to "fully answered with specific detail", so phrase it as "Does the answer ...?" about content, e.g. "Does the answer explain why ...?" or "Does the answer give ...?".
- Every question must check substantive content: a fact, explanation, step, or piece of advice the user asked for or clearly needs to act on what they asked.
- Use any background or situation described only to decide WHAT content the user needs (for example, a safety concern makes safety advice necessary). Do not write questions that merely check whether the answer acknowledges, mentions, or relates to the user's interests, hobbies, or feelings.
- No two questions may check the same thing. Do not ask about tone, reading level, style, length, formatting, or citations.
- Do not mention any particular source, product, or answer.

Return ONLY a JSON array of strings, one per question, with no other text.
```

#### Jev Bag of Questions assembly

```
{
  "model": "typesafe/jev-1.13",
  "state": {
    "query": "fall leaf color change chlorophyll",
    "problem_statement": "My 5-year-old keeps noticing that leaves turn red, orange, and yellow in the fall. ...",
    "background": "This is for a child who loves collecting leaves on walks ...",
    "answer": "Leaves change color in fall because of a green pigment called chlorophyll. When days get shorter and cooler, trees stop making chlorophyll. Without the green, other colors like yellow and orange can show through. Red colors come from sugars trapped in the leaves. The tree will be fine - it drops leaves to save water for winter."
  },
  "questions": {
    "q01": {
      "type": "choice",
      "instructions": "Grade the answer on this question: Does the answer explain why leaves change color in the fall?",
      "criteria": {
        "0": "The answer does not address this question at all.",
        "1": "The answer mentions this topic but gives no real substance or explanation.",
        "2": "The answer partially answers this question; important parts are missing or vague.",
        "3": "The answer fully answers this question with specific, accurate detail."
      }
    },
    "q02": {
      "type": "choice",
      "instructions": "Grade the answer on this question: Does the answer explain whether the leaves are dying when they change color?",
      "criteria": { "0": "...", "1": "...", "2": "...", "3": "..." }
    }, ...
  }
}
```

#### 5.1.3 Judge Jev

```
{
  "model": "typesafe/jev-1.13",
  "state": {
    "query": "fall leaf color change chlorophyll",
    "document": "Chlorophyll Chlorophyll is a green pigment found in plants that absorbs light energy for photosynthesis. It gives leaves their green color during spring and summer."
  },
  "questions": {
    "relevant_noul": {
      "type": "noul",
      "instructions": "The cited document `document` is relevant to the `query`: it contains information that answers or directly addresses what the query asks about.",
      "criteria": {
        "true": "The document contains information that answers the query or directly addresses what it asks about.",
        "false": "The document is only loosely related, on a similar topic, or does not address what the query asks."
      }
    },
    "relevant_choice_fwd": {
      "type": "choice",
      "instructions": "Choose the relevance level that best describes how well the cited document `document` addresses the `query`.",
      "criteria": {"0": "Not relevant...", "1": "Marginally relevant...", "2": "Relevant...", "3": "Highly relevant..."}
    },
    "relevant_choice_rev": { "...": "same criteria, keys inserted 3,2,1,0" },
    "relevant_score_fwd": {
      "type": "score",
      "instructions": "Rate how relevant the cited document `document` is to the `query`, on the scale below.",
      "criteria": ["Not relevant...", "Marginally relevant...", "Relevant...", "Highly relevant..."]
    },
    "relevant_score_rev": { "...": "same array, reversed" }
  }
}
```
