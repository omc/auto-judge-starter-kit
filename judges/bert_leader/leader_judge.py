from autojudge_base import Leaderboard, LeaderboardBuilder, LeaderboardSpec, MeasureSpec

import json

BERT_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("SIM_SCORE", description="Average Dot Product similarity between topic and individual sentences"),
))

class LeaderJudge:
    def judge(self, rag_responses, rag_topics, llm_config, **kwargs) -> Leaderboard:
        builder = LeaderboardBuilder(BERT_SPEC)

        # Pull in precomputed scores via path in workflow config
        # See minilm-scripts folder for info on how to generate score banks
        score_path = kwargs['score_path']
        with open(score_path, 'r', encoding='utf-8') as score_cache: 
            score_lookup = json.load(score_cache)

        for response in rag_responses:
            score = score_lookup[f'{response.metadata.run_id}-{response.metadata.topic_id}'] 
            builder.add(
                run_id=response.metadata.run_id,
                topic_id=response.metadata.topic_id,
                values={"SIM_SCORE": score},
            )
        topic_ids = [t.request_id for t in rag_topics]
        return builder.build(expected_topic_ids=topic_ids, on_missing="fix_aggregate")
