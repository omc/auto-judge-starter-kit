"""Pure functions of the bag-of-decisions judge (no data, no network)."""
import pytest

from judges.bonsai_judge.bod import (
    PADDING_ID, dedupe_questions, grade_probs, jev_questions, parse_questions, tally,
)


def test_parse_questions_tolerates_fences():
    assert parse_questions('```json\n["Does the answer x?", "Does it y?"]\n```') == \
        ["Does the answer x?", "Does it y?"]
    with pytest.raises(ValueError):
        parse_questions("no array here")


def test_dedupe_keeps_order_and_drops_near_duplicates():
    qs = ["Does the answer explain X?", "does the answer explain x?",
          "Does the answer explain why X happens?", "  ", "Does the answer give Y?"]
    assert dedupe_questions(qs, 0.8) == ["Does the answer explain X?",
                                         "Does the answer explain why X happens?",
                                         "Does the answer give Y?"]


def test_grade_probs_renormalises_and_rejects_bad():
    p = grade_probs({"type": "choice", "probabilities": {"0": 0, "1": 0, "2": 1, "3": 1}})
    assert p == {0: 0.0, 1: 0.0, 2: 0.5, 3: 0.5}
    assert grade_probs({"type": "choice", "probabilities": {"A": 1}}) is None
    assert grade_probs({"type": "noul", "noul": 0.3}) is None


def test_jev_questions_ids_and_padding():
    qs = jev_questions(["a?", "b?"])
    assert list(qs) == ["q01", "q02", PADDING_ID]
    assert set(qs["q01"]["criteria"]) == {"0", "1", "2", "3"}
    assert PADDING_ID not in jev_questions(["a?"], padding=False)


def _row(run, topic, qid, probs):
    p = dict(zip("0123", probs))
    return {"run_id": run, "topic_id": topic, "qid": qid, "probs": p,
            "expected": sum(int(g) * v for g, v in p.items())}


def test_tally_measures_and_empty_rows():
    rows = [_row("r1", "t1", "q01", (0, 0, 0, 1)), _row("r1", "t1", "q02", (0, 0, 1, 0)),
            {"run_id": "r1", "topic_id": "t1", "qid": PADDING_ID, "p_yes": 0.2}]
    lb = tally(rows, ["t1"], empties=[("t1", "r2")])
    vals = {(e.run_id, e.topic_id): e.values for e in lb.entries}
    assert vals[("r1", "t1")]["BOD_MEAN"] == pytest.approx(5 / 6)
    assert vals[("r1", "t1")]["BOD_COVERAGE"] == 1.0
    assert vals[("r1", "t1")]["BOD_FULL"] == 0.5
    assert vals[("r1", "t1")]["BOD_PADDING"] == 0.2
    assert vals[("r2", "t1")]["BOD_MEAN"] == 0.0


def test_tally_skips_partially_invalid_answers():
    rows = [_row("r1", "t1", "q01", (0, 0, 0, 1)),
            {"run_id": "r1", "topic_id": "t1", "qid": "q02", "probs": None, "expected": None}]
    lb = tally(rows, ["t1"], on_missing="default")
    vals = {(e.run_id, e.topic_id): e.values for e in lb.entries}
    assert ("r1", "t1") not in vals or vals[("r1", "t1")]["BOD_MEAN"] == 0.0
