"""Pure functions of the BOD x Hallucination Risk judge (no data, no network, no spaCy)."""
import pytest

from judges.bonsai_judge.bod_risk import risk_from_concepts, tally_risk

F = frozenset


def test_risk_counts_only_unsupported_concepts_no_peer_states():
    answers = {("a", "t"): F({"x", "y", "z", "w"}), ("b", "t"): F({"z"})}
    cited = {("a", "t"): F({"d1"}), ("b", "t"): F({"d1"})}
    docs = {"d1": F({"x"})}
    r = risk_from_concepts(answers, cited, docs)
    # a: ungrounded {y, z, w}; z is also stated by b, so private = {y, w}
    assert r[("a", "t")]["risk"] == pytest.approx(2 / 4)
    assert r[("a", "t")]["ungrounded"] == pytest.approx(3 / 4)
    assert r[("b", "t")]["risk"] == 0.0          # z is shared, so not private
    assert r[("b", "t")]["ungrounded"] == 1.0


def test_df_is_per_topic():
    answers = {("a", "t1"): F({"y"}), ("b", "t2"): F({"y"})}
    cited = {k: F({"d"}) for k in answers}
    r = risk_from_concepts(answers, cited, {"d": F({"x"})})
    assert r[("a", "t1")]["risk"] == 1.0 and r[("b", "t2")]["risk"] == 1.0


def test_undefined_risk_uses_no_support_value():
    answers = {("a", "t"): F({"x"}), ("b", "t"): F()}
    cited = {("a", "t"): F(), ("b", "t"): F({"d"})}
    r = risk_from_concepts(answers, cited, {"d": F({"x"})}, no_support_risk=0.7)
    assert r[("a", "t")]["risk"] == 0.7 and not r[("a", "t")]["defined"]
    assert r[("b", "t")]["risk"] == 0.7


def _rows(run, topic, grade):
    p = {str(g): float(g == grade) for g in range(4)}
    return [{"run_id": run, "topic_id": topic, "qid": "q01", "probs": p, "expected": grade}]


def test_tally_multiplies_and_scores_empties_zero():
    rows = _rows("a", "t1", 3) + _rows("a", "t2", 3) + _rows("b", "t1", 3)
    risk = {("a", "t1"): {"risk": 0.5, "ungrounded": 0.6},
            ("a", "t2"): {"risk": 0.0, "ungrounded": 0.1}}   # b/t1 missing -> no_support_risk
    lb = tally_risk(rows, risk, ["t1", "t2"], empties=[("t2", "b")])
    got = {(e.run_id, e.topic_id): e.values for e in lb.entries}
    assert got[("a", "t1")]["BOD_GROUNDED"] == pytest.approx(0.5)
    assert got[("a", "t2")]["BOD_GROUNDED"] == pytest.approx(1.0)
    assert got[("b", "t1")]["BOD_GROUNDED"] == 0.0
    assert got[("b", "t2")]["BOD_GROUNDED"] == 0.0
    assert got[("a", "all")]["BOD_GROUNDED"] == pytest.approx(0.75)
