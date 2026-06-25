"""Reference-bank retrieval core (pure numpy — no torch/GPU): CSLS de-hubbing, kNN class-vote suggest, and the
ReferenceBank container. Run: pytest tools/curator/tests/test_reference_bank.py -q
"""
from __future__ import annotations

import numpy as np


def _norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def test_csls_de_hubs_retrieval():
    """The motivating failure: a 'hub' reference is broadly similar to every query, so plain cosine-NN
    collapses onto it. CSLS penalizes the hub (high mean sim to many queries) so each query maps to its
    DISTINCTIVE class instead."""
    from tools.curator import reference_bank as rb
    rng = np.random.default_rng(0)
    qs = np.stack([[1, 0, 0, 0.0]] * 3 + [[0, 1, 0, 0.0]] * 3) + 0.02 * rng.standard_normal((6, 4))
    truth = ["A"] * 3 + ["B"] * 3
    # true-class refs are DOMAIN-SHIFTED (cos ~0.6 to their queries); the hub sits centrally, closer (0.72)
    # to EVERY query -> plain cosine-NN collapses onto it.
    Avec = _norm(np.array([[0.6, 0, 0.8, 0.0]]))[0]
    Bvec = _norm(np.array([[0, 0.6, 0, 0.8]]))[0]
    hub = _norm(np.array([[0.72, 0.72, 0, 0.0]]))[0]
    bank = np.stack([hub, Avec, Bvec]); labels = ["hub", "A", "B"]
    plain = np.array(labels)[(_norm(qs) @ _norm(bank).T).argmax(1)]
    assert (plain == "hub").sum() >= 6                                        # plain NN collapses to the hub
    picked = np.array(labels)[rb.csls_matrix(qs, bank).argmax(1)]
    assert "hub" not in set(picked)                                          # CSLS never picks the hub
    assert (picked == np.array(truth)).mean() >= 0.9                         # and maps each query to its class


def test_suggest_ranks_correct_class():
    from tools.curator import reference_bank as rb
    bank = _norm(np.array([[1, 0, 0.0], [0.95, 0.05, 0], [0, 1, 0], [0, 0.95, 0.05]]))
    labels = ["coin", "coin", "lead", "lead"]
    Q = _norm(np.array([[1, 0.02, 0.0], [0.0, 1, 0.02]]))
    s = rb.suggest(Q, bank, labels, topk=2, knn=3)
    assert s[0][0][0] == "coin" and s[1][0][0] == "lead"                      # top suggestion is the right class
    assert all(isinstance(score, float) for _, score in s[0])


def test_rank_instances_for_class():
    """The inverse of suggest: fix a reference class, rank the curator's instances by resemblance. The two
    'lead'-like queries must outrank the 'coin'-like one for class 'lead', and scores stay aligned to the
    original query rows (so the engine can map order[i] back to its iuid)."""
    from tools.curator import reference_bank as rb
    bank = _norm(np.array([[1, 0, 0.0], [0.95, 0.05, 0], [0, 1, 0], [0, 0.95, 0.05]]))
    labels = ["coin", "coin", "lead", "lead"]
    Q = _norm(np.array([[1, 0.02, 0.0], [0.0, 1, 0.02], [0.05, 0.97, 0.0]]))   # coin-like, lead-like, lead-like
    order, scores = rb.rank_instances_for_class(Q, bank, labels, "lead", knn=3)
    assert scores.shape == (3,)                                                # aligned to original query rows
    assert set(order[:2].tolist()) == {1, 2}                                   # the two lead-like queries rank first
    assert int(order[0]) in (1, 2) and int(order[-1]) == 0                     # the coin-like query ranks last
    o2, _ = rb.rank_instances_for_class(Q, bank, labels, "absent-class", knn=3)
    assert o2.size == 0                                                        # unknown class -> empty


def test_reference_bank_container(tmp_path):
    from tools.curator.reference_bank import ReferenceBank
    b = ReferenceBank(np.eye(4, dtype=np.float32)[:3], ["coin", "coin", "lead"],
                      {"coin": "coin", "lead": "lead"},
                      [{"file_name": "raw/coin/a.jpg", "bbox": [0, 0, 5, 5], "cls": "coin"}])
    assert b.n == 3 and b.classes() == ["coin", "lead"]
    assert len(b.exemplars_for("coin")) == 1 and b.exemplars_for("lead") == []
    b.add(np.eye(4, dtype=np.float32)[3:4], ["pacemaker"])                    # self-improving add
    assert b.n == 4 and "pacemaker" in b.classes()
    b.save(tmp_path / "bank")
    r = ReferenceBank.load(tmp_path / "bank")
    assert r.n == 4 and r.classes() == b.classes() and r.emb.shape == (4, 4)
    assert ReferenceBank.load(tmp_path / "nope") is None
