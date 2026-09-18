"""Tests for the exclusion-rule recovery suggestion.

Covers the solver-level ``recover_exclusions`` helper with a brute-force
differential check over small batches (fewest removals, then lexicographically
smallest sorted pair list), plus the ``POST /reconcile/exclusion-recovery``
endpoint behaviour (bodies, validation parity with ``/reconcile``).
"""

from __future__ import annotations

import random
from itertools import combinations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.solver import (
    ExcludedPair,
    LockedGroup,
    Reconciler,
    Txn,
    recover_exclusions,
    resolve_excluded_pairs,
    resolve_locked_groups,
)

client = TestClient(app)

T0 = "2026-09-17T00:00:00Z"
BASE_TS = 1_758_000_000.0
AMOUNTS = [-300, -200, -100, -50, 50, 100, 200, 300]


def txn(id: str, amount: int, time: str = T0) -> dict:
    return {"id": id, "time": time, "amount": amount}


def excluded(left_id: str, right_id: str) -> dict:
    return {"left_id": left_id, "right_id": right_id}


def post_recovery(body: dict):
    return client.post("/reconcile/exclusion-recovery", json=body)


def make_txn(id: str, amount: int, ts: float = BASE_TS) -> Txn:
    return Txn(id=id, ts=ts, amount=amount)


def build_rec(left, right, pair_specs, locked_specs=()):
    pairs = [ExcludedPair(left_id=l, right_id=r) for l, r in pair_specs]
    resolved = resolve_excluded_pairs({t.id for t in left}, {t.id for t in right}, pairs)
    rec = Reconciler(left, right, excluded_pairs=resolved)
    locked = [LockedGroup(left_ids=tuple(sorted(l)), right_ids=tuple(sorted(r)))
              for l, r in locked_specs]
    rec.locked = resolve_locked_groups(rec, locked)
    return rec


def brute_force_recovery(rec: Reconciler, target: int):
    """Return (reachable, removed_pairs, total) by solving every subset of the
    excluded pairs independently, mirroring the minimality / lexicographic
    specification."""
    pairs = sorted(rec.excluded_pairs)
    original_groups = rec.solve()
    original_total = sum(c.weight for c in original_groups)

    # full candidate pool without exclusions, shared by the variants
    full = Reconciler(rec.left, rec.right, locked=rec.locked)
    totals = {}
    for k in range(len(pairs) + 1):
        for chosen in combinations(pairs, k):
            active = set(pairs) - set(chosen)
            kept = [
                cand
                for cand in full.candidates
                if not any(
                    lid in cand.left_ids and rid in cand.right_ids for lid, rid in active
                )
            ]
            sub = Reconciler(
                full.left, full.right, locked=full.locked,
                excluded_pairs=active, candidates=kept,
            )
            totals[chosen] = sum(c.weight for c in sub.solve())

    if original_total >= target:
        return True, (), original_total
    feasible = [chosen for chosen, total in totals.items() if total >= target]
    if not feasible:
        return False, (), totals[tuple(pairs)]
    best = min(feasible, key=lambda ch: (len(ch), list(ch)))
    return True, best, totals[best]


# --------------------------------------------------------------------- #
# solver-level: fixed scenarios
# --------------------------------------------------------------------- #
def test_no_removal_needed_returns_empty():
    left = [make_txn("A", 100), make_txn("B", 50)]
    right = [make_txn("X", 150), make_txn("Y", 100)]
    rec = build_rec(left, right, [("A", "X")])
    groups = rec.solve()
    outcome = recover_exclusions(rec, 100, groups)
    assert outcome.reachable
    assert outcome.removed_pairs == ()
    assert outcome.total == 100


def test_target_zero_always_reachable_without_removal():
    left = [make_txn("A", 100)]
    right = [make_txn("X", 100)]
    rec = build_rec(left, right, [("A", "X")])
    groups = rec.solve()
    assert sum(c.weight for c in groups) == 0
    outcome = recover_exclusions(rec, 0, groups)
    assert outcome.reachable
    assert outcome.removed_pairs == ()
    assert outcome.total == 0


def test_single_pair_removal_reaches_target():
    left = [make_txn("A", 100), make_txn("B", 50)]
    right = [make_txn("X", 150), make_txn("Y", 100)]
    rec = build_rec(left, right, [("A", "X")])
    groups = rec.solve()
    assert sum(c.weight for c in groups) == 100
    outcome = recover_exclusions(rec, 150, groups)
    assert outcome.reachable
    assert outcome.removed_pairs == (("A", "X"),)
    assert outcome.total == 150
    assert sorted(c.encoding for c in outcome.groups) == [
        (("A", "B"), ("X",))
    ]


def test_unreachable_returns_empty_and_max_total():
    left = [make_txn("A", 100), make_txn("B", 50)]
    right = [make_txn("X", 150), make_txn("Y", 100)]
    rec = build_rec(left, right, [("A", "X")])
    groups = rec.solve()
    outcome = recover_exclusions(rec, 999, groups)
    assert not outcome.reachable
    assert outcome.removed_pairs == ()
    assert outcome.total == 150
    assert sorted(c.encoding for c in outcome.groups) == [
        (("A", "B"), ("X",))
    ]


def test_tie_break_prefers_lexicographically_smallest_pair():
    # excluding (A,X) and (A,Y) leaves only B matching; removing either pair
    # restores the perfect matching -> the smaller pair (A,X) is suggested
    left = [make_txn("A", 100), make_txn("B", 100)]
    right = [make_txn("X", 100), make_txn("Y", 100)]
    rec = build_rec(left, right, [("A", "X"), ("A", "Y")])
    groups = rec.solve()
    assert sum(c.weight for c in groups) == 100
    outcome = recover_exclusions(rec, 200, groups)
    assert outcome.reachable
    assert outcome.removed_pairs == (("A", "X"),)


def test_input_order_does_not_change_tie_break():
    # the pair list is sorted internally; a reversed request must yield the
    # same unique suggestion
    left = [make_txn("A", 100), make_txn("B", 100)]
    right = [make_txn("X", 100), make_txn("Y", 100)]
    rec_reversed = build_rec(left, right, [("A", "Y"), ("A", "X")])
    outcome = recover_exclusions(rec_reversed, 200, rec_reversed.solve())
    assert outcome.removed_pairs == (("A", "X"),)


def test_fewer_removals_beat_more_effective_ones():
    # 2x2 with (A,X),(A,Y),(B,X) excluded: only (B,Y) survives so the original
    # total is 100.  Removing (A,X) alone restores the perfect matching
    # (A,X)+(B,Y); the other single removals stay at 100, so the minimum answer
    # is exactly the one pair -- never a two-pair superset of it.
    left = [make_txn("A", 100), make_txn("B", 100)]
    right = [make_txn("X", 100), make_txn("Y", 100)]
    rec = build_rec(left, right, [("A", "X"), ("A", "Y"), ("B", "X")])
    groups = rec.solve()
    assert sum(c.weight for c in groups) == 100
    outcome = recover_exclusions(rec, 200, groups)
    assert outcome.reachable
    assert len(outcome.removed_pairs) == 1
    assert outcome.removed_pairs == (("A", "X"),)
    assert outcome.total == 200


def test_locked_groups_are_preserved():
    left = [make_txn("A", 100), make_txn("B", 100)]
    right = [make_txn("X", 100), make_txn("Y", 100)]
    rec = build_rec(left, right, [("B", "X"), ("B", "Y")], [(("A",), ("Y",))])
    groups = rec.solve()
    outcome = recover_exclusions(rec, 200, groups)
    assert outcome.reachable
    encs = [c.encoding for c in outcome.groups]
    assert (("A",), ("Y",)) in encs
    assert (("B",), ("X",)) in encs


def test_recovery_does_not_mutate_original_reconciler():
    left = [make_txn("A", 100), make_txn("B", 50)]
    right = [make_txn("X", 150), make_txn("Y", 100)]
    rec = build_rec(left, right, [("A", "X")])
    groups = rec.solve()
    before_candidates = len(rec.candidates)
    before_groups = sorted(c.encoding for c in groups)
    recover_exclusions(rec, 150, groups)
    assert len(rec.candidates) == before_candidates
    assert sorted(c.encoding for c in rec.solve()) == before_groups
    assert rec.excluded_pairs == frozenset({("A", "X")})


# --------------------------------------------------------------------- #
# solver-level: differential checks against brute force
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(120))
def test_recovery_against_brute_force(seed: int):
    rng = random.Random(40_000 + seed)
    n_left = rng.randint(1, 5)
    n_right = rng.randint(1, 5)
    left = [
        Txn(f"L{i}", BASE_TS + rng.randint(0, 4 * 86400), rng.choice(AMOUNTS))
        for i in range(n_left)
    ]
    right = [
        Txn(f"R{i}", BASE_TS + rng.randint(0, 4 * 86400), rng.choice(AMOUNTS))
        for i in range(n_right)
    ]
    base = Reconciler(left, right)
    if not base.candidates:
        pytest.skip("batch has no legal groups")

    # seed exclusions from actual candidate pairs, plus arbitrary pairs,
    # capped at 5 so the brute force stays small
    pair_specs = set()
    for _ in range(rng.randint(1, 5)):
        cand = rng.choice(base.candidates)
        pair_specs.add((rng.choice(cand.left_ids), rng.choice(cand.right_ids)))
    for _ in range(rng.randint(0, 2)):
        pair_specs.add((rng.choice(left).id, rng.choice(right).id))
    pair_specs = set(sorted(pair_specs)[:5])

    rec = build_rec(left, right, sorted(pair_specs))
    groups = rec.solve()
    original_total = sum(c.weight for c in groups)

    # probe several interesting targets
    for target in (0, original_total, original_total + 1, original_total + 10_000):
        outcome = recover_exclusions(rec, target, groups)
        bf_reachable, bf_removed, bf_total = brute_force_recovery(rec, target)
        assert outcome.reachable == bf_reachable
        assert outcome.removed_pairs == bf_removed
        assert outcome.total == bf_total


# --------------------------------------------------------------------- #
# API-level behaviour
# --------------------------------------------------------------------- #
def test_endpoint_recovery_suggestion_body():
    body = {
        "left": [txn("A", 100), txn("B", 50)],
        "right": [txn("X", 150), txn("Y", 100)],
        "excluded_pairs": [excluded("A", "X")],
        "target_total": 150,
    }
    resp = post_recovery(body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["reachable"] is True
    assert data["removed_pairs"] == [{"left_id": "A", "right_id": "X"}]
    assert data["original_total"] == 100
    assert data["max_total"] is None
    assert data["result"]["groups"] == [
        {"left_ids": ["A", "B"], "right_ids": ["X"], "amount": 150}
    ]
    assert data["result"]["total_amount"] == 150
    assert data["result"]["unmatched_right_ids"] == ["Y"]


def test_endpoint_already_reachable_without_removal():
    body = {
        "left": [txn("A", 100)],
        "right": [txn("X", 100)],
        "excluded_pairs": [excluded("A", "X")],
        "target_total": 0,
    }
    resp = post_recovery(body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["reachable"] is True
    assert data["removed_pairs"] == []
    assert data["original_total"] == 0
    assert data["result"]["total_amount"] == 0
    assert data["result"]["unmatched_left_ids"] == ["A"]
    assert data["max_total"] is None


def test_endpoint_unreachable_reports_max_total():
    body = {
        "left": [txn("A", 100), txn("B", 50)],
        "right": [txn("X", 150), txn("Y", 100)],
        "excluded_pairs": [excluded("A", "X")],
        "target_total": 999,
    }
    resp = post_recovery(body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["reachable"] is False
    assert data["removed_pairs"] == []
    assert data["original_total"] == 100
    assert data["max_total"] == 150
    # result is the all-pairs-removed optimum
    assert data["result"]["total_amount"] == 150


def test_endpoint_no_excluded_pairs_unreachable():
    body = {
        "left": [txn("A", 100)],
        "right": [txn("X", 101)],
        "target_total": 50,
    }
    resp = post_recovery(body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["reachable"] is False
    assert data["removed_pairs"] == []
    assert data["original_total"] == 0
    assert data["max_total"] == 0


def test_endpoint_tie_break_lexicographic():
    body = {
        "left": [txn("A", 100), txn("B", 100)],
        "right": [txn("X", 100), txn("Y", 100)],
        "excluded_pairs": [excluded("A", "Y"), excluded("A", "X")],
        "target_total": 200,
    }
    resp = post_recovery(body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["removed_pairs"] == [{"left_id": "A", "right_id": "X"}]


def test_endpoint_removed_pairs_sorted_independent_of_request_order():
    # both disjoint pairs must be removed to reach 300; the response list is
    # sorted by (left_id, right_id) even though the request reverses it
    body = {
        "left": [txn("L2", 200), txn("L1", 100)],
        "right": [txn("R2", 200), txn("R1", 100)],
        "excluded_pairs": [excluded("L2", "R2"), excluded("L1", "R1")],
        "target_total": 300,
    }
    resp = post_recovery(body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["reachable"] is True
    assert data["removed_pairs"] == [
        {"left_id": "L1", "right_id": "R1"},
        {"left_id": "L2", "right_id": "R2"},
    ]
    assert data["result"]["total_amount"] == 300


def test_endpoint_preserves_locked_groups():
    body = {
        "left": [txn("A", 100), txn("B", 100)],
        "right": [txn("X", 100), txn("Y", 100)],
        "locked_groups": [{"left_ids": ["A"], "right_ids": ["Y"]}],
        "excluded_pairs": [excluded("B", "X"), excluded("B", "Y")],
        "target_total": 200,
    }
    resp = post_recovery(body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["reachable"] is True
    assert data["removed_pairs"] == [{"left_id": "B", "right_id": "X"}]
    encs = [(g["left_ids"], g["right_ids"]) for g in data["result"]["groups"]]
    assert (["A"], ["Y"]) in encs


def test_target_total_negative_422():
    resp = post_recovery({"left": [], "right": [], "target_total": -1})
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "target_total"]


def test_target_total_missing_422():
    resp = post_recovery({"left": [], "right": []})
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "target_total"]


@pytest.mark.parametrize("bad", [1.5, True, "100"])
def test_target_total_must_be_integer(bad):
    resp = post_recovery({"left": [], "right": [], "target_total": bad})
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "target_total"]


def test_more_than_8_excluded_pairs_422():
    pairs = [excluded("A", f"R{i}") for i in range(9)]
    body = {
        "left": [txn("A", 1)],
        "right": [txn(f"R{i}", 1) for i in range(9)],
        "excluded_pairs": pairs,
        "target_total": 0,
    }
    resp = post_recovery(body)
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "excluded_pairs"]


def test_eight_excluded_pairs_accepted():
    pairs = [excluded("A", f"R{i}") for i in range(8)]
    body = {
        "left": [txn("A", 1)],
        "right": [txn(f"R{i}", 1) for i in range(8)],
        "excluded_pairs": pairs,
        "target_total": 0,
    }
    resp = post_recovery(body)
    assert resp.status_code == 200, resp.text


def test_duplicate_ids_return_400():
    body = {
        "left": [txn("X", 1), txn("X", 2)],
        "right": [],
        "target_total": 0,
    }
    assert post_recovery(body).status_code == 400


def test_excluded_pair_validation_matches_reconcile():
    body = {
        "left": [txn("A", 100)],
        "right": [txn("X", 100)],
        "excluded_pairs": [excluded("NOPE", "X")],
        "target_total": 0,
    }
    resp = post_recovery(body)
    assert resp.status_code == 422
    err = resp.json()["detail"][0]
    assert err["type"] == "excluded_pair.invalid"
    assert err["loc"] == ["body", "excluded_pairs", 0]


def test_locked_group_validation_matches_reconcile():
    body = {
        "left": [txn("A", 100)],
        "right": [txn("X", 100)],
        "locked_groups": [{"left_ids": ["A"], "right_ids": ["NOPE"]}],
        "target_total": 0,
    }
    resp = post_recovery(body)
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "locked_groups", 0]


def test_locked_group_hitting_excluded_pair_returns_422():
    body = {
        "left": [txn("A", 100)],
        "right": [txn("X", 100)],
        "locked_groups": [{"left_ids": ["A"], "right_ids": ["X"]}],
        "excluded_pairs": [excluded("A", "X")],
        "target_total": 0,
    }
    resp = post_recovery(body)
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "locked_groups", 0]


def test_zero_amount_transaction_422():
    body = {
        "left": [txn("A", 0)],
        "right": [],
        "target_total": 0,
    }
    assert post_recovery(body).status_code == 422
