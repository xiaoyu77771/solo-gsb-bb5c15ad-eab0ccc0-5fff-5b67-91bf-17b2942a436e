"""API-level tests for POST /reconcile/exclusion-recovery."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

T0 = "2026-09-17T00:00:00Z"


def txn(id: str, amount: int, time: str = T0) -> dict:
    return {"id": id, "time": time, "amount": amount}


def excl(left_id: str, right_id: str) -> dict:
    return {"left_id": left_id, "right_id": right_id}


def post_recovery(body: dict):
    return client.post("/reconcile/exclusion-recovery", json=body)


def recovery(left, right, target, excluded=None, locked=None):
    body: dict = {"left": left, "right": right, "target_total": target}
    if excluded is not None:
        body["excluded_pairs"] = excluded
    if locked is not None:
        body["locked_groups"] = locked
    resp = post_recovery(body)
    assert resp.status_code == 200, resp.text
    return resp.json()


# --------------------------------------------------------------------- #
# target already met
# --------------------------------------------------------------------- #
def test_target_already_met_suggests_no_removals():
    # the exclusion only reroutes the equally optimal matching
    body = recovery(
        [txn("A", 100), txn("B", 100)],
        [txn("X", 100), txn("Y", 100)],
        200,
        [excl("A", "X")],
    )
    assert body["reachable"] is True
    assert body["removed_pairs"] == []
    assert body["original_total"] == 200
    assert "max_total" not in body
    # result equals the ordinary reconciliation
    plain = client.post(
        "/reconcile",
        json={
            "left": [txn("A", 100), txn("B", 100)],
            "right": [txn("X", 100), txn("Y", 100)],
            "excluded_pairs": [excl("A", "X")],
        },
    ).json()
    assert body["result"] == plain


def test_zero_target_always_reachable_without_removals():
    body = recovery(
        [txn("A", 100)],
        [txn("X", 100)],
        0,
        [excl("A", "X")],
    )
    assert body["reachable"] is True
    assert body["removed_pairs"] == []
    assert body["original_total"] == 0
    assert body["result"]["groups"] == []
    assert "max_total" not in body


def test_no_excluded_pairs_above_total_is_unreachable():
    # with nothing to remove, a positive target above the structural optimum
    # must be reported unreachable immediately
    body = recovery([txn("A", 100)], [txn("X", 101)], 100)
    assert body == {
        "reachable": False,
        "removed_pairs": [],
        "result": {
            "groups": [],
            "total_amount": 0,
            "unmatched_left_ids": ["A"],
            "unmatched_right_ids": ["X"],
        },
        "original_total": 0,
        "max_total": 0,
    }


# --------------------------------------------------------------------- #
# fewest removals and lexicographic tie-break
# --------------------------------------------------------------------- #
def test_single_pair_removal_restores_larger_group():
    # natural optimum (A+B,X)=150; excluding (A,X) drops it to (A,Y)=100
    body = recovery(
        [txn("A", 100), txn("B", 50)],
        [txn("X", 150), txn("Y", 100)],
        150,
        [excl("A", "X")],
    )
    assert body["reachable"] is True
    assert body["removed_pairs"] == [excl("A", "X")]
    assert body["original_total"] == 100
    assert body["result"] == {
        "groups": [{"left_ids": ["A", "B"], "right_ids": ["X"], "amount": 150}],
        "total_amount": 150,
        "unmatched_left_ids": [],
        "unmatched_right_ids": ["Y"],
    }
    assert "max_total" not in body


def test_two_independent_pairs_both_required():
    # unique amounts: L1-R1=300 and L2-R2=400 are the only legal groups;
    # excluding both drops total to 0 and both pairs must be removed
    body = recovery(
        [txn("L1", 300), txn("L2", 400)],
        [txn("R1", 300), txn("R2", 400)],
        700,
        [excl("L2", "R2"), excl("L1", "R1")],
    )
    assert body["reachable"] is True
    # returned sorted lexicographically regardless of input order
    assert body["removed_pairs"] == [excl("L1", "R1"), excl("L2", "R2")]
    assert body["original_total"] == 0
    assert body["result"]["total_amount"] == 700


def test_intermediate_target_needs_one_pair_and_lex_wins():
    # with both disjoint groups excluded, target 300 needs one removal;
    # either pair alone reaches 300, so (L1,R1) wins lexicographically
    body = recovery(
        [txn("L1", 300), txn("L2", 400)],
        [txn("R1", 300), txn("R2", 400)],
        300,
        [excl("L2", "R2"), excl("L1", "R1")],
    )
    assert body["reachable"] is True
    assert body["removed_pairs"] == [excl("L1", "R1")]
    assert body["result"]["total_amount"] == 300
    assert body["result"]["unmatched_left_ids"] == ["L2"]


def test_tie_break_between_different_single_pair_removals():
    # excluding both of A's pairs wipes A out; restoring either (A,X) -> 150
    # group or (A,Y) -> 100. Target 150 uniquely picks (A,X) lex-first, and
    # a target of 100 additionally proves the lex tie-break itself.
    left, right = [txn("A", 100), txn("B", 50)], [txn("X", 150), txn("Y", 100)]
    pairs = [excl("A", "Y"), excl("A", "X")]
    body = recovery(left, right, 150, pairs)
    assert body["removed_pairs"] == [excl("A", "X")]
    # (A,X) removed gives 150 >= 100 too; the lex-smallest pair still wins
    body_100 = recovery(left, right, 100, pairs)
    assert body_100["removed_pairs"] == [excl("A", "X")]
    assert body_100["original_total"] == 0


def test_recovered_result_uses_standard_lex_tie_break():
    # after the suggested removals the response groups are the canonical
    # optimum (sorted by group encoding) for the relaxed exclusion set
    body = recovery(
        [txn("L1", 300), txn("L2", 400)],
        [txn("R1", 300), txn("R2", 400)],
        700,
        [excl("L1", "R1"), excl("L2", "R2")],
    )
    assert [(g["left_ids"], g["right_ids"]) for g in body["result"]["groups"]] == [
        (["L1"], ["R1"]),
        (["L2"], ["R2"]),
    ]
    assert body["result"]["unmatched_left_ids"] == []
    assert body["result"]["unmatched_right_ids"] == []


# --------------------------------------------------------------------- #
# unreachable
# --------------------------------------------------------------------- #
def test_unreachable_returns_empty_pairs_and_max_total():
    # (A,R1)=100 is forbidden; removing it is the structural maximum, below 101
    body = recovery(
        [txn("A", 100)],
        [txn("R1", 100), txn("R2", 101)],
        101,
        [excl("A", "R1")],
    )
    assert body["reachable"] is False
    assert body["removed_pairs"] == []
    assert body["original_total"] == 0
    assert body["max_total"] == 100
    # result shows the best achievable plan (all exclusions removed)
    assert body["result"]["total_amount"] == 100
    assert body["result"]["groups"] == [
        {"left_ids": ["A"], "right_ids": ["R1"], "amount": 100}
    ]


def test_unreachable_when_target_exceeds_all_side_sums():
    body = recovery(
        [txn("L1", 300)],
        [txn("R1", 100), txn("R2", 200)],
        301,
        [excl("L1", "R1"), excl("L1", "R2")],
    )
    assert body["reachable"] is False
    assert body["removed_pairs"] == []
    assert body["max_total"] == 300
    assert body["result"]["groups"] == [
        {"left_ids": ["L1"], "right_ids": ["R1", "R2"], "amount": 300}
    ]


# --------------------------------------------------------------------- #
# locked groups stay untouched
# --------------------------------------------------------------------- #
def test_locked_groups_kept_while_free_pool_recovers():
    # locked (A,X)=100; the free 150 group (B+C,Y) needs both exclusions gone
    body = recovery(
        [txn("A", 100), txn("B", 50), txn("C", 100)],
        [txn("X", 100), txn("Y", 150), txn("Z", 100)],
        250,
        [excl("B", "Y"), excl("C", "Y")],
        [{"left_ids": ["A"], "right_ids": ["X"]}],
    )
    assert body["reachable"] is True
    assert body["removed_pairs"] == [excl("B", "Y"), excl("C", "Y")]
    assert body["result"]["total_amount"] == 250
    assert {"left_ids": ["A"], "right_ids": ["X"], "amount": 100} in body["result"]["groups"]
    assert {"left_ids": ["B", "C"], "right_ids": ["Y"], "amount": 150} in body["result"]["groups"]
    assert body["result"]["unmatched_right_ids"] == ["Z"]


def test_locked_group_weight_counts_toward_totals():
    # (A,X)=100 is locked and legal; the exclusion only touches free flows, so
    # the original optimum already includes the locked group's weight
    body = recovery(
        [txn("A", 100), txn("B", 100)],
        [txn("X", 100), txn("Y", 100)],
        100,
        [excl("B", "Y")],
        [{"left_ids": ["A"], "right_ids": ["X"]}],
    )
    assert body["reachable"] is True
    assert body["removed_pairs"] == []
    assert body["original_total"] == 100
    assert body["result"]["groups"] == [
        {"left_ids": ["A"], "right_ids": ["X"], "amount": 100}
    ]
    assert body["result"]["unmatched_left_ids"] == ["B"]
    assert body["result"]["unmatched_right_ids"] == ["Y"]


# --------------------------------------------------------------------- #
# request validation parity with /reconcile
# --------------------------------------------------------------------- #
def test_duplicate_batch_ids_return_400():
    resp = post_recovery(
        {
            "left": [txn("A", 1), txn("A", 2)],
            "right": [],
            "target_total": 0,
        }
    )
    assert resp.status_code == 400
    assert resp.json() == {"detail": "duplicate transaction id in batch"}


def test_invalid_excluded_pair_returns_422_with_index():
    resp = post_recovery(
        {
            "left": [txn("A", 1)],
            "right": [txn("X", 1)],
            "excluded_pairs": [excl("NOPE", "X")],
            "target_total": 0,
        }
    )
    assert resp.status_code == 422
    err = resp.json()["detail"][0]
    assert err["loc"] == ["body", "excluded_pairs", 0]
    assert err["type"] == "excluded_pair.invalid"


def test_locked_group_hitting_excluded_pair_returns_422():
    resp = post_recovery(
        {
            "left": [txn("A", 100)],
            "right": [txn("X", 100)],
            "locked_groups": [{"left_ids": ["A"], "right_ids": ["X"]}],
            "excluded_pairs": [excl("A", "X")],
            "target_total": 100,
        }
    )
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "locked_groups", 0]


def test_more_than_eight_excluded_pairs_returns_422():
    resp = post_recovery(
        {
            "left": [txn("L", 1)],
            "right": [txn(f"R{i}", 1) for i in range(9)],
            "excluded_pairs": [excl("L", f"R{i}") for i in range(9)],
            "target_total": 0,
        }
    )
    assert resp.status_code == 422
    err = resp.json()["detail"][0]
    assert err["loc"] == ["body", "excluded_pairs"]


def test_exactly_eight_excluded_pairs_accepted():
    resp = post_recovery(
        {
            "left": [txn("L", 1)],
            "right": [txn(f"R{i}", 1) for i in range(8)],
            "excluded_pairs": [excl("L", f"R{i}") for i in range(8)],
            "target_total": 0,
        }
    )
    assert resp.status_code == 200, resp.text


def test_target_total_validation():
    base = {"left": [], "right": []}
    for bad in (-1, -100, 1.5, True, False, "100", None):
        resp = post_recovery({**base, "target_total": bad})
        assert resp.status_code == 422, bad
        assert resp.json()["detail"][0]["loc"] == ["body", "target_total"], bad
    # missing field
    resp = post_recovery(dict(base))
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "target_total"]
    # non-negative integers are fine
    for good in (0, 1, 10**12):
        resp = post_recovery({**base, "target_total": good})
        assert resp.status_code == 200, good


def test_batch_field_validation_matches_reconcile():
    # zero amount, side length and timestamp rules are untouched
    for bad_body in (
        {"left": [txn("A", 0)], "right": [], "target_total": 0},
        {
            "left": [txn(f"L{i}", 1) for i in range(13)],
            "right": [],
            "target_total": 0,
        },
        {"left": [txn("A", 1, "nonsense")], "right": [], "target_total": 0},
    ):
        resp = post_recovery(bad_body)
        assert resp.status_code == 422, bad_body
