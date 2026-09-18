"""Solver tests for the exclusion-recovery search, incl. brute-force checks."""

from __future__ import annotations

import random
from itertools import combinations

import pytest

from app.solver import (
    ExcludedPair,
    Reconciler,
    Txn,
    find_exclusion_recovery,
    resolve_excluded_pairs,
)

BASE_TS = 1_758_000_000.0
AMOUNTS = [-300, -200, -100, -50, 50, 100, 200, 300]


def brute_force_recovery(
    left: list[Txn],
    right: list[Txn],
    pair_specs: list[tuple[str, str]],
    target: int,
):
    """Reference oracle: try every removal subset in the documented order.

    Returns ``(reachable, sorted removed pairs, total)`` where the total is the
    canonical optimum after removal (all pairs removed when unreachable).
    """
    n = len(pair_specs)
    ordered = sorted(pair_specs)
    best_for: dict[tuple, int] = {}
    for k in range(n + 1):
        for removed_idx in combinations(range(n), k):
            removed = tuple(ordered[p] for p in removed_idx)
            kept = set(pair_specs) - set(removed)
            rec = Reconciler(left, right, excluded_pairs=kept)
            groups = rec.solve()
            total = sum(c.weight for c in groups)
            best_for[removed] = total
            if total >= target:
                return True, list(removed), total
    return False, [], best_for[tuple(ordered)]


def run_recovery(
    left: list[Txn],
    right: list[Txn],
    pair_specs: list[tuple[str, str]],
    target: int,
):
    pairs = [ExcludedPair(left_id=l, right_id=r) for l, r in pair_specs]
    resolved = resolve_excluded_pairs(
        {t.id for t in left}, {t.id for t in right}, pairs
    )
    rec = Reconciler(left, right, excluded_pairs=resolved)
    out = find_exclusion_recovery(rec, target)
    total = sum(c.weight for c in out.groups)
    return out, total


def test_target_zero_never_removes_pairs():
    left = [Txn("A", BASE_TS, 100)]
    right = [Txn("X", BASE_TS, 100)]
    out, total = run_recovery(left, right, [("A", "X")], 0)
    assert (out.reachable, out.removed_pairs, total, out.original_total) == (
        True, (), 0, 0
    )


def test_no_pairs_target_too_high_unreachable():
    left = [Txn("A", BASE_TS, 100)]
    right = [Txn("X", BASE_TS, 101)]
    out, total = run_recovery(left, right, [], 100)
    assert out.reachable is False
    assert out.removed_pairs == ()
    assert out.max_total == total == 0


def test_simple_recovery():
    left = [Txn("A", BASE_TS, 100), Txn("B", BASE_TS, 50)]
    right = [Txn("X", BASE_TS, 150), Txn("Y", BASE_TS, 100)]
    out, total = run_recovery(left, right, [("A", "X")], 150)
    assert out.reachable is True
    assert out.removed_pairs == (("A", "X"),)
    assert total == 150
    assert out.original_total == 100
    # recovered groups are a legal packing of the relaxed view
    assert sum(c.weight for c in out.groups) == 150


def test_unreachable_reports_all_removed_optimum():
    left = [Txn("A", BASE_TS, 100)]
    right = [Txn("X", BASE_TS, 100), Txn("Y", BASE_TS, 101)]
    out, total = run_recovery(left, right, [("A", "X")], 101)
    assert out.reachable is False
    assert out.removed_pairs == ()
    assert out.max_total == total == 100
    assert out.original_total == 0


def test_fewest_count_then_lex_order():
    # three disjoint unique-weight groups; restoring one group only yields its
    # own weight, so each target below selects exactly one required removal
    vals = [100, 200, 400]
    left = [Txn(f"L{i}", BASE_TS, vals[i]) for i in range(3)]
    right = [Txn(f"R{i}", BASE_TS, vals[i]) for i in range(3)]
    pairs = [("L2", "R2"), ("L0", "R0"), ("L1", "R1")]
    # target 300: only the 400 group's single removal suffices
    out, total = run_recovery(left, right, pairs, 300)
    assert out.reachable is True
    assert out.removed_pairs == (("L2", "R2"),)
    assert total == 400
    # target 200: the single (weight 200) removal
    out, total = run_recovery(left, right, pairs, 200)
    assert out.removed_pairs == (("L1", "R1"),)
    assert total == 200
    # total 700 forces all three removals
    out, total = run_recovery(left, right, pairs, 700)
    assert out.removed_pairs == (("L0", "R0"), ("L1", "R1"), ("L2", "R2"))
    assert total == 700


def test_lex_tie_between_equal_single_removals():
    # Two isolated equal-weight groups with amounts that never match crosswise
    # (100 vs 1000), so the 1-1 groups cannot reroute.  Excluding one pair
    # per group leaves total 0; target 100 admits either single removal, and
    # the lex-smallest pair wins regardless of input order.
    left = [Txn("A", BASE_TS, 100), Txn("B", BASE_TS, 1000)]
    right = [Txn("X", BASE_TS, 100), Txn("Y", BASE_TS, 1000)]
    pairs = [("B", "Y"), ("A", "X")]
    out, total = run_recovery(left, right, pairs, 100)
    assert out.reachable is True
    assert out.removed_pairs == (("A", "X"),)
    assert total == 100
    assert out.original_total == 0
    # target 1100 needs both removals
    out, total = run_recovery(left, right, pairs, 1100)
    assert out.removed_pairs == (("A", "X"), ("B", "Y"))
    assert total == 1100


@pytest.mark.parametrize("seed", range(300))
def test_recovery_against_brute_force(seed: int):
    rng = random.Random(40_000 + seed)
    n_left = rng.randint(0, 5)
    n_right = rng.randint(0, 5)
    left = [
        Txn(f"L{i}", BASE_TS + rng.randint(0, 4 * 86400), rng.choice(AMOUNTS))
        for i in range(n_left)
    ]
    right = [
        Txn(f"R{i}", BASE_TS + rng.randint(0, 4 * 86400), rng.choice(AMOUNTS))
        for i in range(n_right)
    ]

    base = Reconciler(left, right)
    pair_specs = set()
    # seed pairs that actually bite (belong to a legal candidate)
    if base.candidates:
        for _ in range(rng.randint(0, min(4, len(base.candidates)))):
            cand = rng.choice(base.candidates)
            pair_specs.add((rng.choice(cand.left_ids), rng.choice(cand.right_ids)))
    # plus a few arbitrary pairs
    if left and right:
        for _ in range(rng.randint(0, 3)):
            pair_specs.add((rng.choice(left).id, rng.choice(right).id))
    pair_list = list(pair_specs)[:8]

    structural_max = sum(c.weight for c in Reconciler(left, right).solve())
    target = rng.choice(
        [0, 1]
        + ([structural_max] if structural_max else [])
        + [rng.randint(0, max(structural_max, 1) + 50)]
    )

    out, total = run_recovery(left, right, pair_list, target)
    exp_reachable, exp_removed, exp_total = brute_force_recovery(
        left, right, pair_list, target
    )
    assert out.reachable == exp_reachable
    assert list(out.removed_pairs) == exp_removed
    assert total == exp_total
    # original total is invariant to the target and matches /reconcile
    kept_rec = Reconciler(left, right, excluded_pairs=set(pair_list))
    assert out.original_total == sum(c.weight for c in kept_rec.solve())
    # all-removed bound never decreases and matches the relaxed view
    full_rec = Reconciler(left, right)
    assert out.max_total == sum(c.weight for c in full_rec.solve())
    if out.reachable:
        assert total >= target
        # reported groups respect the remaining (kept) exclusions
        kept = set(pair_list) - set(out.removed_pairs)
        for cand in out.groups:
            for lid, rid in kept:
                assert not (lid in cand.left_ids and rid in cand.right_ids)


@pytest.mark.parametrize("seed", range(120))
def test_recovery_with_locked_groups(seed: int):
    rng = random.Random(50_000 + seed)
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

    # pick disjoint legal candidates as locked groups, then build pairs that
    # never hit them (so the batch stays valid)
    base = Reconciler(left, right)
    pool = list(base.candidates)
    rng.shuffle(pool)
    used_mask = 0
    locked_cands = []
    locked_ids = set()
    for cand in pool:
        if cand.mask & used_mask == 0 and rng.random() < 0.3:
            locked_cands.append(cand)
            used_mask |= cand.mask
            locked_ids.update(cand.left_ids)
            locked_ids.update(cand.right_ids)

    pair_specs = set()
    for cand in base.candidates:
        if any(i in locked_ids for i in (*cand.left_ids, *cand.right_ids)):
            continue
        if rng.random() < 0.3:
            pair_specs.add((rng.choice(cand.left_ids), rng.choice(cand.right_ids)))
    if left and right:
        for _ in range(2):
            l, r = rng.choice(left).id, rng.choice(right).id
            if l not in locked_ids and r not in locked_ids:
                pair_specs.add((l, r))
    pair_list = list(pair_specs)[:8]

    # validate under the locked reconciler exactly like the endpoint
    pairs_in = [ExcludedPair(left_id=l, right_id=r) for l, r in pair_list]
    resolved = resolve_excluded_pairs(
        {t.id for t in left}, {t.id for t in right}, pairs_in
    )
    rec = Reconciler(left, right, locked=locked_cands, excluded_pairs=resolved)
    rec.locked = locked_cands

    structural_max = sum(c.weight for c in locked_cands) + rec._max_value(
        ((1 << len(rec.txns)) - 1) ^ used_mask
    )
    target = rng.choice([0, rng.randint(0, max(structural_max, 1))])

    out = find_exclusion_recovery(rec, target)
    total = sum(c.weight for c in out.groups)
    # locked groups always survive any recovery
    locked_encs = {c.encoding for c in locked_cands}
    assert locked_encs <= {c.encoding for c in out.groups}
    if out.reachable:
        assert total >= target
        kept = set(pair_list) - set(out.removed_pairs)
        for cand in out.groups:
            for lid, rid in kept:
                assert not (lid in cand.left_ids and rid in cand.right_ids)
    else:
        assert total == out.max_total < target
        assert out.removed_pairs == ()


def test_eight_pair_worst_case_runs_fast():
    # eight disjoint unique-weight (superincreasing) groups, one pair each:
    # every subset must be examined before all-eight removal wins
    vals = [100 * (2**i) for i in range(8)]
    left = [Txn(f"L{i}", BASE_TS, vals[i]) for i in range(8)]
    right = [Txn(f"R{i}", BASE_TS, vals[i]) for i in range(8)]
    pairs = [(f"L{i}", f"R{i}") for i in range(8)]
    import time

    t0 = time.perf_counter()
    out, total = run_recovery(left, right, pairs, sum(vals))
    elapsed = time.perf_counter() - t0
    assert out.reachable is True
    assert len(out.removed_pairs) == 8
    assert total == sum(vals)
    assert elapsed < 2.0
