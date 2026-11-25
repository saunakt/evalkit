"""Agreement statistics: Cohen's kappa, Krippendorff's alpha, judge agreement (Plan.md §10)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from sklearn.metrics import cohen_kappa_score


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two equal, non-empty label lists")
    if len(set(a) | set(b)) == 1:
        return 1.0
    return float(cohen_kappa_score(list(a), list(b)))


def krippendorff_alpha_nominal(matrix: Sequence[Sequence[Any]]) -> float:
    """Nominal alpha. `matrix[coder][unit]`; None marks a missing rating."""
    units: list[list[Any]] = []
    for u in range(max((len(r) for r in matrix), default=0)):
        vals = [row[u] for row in matrix if u < len(row) and row[u] is not None]
        if len(vals) >= 2:
            units.append(vals)
    if not units:
        return 1.0
    coincidence: defaultdict[tuple[Any, Any], float] = defaultdict(float)
    for vals in units:
        m = len(vals)
        for i, x in enumerate(vals):
            for j, y in enumerate(vals):
                if i != j:
                    coincidence[(x, y)] += 1 / (m - 1)
    totals: defaultdict[Any, float] = defaultdict(float)
    for (x, _), w in coincidence.items():
        totals[x] += w
    n = sum(totals.values())
    observed = sum(w for (x, y), w in coincidence.items() if x != y)
    expected = sum(totals[x] * totals[y] for x in totals for y in totals if x != y) / (n - 1)
    if expected == 0:
        return 1.0
    return 1 - observed / expected


def judge_agreement(
    human: Sequence[str], judge: Sequence[str], positive: str = "FAIL"
) -> dict[str, Any]:
    """Accuracy, kappa, precision/recall/F1 for `positive` (FAIL), confusion matrix."""
    pairs = list(zip(human, judge, strict=True))
    n = len(pairs)
    tp = sum(1 for h, j in pairs if h == positive and j == positive)
    fp = sum(1 for h, j in pairs if h != positive and j == positive)
    fn = sum(1 for h, j in pairs if h == positive and j != positive)
    tn = n - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "kappa": cohen_kappa(human, judge) if n else 0.0,
        "fail_precision": prec,
        "fail_recall": rec,
        "fail_f1": f1,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
