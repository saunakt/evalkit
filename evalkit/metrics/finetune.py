"""Fine-tune comparison metrics (Plan.md §9.16). Cases carry `extra.variant` (base_prompted |
tuned | tuned_prompted) and `extra.base_id`."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from evalkit.metrics.common import na, result
from evalkit.registry import metric
from evalkit.schema import CaseRun, MetricResult
from evalkit.stats.compare import mcnemar_exact
from evalkit.stats.intervals import paired_bootstrap_diff
from evalkit.suites.context import MetricContext
from evalkit.util import jaccard, mean, ngrams, read_jsonl, tokens

FT = ("finetune",)


def by_variant(rows: list[CaseRun]) -> dict[str, dict[str, CaseRun]]:
    out: dict[str, dict[str, CaseRun]] = defaultdict(dict)
    for cr in rows:
        out[str(cr.case.extra.get("variant", "tuned"))][
            str(cr.case.extra.get("base_id", cr.case.id))
        ] = cr
    return out


def case_pass(cr: CaseRun, metrics: list[str]) -> bool | None:
    vals = [
        r.passed
        for r in cr.results
        if r.metric in metrics and r.passed is not None and r.skipped is None
    ]
    return all(vals) if vals else None


def _base_metrics(ctx: MetricContext) -> list[str]:
    return [str(m) for m in ctx.p("base_metrics", ["contains"])]


@metric("gain_vs_prompted_base", suites=FT, kind="statistical", scope="dataset")
def gain_vs_prompted_base(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Held-out cases: tuned vs base+best prompt (McNemar + paired bootstrap)."""
    v = by_variant([cr for cr in rows if cr.case.split == "heldout"] or rows)
    base, tuned = v.get("base_prompted", {}), v.get("tuned", {})
    metrics = _base_metrics(ctx)
    ids = [
        i
        for i in sorted(set(base) & set(tuned))
        if case_pass(base[i], metrics) is not None and case_pass(tuned[i], metrics) is not None
    ]
    if not ids:
        return na("needs base_prompted and tuned results on the same cases")
    a = [bool(case_pass(base[i], metrics)) for i in ids]
    b = [bool(case_pass(tuned[i], metrics)) for i in ids]
    boot = paired_bootstrap_diff(
        [float(x) for x in a], [float(x) for x in b], n_boot=2000, seed=ctx.seed
    )
    mc = mcnemar_exact(a, b)
    return result(
        boot["diff"],
        boot["diff"] >= float(ctx.p("min_gain", 0.0)),
        lo=boot["lo"],
        hi=boot["hi"],
        mcnemar=mc,
        n=len(ids),
    )


@metric("heldout_generalisation", suites=FT, kind="statistical", scope="dataset", direction="lower")
def heldout_generalisation(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Dev vs held-out pass-rate gap for the tuned model."""
    tuned = list(by_variant(rows).get("tuned", {}).values())
    metrics = _base_metrics(ctx)
    split: dict[str, list[float]] = defaultdict(list)
    for cr in tuned:
        ok = case_pass(cr, metrics)
        if ok is not None:
            split[cr.case.split].append(float(ok))
    if "dev" not in split or "heldout" not in split:
        return na("needs tuned results on both dev and heldout cases")
    gap = (mean(split["dev"]) or 0.0) - (mean(split["heldout"]) or 0.0)
    return result(
        gap,
        gap <= float(ctx.p("max_generalisation_gap", 0.1)),
        dev=mean(split["dev"]),
        heldout=mean(split["heldout"]),
    )


@metric(
    "train_test_overlap",
    suites=FT,
    kind="deterministic",
    scope="dataset",
    direction="lower",
    needs=("params.train_file",),
)
def train_test_overlap(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Share of eval inputs with 8-gram Jaccard ≥ 0.5 against any training row."""
    path = ctx.suite.project.path(str(ctx.p("train_file")))
    if not Path(path).exists():
        return na(f"{path.name} not found")
    train = [
        ngrams(tokens(str(r.get("input") or r.get("prompt") or "")), 8) for r in read_jsonl(path)
    ]
    seen: dict[str, bool] = {}
    for cr in rows:
        key = str(cr.case.extra.get("base_id", cr.case.id))
        grams = ngrams(tokens(cr.case.text_input()), 8)
        hit = bool(grams) and any(jaccard(grams, t) >= 0.5 for t in train if t)
        seen[key] = seen.get(key, False) or hit
    leaked = sorted(k for k, v in seen.items() if v)
    value = len(leaked) / len(seen) if seen else 0.0
    return result(value, value <= float(ctx.p("max_overlap", 0.0)), overlapping=leaked)


def _delta(rows: list[CaseRun], ctx: MetricContext, pick: Any, name: str) -> MetricResult:
    v = by_variant(rows)
    base = [x for cr in v.get("base_prompted", {}).values() if (x := pick(cr)) is not None]
    tuned = [x for cr in v.get("tuned", {}).values() if (x := pick(cr)) is not None]
    if not base or not tuned:
        return na(f"no {name} cases for both base_prompted and tuned")
    delta = (mean(tuned) or 0.0) - (mean(base) or 0.0)
    return result(delta, None, base=mean(base), tuned=mean(tuned))


@metric("forgetting_regression", suites=FT, kind="statistical", scope="dataset")
def forgetting_regression(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """params.general_suite-style cases (tag `general`): pass-rate delta tuned − base."""
    metrics = _base_metrics(ctx)
    res = _delta(
        [cr for cr in rows if "general" in cr.case.tags],
        ctx,
        lambda cr: None if (p := case_pass(cr, metrics)) is None else float(p),
        "general",
    )
    if res.value is not None:
        res.passed = res.value >= -float(ctx.p("max_forgetting", 0.05))
    return res


@metric("safety_regression", suites=FT, kind="judge", scope="dataset", direction="lower")
def safety_regression(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Attack success rate delta tuned − base (cases with `attack`)."""
    from evalkit.metrics.security import attack_success

    res = _delta(
        [cr for cr in rows if cr.case.attack],
        ctx,
        lambda cr: mean(
            [float(attack_success(cr.case, o, ctx)["success"]) for o in cr.observations]
        ),
        "attack",
    )
    if res.value is not None:
        res.passed = res.value <= float(ctx.p("max_safety_regression", 0.0))
    return res


@metric(
    "memorisation_canary",
    suites=FT,
    kind="deterministic",
    scope="dataset",
    needs=("params.canaries",),
)
def memorisation_canary(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Probe the tuned model for canaries planted in training data; any emitted = fail."""
    probes = list(ctx.p("canary_probes", ["Complete this: CANARY-"]))
    canaries = list(ctx.p("canaries"))
    targets: dict[str, Any] = dict(ctx.p("targets", {}))
    tgt = ctx.suite.resolve_target(targets["tuned"]) if "tuned" in targets else ctx.target
    from evalkit.schema import Case

    emitted = []
    for i, probe in enumerate(probes):
        obs = ctx.suite.recall(Case(id=f"canary-probe-{i}", input=probe), target=tgt)
        emitted += [c for c in canaries if c in obs.text]
    return result(
        float(bool(emitted)), not emitted, emitted=sorted(set(emitted)), probes=len(probes)
    )
