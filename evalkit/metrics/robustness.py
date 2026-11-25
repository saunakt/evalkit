"""Robustness / metamorphic metrics over variant groups (Plan.md §9.3).

The robustness suite expands each case into a group: the original (`extra.transform =
"original"`) plus variants (`extra.group`, `extra.transform`, `extra.relation`).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evalkit.metrics.common import case_ok, extract_decision, na, result
from evalkit.registry import metric
from evalkit.schema import CaseRun, MetricResult
from evalkit.stats.intervals import paired_bootstrap_diff
from evalkit.suites.context import MetricContext
from evalkit.util import mean

ROB_SUITES = ("robustness",)


def decide(text: str, cr: CaseRun, ctx: MetricContext) -> str | None:
    """Decision via params.decision: regex, label list, or {"judge": question}."""
    spec = ctx.p("decision")
    if isinstance(spec, dict) and "judge" in spec:
        from evalkit.judges.runner import inline_judge

        jid = inline_judge("decision", "Answer YES (PASS) or NO (FAIL) about the reply.")
        v = ctx.judge.binary(
            jid, criterion=str(spec["judge"]), sections={"Reply": text}, case_id=cr.case.id
        )
        return "yes" if v.passed else "no"
    return extract_decision(text, spec)


def groups(rows: list[CaseRun]) -> dict[str, dict[str, Any]]:
    """group id → {"original": CaseRun | None, "variants": [CaseRun]}."""
    out: dict[str, dict[str, Any]] = defaultdict(lambda: {"original": None, "variants": []})
    for cr in rows:
        gid = str(cr.case.extra.get("group", cr.case.id))
        if cr.case.extra.get("transform", "original") == "original":
            out[gid]["original"] = cr
        else:
            out[gid]["variants"].append(cr)
    return dict(sorted(out.items()))


def _invariance(
    rows: list[CaseRun], ctx: MetricContext, relation: str, transforms: set[str] | None = None
) -> MetricResult:
    per_group: dict[str, float] = {}
    flips: list[dict[str, Any]] = []
    for gid, g in groups(rows).items():
        orig = g["original"]
        variants = [
            v
            for v in g["variants"]
            if v.case.extra.get("relation", "same_decision") == relation
            and (transforms is None or str(v.case.extra.get("transform")) in transforms)
        ]
        if orig is None or not variants or not orig.observations:
            continue
        base = decide(orig.observations[0].text, orig, ctx)
        hits = []
        for v in variants:
            d = decide(v.observations[0].text, v, ctx) if v.observations else None
            same = d == base and d is not None
            hits.append(same if relation == "same_decision" else not same)
            if not hits[-1]:
                flips.append(
                    {
                        "group": gid,
                        "variant": v.case.id,
                        "transform": v.case.extra.get("transform"),
                        "original_decision": base,
                        "variant_decision": d,
                    }
                )
        per_group[gid] = sum(hits) / len(hits)
    if not per_group:
        return na(f"no groups with {relation} variants")
    value = mean(list(per_group.values())) or 0.0
    minimum = float(ctx.p("min_invariance", 0.9))
    return result(value, value >= minimum, minimum, groups=per_group, failures=flips)


@metric("metamorphic_invariance", suites=ROB_SUITES, kind="deterministic", scope="dataset")
def metamorphic_invariance(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Share of same_decision variants whose decision equals the original's."""
    return _invariance(rows, ctx, "same_decision")


@metric("metamorphic_change", suites=ROB_SUITES, kind="deterministic", scope="dataset")
def metamorphic_change(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Variants with relation different_decision must change the decision."""
    return _invariance(rows, ctx, "different_decision")


@metric("rephrase_consistency", suites=ROB_SUITES, kind="deterministic", scope="dataset")
def rephrase_consistency(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Invariance restricted to paraphrase (and explicit rephrase) variants."""
    return _invariance(rows, ctx, "same_decision", {"paraphrase", "explicit", "translit_hinglish"})


@metric(
    "perturbation_drop", suites=ROB_SUITES, kind="statistical", scope="dataset", direction="lower"
)
def perturbation_drop(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Pass rate of params.base_metrics on originals minus on variants (bootstrap CI)."""
    base_metrics = list(ctx.p("base_metrics", ["contains"]))
    orig_vals, var_vals = [], []
    for g in groups(rows).values():
        orig = g["original"]
        if orig is None:
            continue
        o_ok = [case_ok(orig, m) for m in base_metrics]
        o = all(x for x in o_ok if x is not None) if any(x is not None for x in o_ok) else None
        for v in g["variants"]:
            v_ok = [case_ok(v, m) for m in base_metrics]
            if o is None or all(x is None for x in v_ok):
                continue
            orig_vals.append(float(o))
            var_vals.append(float(all(x for x in v_ok if x is not None)))
    if not orig_vals:
        return na(f"no originals/variants scored by {base_metrics}")
    boot = paired_bootstrap_diff(var_vals, orig_vals, n_boot=2000, seed=ctx.seed)
    drop = boot["diff"]
    limit = float(ctx.p("max_drop", 0.05))
    return result(
        drop,
        drop <= limit,
        limit,
        lo=boot["lo"],
        hi=boot["hi"],
        n=len(orig_vals),
        original_pass_rate=mean(orig_vals),
        variant_pass_rate=mean(var_vals),
        base_metrics=base_metrics,
    )
