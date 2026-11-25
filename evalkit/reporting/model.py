"""Build the report data model (one dict per suite + summary) from the results store."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from evalkit.config import Thresholds
from evalkit.gate import GateResult, SuiteGate, evaluate
from evalkit.registry import METRICS, SUITES, load_all
from evalkit.reporting.aggregate import MetricSummary, SuiteSummary, summarize_suite
from evalkit.reporting.store import StoredRun
from evalkit.schema import ResultRow

CHECKLIST = [
    ("Quality thresholds met for every selected suite", "suites", None),
    ("No must-pass case failed in any repeat", "must_pass", None),
    ("Security attack success rate under threshold", "metric", ("security", "attack_success_rate")),
    ("Agent task success meets target", "metric", ("agent", "task_success")),
    ("RAG retrieval recall meets target", "metric", ("rag", "recall_at_k")),
    ("Faithfulness meets target", "metric", ("rag", "faithfulness")),
    ("Bias gap within threshold", "metric", ("responsible_ai", "bias_gap")),
    ("PII leakage rate within threshold", "metric", ("responsible_ai", "pii_leakage_rate")),
    ("Guardrail recall meets target", "metric", ("guardrails", "guard_recall")),
    ("Latency p95 within budget", "metric", ("performance", "latency_p95")),
    ("Compliance requirements covered", "metric", ("compliance", "requirement_coverage")),
    ("Every judge used is calibrated", "judges", None),
    ("No regression vs baseline", "regression", None),
]


def subset_rows(
    rows: list[ResultRow], suites: list[str] | None, metrics: list[str] | None
) -> list[ResultRow]:
    return [
        r
        for r in rows
        if (suites is None or r.suite in suites) and (metrics is None or r.metric in metrics)
    ]


def selected(
    stored: StoredRun, suites: list[str] | None, metrics: list[str] | None
) -> dict[str, list[str]]:
    out = {}
    for suite, ids in stored.info.suites.items():
        if suites is not None and suite not in suites:
            continue
        out[suite] = [m for m in ids if metrics is None or m in metrics]
    return out


def compute(
    stored: StoredRun,
    thr: Thresholds,
    baseline: StoredRun | None,
    suites: list[str] | None = None,
    metrics: list[str] | None = None,
) -> tuple[dict[str, list[str]], dict[str, SuiteSummary], GateResult, list[ResultRow]]:
    sel = selected(stored, suites, metrics)
    rows = subset_rows(stored.rows, list(sel), metrics)
    summaries = {
        s: summarize_suite(s, [r for r in rows if r.suite == s], ids) for s, ids in sel.items()
    }
    base_rows = subset_rows(baseline.rows, list(sel), metrics) if baseline else None
    return sel, summaries, evaluate(thr, summaries, rows, base_rows), rows


def metric_entry(ms: MetricSummary, sg: SuiteGate) -> dict[str, Any]:
    spec = METRICS.get(ms.metric) if ms.metric in METRICS else None
    check = next((c for c in sg.checks if c.metric == ms.metric), None)
    if ms.not_tested:
        status = "NOT TESTED"
    elif check is not None:
        status = "PASS" if check.ok else "FAIL"
    else:
        status = "INFO"
    if status == "PASS" and ms.uncalibrated:
        status = "WARN"
    target = None
    if check is not None and check.bound != "self":
        target = f"{'≥' if check.bound == 'min' else '≤'} {check.target:g}" + (
            " pass rate" if check.basis == "pass_rate" else ""
        )
    return {
        **ms.as_dict(),
        "status": status,
        "target": target,
        "description": spec.description if spec else "",
        "skipped_summary": ", ".join(f"{k} ({v})" for k, v in ms.skipped_reasons.items()),
    }


def suite_report(
    suite: str,
    ids: list[str],
    summ: SuiteSummary,
    sg: SuiteGate,
    stored: StoredRun,
    baseline: StoredRun | None,
) -> dict[str, Any]:
    info = stored.info
    metrics = [metric_entry(summ.metrics[m], sg) for m in ids]
    judges = [{"id": jid, **st} for jid, st in sorted(info.judges.items())]
    return {
        "run_id": info.run_id,
        "suite": suite,
        "status": sg.status,
        "doc": (SUITES.get(suite).__doc__ or "").strip().split("\n")[0] if suite in SUITES else "",
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "failures": sg.failures,
        "warnings": sg.warnings,
        "tested": {
            "cases": summ.n_cases,
            "repeats": info.repeats,
            "seed": info.seed,
            "target": info.target,
            "profile": info.profile,
            "dataset": info.datasets.get(suite, {}),
            "roles": info.roles,
        },
        "overall": summ.overall,
        "metrics": metrics,
        "comparison": {"baseline": baseline.run_id if baseline else None, "metrics": sg.comparison},
        "slices": summ.slices,
        "failing": summ.failing,
        "flaky": summ.flaky_cases,
        "panels": stored.panels.get(suite, []),
        "judges": judges,
        "not_tested": [
            {"metric": m["metric"], "reason": m["skipped_summary"] or "no cases"}
            for m in metrics
            if m["status"] == "NOT TESTED"
        ],
        "cost_usd": summ.cost_usd,
        "tokens": summ.tokens,
        "trace_url_template": None,
    }


def checklist(
    gate: GateResult,
    summaries: dict[str, SuiteSummary],
    info_judges: dict[str, Any],
    has_baseline: bool,
) -> list[dict[str, str]]:
    items = []
    for label, kind, ref in CHECKLIST:
        state = "–"
        if kind == "suites":
            state = "✓" if all(s.status != "FAIL" for s in gate.suites.values()) else "✗"
        elif kind == "must_pass":
            state = "✓" if not gate.must_pass_failures else "✗"
        elif kind == "judges" and info_judges:
            state = "✓" if all(j.get("calibrated") for j in info_judges.values()) else "✗"
        elif kind == "regression" and has_baseline:
            regressed = any(
                d.get("regressed") for s in gate.suites.values() for d in s.comparison.values()
            )
            state = "✗" if regressed else "✓"
        elif kind == "metric" and ref is not None and ref[0] in summaries:
            state = _metric_tick(gate, summaries, ref[0], ref[1])
        items.append({"item": label, "state": state})
    return items


def _metric_tick(
    gate: GateResult, summaries: dict[str, SuiteSummary], suite: str, metric: str
) -> str:
    ms = summaries[suite].metrics.get(metric)
    if ms is None or ms.not_tested:
        return "–"
    check = next((c for c in gate.suites[suite].checks if c.metric == metric), None)
    return "–" if check is None else ("✓" if check.ok else "✗")


def summary_report(
    stored: StoredRun,
    sel: dict[str, list[str]],
    summaries: dict[str, SuiteSummary],
    gate: GateResult,
    baseline: StoredRun | None,
) -> dict[str, Any]:
    load_all()
    info = stored.info
    rows = []
    for suite in sel:
        summ, sg = summaries[suite], gate.suites[suite]
        key = [metric_entry(summ.metrics[m], sg) for m in sel[suite]]
        rows.append(
            {
                "suite": suite,
                "status": sg.status,
                "n": summ.n_cases,
                "overall": summ.overall,
                "cost_usd": summ.cost_usd,
                "key_metrics": [k for k in key if k["status"] != "NOT TESTED"][:6],
                "failures": sg.failures,
                "warnings": sg.warnings,
            }
        )
    off = [
        {"what": f"suite {s}", "reason": "switched off"}
        for s in SUITES.names()
        if s not in info.suites
    ]
    skipped = [
        {
            "what": f"{s}.{m}",
            "reason": ", ".join(summaries[s].metrics[m].skipped_reasons) or "no applicable cases",
        }
        for s in sel
        for m in sel[s]
        if summaries[s].metrics[m].not_tested
    ]
    return {
        "run_id": info.run_id,
        "passed": gate.passed,
        "verdict": "PASS" if gate.passed else "FAIL",
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "info": info.model_dump(),
        "suites": rows,
        "must_pass_failures": gate.must_pass_failures,
        "checklist": checklist(gate, summaries, info.judges, baseline is not None),
        "not_tested": skipped + off,
        "baseline": baseline.run_id if baseline else None,
        "ledger": info.ledger,
    }
