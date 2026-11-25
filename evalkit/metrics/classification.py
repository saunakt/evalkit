"""Dataset classification metrics over `label` (Plan.md §9.1), via scikit-learn."""

from __future__ import annotations

from typing import Any

from sklearn import metrics as skm

from evalkit.metrics.common import extract_decision, na, result, scored
from evalkit.registry import metric
from evalkit.schema import CaseRun, MetricResult
from evalkit.suites.context import MetricContext

CLS_SUITES = ("quality", "multimodal")


def predictions(rows: list[CaseRun], ctx: MetricContext) -> tuple[list[str], list[str], list[str]]:
    """(y_true, y_pred, labels): prediction = first of params.labels found in the output."""
    labels = [
        str(x) for x in ctx.p("labels") or sorted({cr.case.label for cr in rows if cr.case.label})
    ]
    y_true, y_pred = [], []
    for cr in rows:
        if cr.case.label is None:
            continue
        for obs in cr.observations:
            y_true.append(str(cr.case.label))
            y_pred.append(extract_decision(obs.text, labels) or "(none)")
    return y_true, y_pred, labels


def _averaging(labels: list[str], ctx: MetricContext) -> dict[str, Any]:
    pos = ctx.p("positive")
    if pos is not None and len(labels) <= 2:
        return {"average": "binary", "pos_label": str(pos)}
    return {"average": "macro", "labels": labels}


def _score(fn: Any, rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    y_true, y_pred, labels = predictions(rows, ctx)
    if not y_true:
        return na("no labelled cases")
    kw = _averaging(labels, ctx)
    value = float(fn(y_true, y_pred, zero_division=0, **kw))
    minimum = ctx.p("min_score")
    details = {"n": len(y_true), "average": kw["average"]}
    return (
        scored(value, float(minimum), **details)
        if minimum is not None
        else result(value, None, **details)
    )


@metric("accuracy", suites=CLS_SUITES, kind="statistical", scope="dataset")
def accuracy(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Share of predictions equal to the label."""
    y_true, y_pred, _ = predictions(rows, ctx)
    if not y_true:
        return na("no labelled cases")
    value = float(skm.accuracy_score(y_true, y_pred))
    minimum = ctx.p("min_score")
    return (
        scored(value, float(minimum), n=len(y_true))
        if minimum is not None
        else result(value, None, n=len(y_true))
    )


@metric("precision", suites=CLS_SUITES, kind="statistical", scope="dataset")
def precision(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Precision for params.positive (binary) or macro-averaged."""
    return _score(skm.precision_score, rows, ctx)


@metric("recall", suites=CLS_SUITES, kind="statistical", scope="dataset")
def recall(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Recall for params.positive (binary) or macro-averaged."""
    return _score(skm.recall_score, rows, ctx)


@metric("f1", suites=CLS_SUITES, kind="statistical", scope="dataset")
def f1(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """F1 for params.positive (binary) or macro-averaged."""
    return _score(skm.f1_score, rows, ctx)


@metric("confusion_matrix", suites=CLS_SUITES, kind="statistical", scope="dataset", unit="count")
def confusion_matrix(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Confusion matrix (informational; rows = true label, columns = predicted)."""
    y_true, y_pred, labels = predictions(rows, ctx)
    if not y_true:
        return na("no labelled cases")
    cols = labels + (["(none)"] if "(none)" in y_pred else [])
    cm = skm.confusion_matrix(y_true, y_pred, labels=cols)
    return result(float(len(y_true)), None, labels=cols, matrix=cm.tolist())
