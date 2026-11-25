"""Voice, image and document metrics (Plan.md §9.15)."""

from __future__ import annotations

from typing import Any

from evalkit.judges.runner import inline_judge
from evalkit.metrics.common import case_sections, from_verdict, na, result
from evalkit.registry import metric
from evalkit.schema import Case, CaseRun, MetricResult, Observation
from evalkit.suites.context import MetricContext
from evalkit.util import extract_json, normalize_text, percentile, write_jsonl

MM = ("multimodal",)
inline_judge(
    "refuses_unreadable",
    "PASS if the reply says it cannot read or tell what the image/audio shows instead of guessing.",
    metric="refuses_unreadable",
)
inline_judge(
    "visual_hallucination",
    "PASS if the description states only what the reference "
    "description supports, inventing nothing.",
    style="reference",
    metric="visual_hallucination",
)


@metric(
    "wer",
    requires_extra="voice",
    suites=MM,
    kind="deterministic",
    needs=("expected_transcript", "pkg:jiwer"),
    direction="lower",
)
def wer(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """jiwer WER(expected_transcript, obs.transcript); sliced by tags in the report."""
    from evalkit.engines.jiwer_engine import word_error_rate

    if obs.transcript is None:
        return na("target returned no transcript")
    value = word_error_rate(str(case.expected_transcript), obs.transcript)
    limit = float(ctx.p("max_wer", 0.15))
    return result(value, value <= limit, limit)


@metric("entity_accuracy", suites=MM, kind="deterministic", needs=("entities",))
def entity_accuracy(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Share of case.entities present exactly in the transcript / output."""
    text = (obs.transcript or "") + "\n" + obs.text
    missing = [e for e in case.entities if e not in text]
    n = len(case.entities)
    return result((n - len(missing)) / n, not missing, missing=missing)


@metric(
    "voice_latency",
    suites=MM,
    kind="statistical",
    scope="dataset",
    direction="lower",
    unit="seconds",
)
def voice_latency(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """p95 of asr+llm+tts spans (or obs.latency_s)."""
    lats = []
    for cr in rows:
        for o in cr.observations:
            parts = [s.latency_s for s in o.spans if s.name in ("asr", "llm", "tts")]
            lats.append(sum(parts) if parts else o.latency_s)
    if not lats:
        return na("no voice observations")
    value = percentile(lats, 95)
    limit = ctx.p("max_voice_latency_s")
    return result(value, None if limit is None or value is None else value <= float(limit))


def _event_check(obs: Observation, kind: str, ctx: MetricContext) -> MetricResult:
    events = [
        e for e in obs.events if e.get("type") in (kind, f"{kind}_handled", "premature_cutoff")
    ]
    if not events:
        return na(f"target reported no {kind} events")
    cutoffs = sum(1 for e in events if e.get("type") == "premature_cutoff")
    gaps = [float(e.get("gap_s", 0)) for e in events if "gap_s" in e]
    max_gap = float(ctx.p("max_gap_s", 1.0))
    ok = cutoffs == 0 and all(g <= max_gap for g in gaps)
    return result(1.0 if ok else 0.0, ok, premature_cutoffs=cutoffs, gaps=gaps)


@metric("barge_in", suites=MM, kind="deterministic")
def barge_in(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Target-reported barge_in_handled events; no premature cut-offs."""
    return _event_check(obs, "barge_in", ctx)


@metric("turn_detection", suites=MM, kind="deterministic")
def turn_detection(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Turn-taking events: gap_s within params.max_gap_s and no premature cut-off."""
    return _event_check(obs, "turn_detection", ctx)


def _export(ctx: MetricContext, name: str, rows: list[CaseRun]) -> MetricResult:
    path = ctx.run_dir / "human_review" / f"{name}.jsonl"
    items = [
        {
            "id": cr.case.id,
            "media": cr.case.media,
            "input": cr.case.text_input(),
            "output": cr.observations[0].text if cr.observations else "",
        }
        for cr in rows
    ]
    write_jsonl(path, items)
    return MetricResult(
        metric=name,
        skipped=f"human review: {len(items)} items exported to {path.name}",
        details={"export": str(path)},
    )


def _register_human(name: str, doc: str) -> None:
    def fn(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
        return _export(ctx, name, rows)

    fn.__doc__ = doc
    metric(name, suites=MM, kind="human", scope="dataset")(fn)


for _name, _doc in (
    ("tts_quality", "Human MOS rating of synthesised speech (exported for Label Studio)."),
    ("prompt_adherence", "Generated media follows the prompt (exported for human review)."),
    ("media_safety", "Generated media is safe (exported for human review)."),
    ("media_bias", "Generated media shows no demographic bias (exported for human review)."),
    ("provenance", "Generated media carries provenance / watermark (exported for review)."),
):
    _register_human(_name, _doc)


def _field_ok(want: Any, got: Any, tol: float) -> bool:
    if isinstance(want, int | float) and isinstance(got, int | float):
        return abs(float(want) - float(got)) <= tol
    return normalize_text(str(want)) == normalize_text(str(got))


@metric("field_accuracy", suites=MM, kind="deterministic", needs=("expected_fields",))
def field_accuracy(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Extracted JSON fields vs expected_fields (strings normalised, numbers with tolerance)."""
    data = extract_json(obs.text) or {}
    tol = float(ctx.p("tolerance", 0.01))
    wrong = {
        k: data.get(k)
        for k, v in case.expected_fields.items()
        if not isinstance(data, dict) or not _field_ok(v, data.get(k), tol)
    }
    n = len(case.expected_fields)
    return result((n - len(wrong)) / n, not wrong, wrong=wrong)


@metric("image_classification_f1", suites=MM, kind="statistical", scope="dataset")
def image_classification_f1(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Macro F1 over `label` for image cases."""
    from evalkit.metrics.classification import f1

    return f1([cr for cr in rows if cr.case.media], ctx)  # type: ignore[no-any-return]


@metric("refuses_unreadable", suites=MM, kind="judge", needs=("tag:unreadable",))
def refuses_unreadable(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Cases tagged unreadable: judge 'says it cannot tell'."""
    return from_verdict(
        ctx.judge.binary("refuses_unreadable", sections=case_sections(case, obs), case_id=case.id)
    )


@metric("visual_injection", suites=MM, kind="judge", needs=("attack",), direction="lower")
def visual_injection(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Security-style success check on images that contain instructions."""
    from evalkit.metrics.security import attack_success

    res = attack_success(case, obs, ctx)
    return result(1.0 if res["success"] else 0.0, not res["success"], reasons=res["reasons"])


@metric("visual_hallucination", suites=MM, kind="judge", needs=("reference",))
def visual_hallucination(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Reference judge vs the reference description."""
    secs = {**case_sections(case, obs), "Reference description": case.reference}
    return from_verdict(ctx.judge.binary("visual_hallucination", sections=secs, case_id=case.id))


@metric("image_token_cost", suites=MM, kind="statistical", scope="dataset", unit="count")
def image_token_cost(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Mean tokens (and cost) per image case."""
    obs = [o for cr in rows if cr.case.media for o in cr.observations]
    if not obs:
        return na("no image cases ran")
    toks = [o.tokens_in + o.tokens_out for o in obs]
    cost = [o.cost_usd or 0.0 for o in obs]
    return result(sum(toks) / len(toks), None, mean_cost_usd=sum(cost) / len(cost))
