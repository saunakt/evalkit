"""Unit tests for quality / prompt / robustness / classic metrics (fixtures, fake judge)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalkit.schema import Case, MetricResult, Observation, ToolCall
from evalkit.targets.base import load_target
from tests.metric_helpers import ROOT, case_run, make_ctx, mr, run_case, run_dataset


def C(**kw: object) -> Case:
    return Case.model_validate({"id": "c1", **kw})


def O(text: str = "", **kw: object) -> Observation:
    return Observation.model_validate({"text": text, **kw})


# ------------------------------------------------------------------ deterministic


@pytest.mark.parametrize(
    ("metric", "case", "text", "passed"),
    [
        ("exact_match", C(reference="Thirty days."), "  thirty   DAYS ", True),
        ("exact_match", C(label="yes"), "no", False),
        ("contains", C(must_include=["30-day", "refund"]), "A 30-DAY window; Refund later", True),
        ("contains", C(must_include=["30-day", "pickup"]), "a 30-day window", False),
        ("not_contains", C(must_not_include=["CANARY"]), "all good", True),
        ("not_contains", C(must_not_include=["canary"]), "leak: CANARY-7Q2X", False),
        ("regex", C(regex=r"\[returns\]"), "policy [returns]", True),
        ("regex", C(regex=r"^\d+$"), "thirty", False),
        ("numeric_tolerance", C(reference="10000"), "Up to ₹10,000 only", True),
        ("numeric_tolerance", C(reference="30"), "within 31 days", False),
    ],
)
def test_deterministic_checks(
    tmp_path: Path, metric: str, case: Case, text: str, passed: bool
) -> None:
    assert run_case(metric, case, O(text), make_ctx(tmp_path)).passed is passed


def test_contains_details_list_missing(tmp_path: Path) -> None:
    r = run_case("contains", C(must_include=["a", "zz"]), O("a b"), make_ctx(tmp_path))
    assert r.value == 0.5 and r.details["missing"] == ["zz"]


def test_json_schema(tmp_path: Path) -> None:
    case = C(schema={"type": "object", "required": ["status"]})
    ctx = make_ctx(tmp_path)
    assert run_case("json_schema", case, O('```json\n{"status": "ok"}\n```'), ctx).passed
    bad = run_case("json_schema", case, O('{"x": 1}'), ctx)
    assert bad.passed is False and "status" in bad.details["reason"]
    assert run_case("json_schema", case, O("no json"), ctx).passed is False


def test_numeric_tolerance_json_path(tmp_path: Path) -> None:
    case = C(reference="12500", extra={"number_path": "amount"})
    assert run_case(
        "numeric_tolerance", case, O('{"amount": 12500.004}'), make_ctx(tmp_path)
    ).passed


def test_length_and_readability(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, {"max_words": 3})
    assert run_case("length", C(), O("one two three"), ctx).passed
    assert run_case("length", C(), O("one two three four"), ctx).passed is False
    assert run_case("length", C(), O("x"), make_ctx(tmp_path)).skipped.startswith("n/a")
    r = run_case("readability", C(), O("The cat sat on the mat. It was warm."), ctx)
    assert r.value is not None and r.value > 80 and r.passed is None


def test_needs_skip_reason(tmp_path: Path) -> None:
    r = run_case("contains", C(), O("x"), make_ctx(tmp_path))
    assert r.skipped == "n/a: needs must_include"


# ------------------------------------------------------------------ text similarity


def test_token_f1_and_missing_extras(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    r = run_case(
        "token_f1", C(reference="returns within 30 days"), O("returns within 30 days"), ctx
    )
    assert r.value == pytest.approx(1.0) and r.passed is None
    half = run_case("token_f1", C(reference="a b c d"), O("a b"), ctx)
    assert half.value == pytest.approx(2 * 1 * 0.5 / 1.5)
    for m, pkg in (
        ("bleu", "sacrebleu"),
        ("rouge", "rouge-score"),
        ("meteor", "nltk"),
        ("bertscore", "bert-score"),
        ("cosine_similarity", "sentence-transformers"),
    ):
        res = run_case(m, C(reference="x"), O("x"), ctx)
        assert res.skipped is None or res.skipped == f"{pkg} not installed"


# ------------------------------------------------------------------ classification


def test_classification_metrics(tmp_path: Path) -> None:
    rows = [
        case_run(C(id="a", label="harmful"), ["This is harmful."]),
        case_run(C(id="b", label="benign"), ["Looks benign"]),
        case_run(C(id="c", label="harmful"), ["benign, I think"]),
        case_run(C(id="d", label="benign"), ["benign"]),
    ]
    ctx = make_ctx(tmp_path, {"labels": ["harmful", "benign"], "positive": "harmful"})
    assert run_dataset("accuracy", rows, ctx).value == pytest.approx(0.75)
    assert run_dataset("precision", rows, ctx).value == pytest.approx(1.0)
    assert run_dataset("recall", rows, ctx).value == pytest.approx(0.5)
    assert run_dataset("f1", rows, ctx).value == pytest.approx(2 / 3)
    cm = run_dataset("confusion_matrix", rows, ctx)
    assert cm.details["matrix"] == [[1, 1], [0, 2]]


# ------------------------------------------------------------------ judged


def test_judge_binary_per_criterion(tmp_path: Path) -> None:
    script = {"judge_script": [{"case": "c1", "match": "polite", "verdict": "FAIL"}]}
    ctx = make_ctx(tmp_path, script=script)
    r = run_case("judge_binary", C(criteria=["accurate", "polite"]), O("x"), ctx)
    assert r.value == 0.5 and r.passed is False and r.details["failed"] == ["polite"]
    assert r.details["uncalibrated"] is True


def test_completeness_rubric_reference_geval_tone(tmp_path: Path) -> None:
    ctx = make_ctx(
        tmp_path,
        {"tone": "warm"},
        script={"judge_script": [{"judge": "quality_rubric", "score": 3}]},
    )
    assert run_case("completeness_checklist", C(criteria=["a", "b"]), O("x"), ctx).value == 1.0
    assert run_case("judge_rubric", C(input="q"), O("x"), ctx).value == pytest.approx(0.5)
    assert run_case("judge_reference", C(reference="r"), O("x"), ctx).passed is True
    assert run_case("g_eval", C(criteria=["clear"]), O("x"), ctx).value == 1.0
    assert run_case("tone", C(), O("x"), ctx).value == 1.0
    assert run_case("tone", C(), O("x"), make_ctx(tmp_path)).skipped == "n/a: needs params.tone"


def test_judge_pairwise_needs_baseline(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, script={"judge_script": [{"judge": "pairwise", "winner": "tie"}]})
    assert run_case("judge_pairwise", C(), O("x"), ctx).skipped.startswith("n/a")
    r = run_case("judge_pairwise", C(extra={"candidate_b": "y"}), O("x"), ctx)
    assert r.value == 0.5 and r.passed is True


def test_self_consistency(tmp_path: Path) -> None:
    rows = [case_run(C(id="a"), ["Yes.", "yes", "No"]), case_run(C(id="b"), ["x", "x"])]
    r = run_dataset("self_consistency", rows, make_ctx(tmp_path))
    assert r.value == pytest.approx((2 / 3 + 1) / 2) and r.details["inconsistent"] == ["a"]


# ------------------------------------------------------------------ prompt


def test_instruction_polarity_filters(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    pos = C(criteria=["x"], extra={"polarity": "negative"})
    assert run_case("instruction_positive", pos, O("y"), ctx).skipped.startswith("n/a")
    assert run_case("instruction_negative", pos, O("y"), ctx).passed is True


def test_out_of_scope_and_conflict(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, {"scope": "orders"})
    case = C(tags=["out_of_scope"], must_not_include=["def "])
    assert run_case("out_of_scope_redirect", case, O("I only help with orders"), ctx).passed
    assert run_case("out_of_scope_redirect", case, O("def f(): pass"), ctx).passed is False
    assert run_case("instruction_conflict", C(criteria=["no cancel"]), O("x"), ctx).passed


def test_format_compliance(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    case = C(schema={"type": "object", "required": ["status"]}, language="hi")
    bad = run_case("format_compliance", case, O('{"a": "नमस्ते"}'), ctx)
    assert bad.passed is False and bad.details["checks"] == {"json_schema": False, "language": True}
    ok = run_case("format_compliance", C(language="hi"), O("आपका ऑर्डर"), ctx)
    assert ok.passed is True


def test_tool_definition_choice(tmp_path: Path) -> None:
    target = load_target(
        {
            "type": "fake",
            "responses": {
                "c1": {
                    "text": "",
                    "tool_calls": [{"name": "get_order", "args": {"order_id": "ORD-1"}}],
                }
            },
        },
        ROOT,
    )
    ctx = make_ctx(tmp_path, target=target)
    case = C(
        expect_tools=["get_order"],
        expect_trajectory=[{"name": "get_order", "args": {"order_id": "ord-1"}}],
    )
    assert run_case("tool_definition_choice", case, O(), ctx).passed is True
    assert (
        run_case("tool_definition_choice", C(expect_tools=["search_policy"]), O(), ctx).passed
        is False
    )


def test_fixed_vs_broken_and_win_rate(tmp_path: Path) -> None:
    from evalkit.reporting.store import StoredRun
    from evalkit.schema import ResultRow, RunInfo

    def brow(cid: str, ok: bool) -> ResultRow:
        return ResultRow(
            run_id="b",
            suite="quality",
            case_id=cid,
            repeat=0,
            metric="contains",
            value=float(ok),
            passed=ok,
            threshold=None,
            skipped=None,
            details={},
            tags=[],
            language=None,
            split="dev",
            must_pass=False,
            latency_s=None,
            tokens=None,
            cost_usd=None,
            trace_id=None,
        )

    base = StoredRun(
        tmp_path,
        RunInfo(run_id="b", started="x"),
        [brow("a", False), brow("b", True)],
        observations=[{"case_id": "a", "repeat": 0, "observation": {"text": "old"}}],
    )
    rows = [
        case_run(C(id="a"), ["new"], [mr("contains", True)]),
        case_run(C(id="b"), ["x"], [mr("contains", False)]),
    ]
    ctx = make_ctx(tmp_path, baseline=base)
    r = run_dataset("fixed_vs_broken", rows, ctx)
    assert r.details["fixed"] == ["a"] and r.details["broken"] == ["b"] and r.value == 1.0
    w = run_dataset("pairwise_win_rate", rows, ctx)
    assert w.value == 0.5 and w.details["n"] == 1


# ------------------------------------------------------------------ robustness


def _group(
    gid: str,
    transform: str,
    text: str,
    relation: str = "same_decision",
    results: list[MetricResult] | None = None,
) -> object:
    case = C(
        id=f"{gid}~{transform}", extra={"group": gid, "transform": transform, "relation": relation}
    )
    return case_run(case, [text], results or [])


def test_metamorphic_metrics(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, {"decision": ["approved", "denied"], "min_invariance": 0.9})
    rows = [
        _group("g", "original", "Refund approved"),
        _group("g", "typo", "refund approved!"),
        _group("g", "paraphrase", "It is denied"),
        _group("h", "original", "approved"),
        _group("h", "explicit", "denied", relation="different_decision"),
    ]
    inv = run_dataset("metamorphic_invariance", rows, ctx)
    assert inv.value == pytest.approx(0.5) and inv.passed is False
    assert inv.details["failures"][0]["variant"] == "g~paraphrase"
    assert run_dataset("metamorphic_change", rows, ctx).value == 1.0
    assert run_dataset("rephrase_consistency", rows, ctx).value == 0.0


def test_perturbation_drop(tmp_path: Path) -> None:
    rows = [
        _group("g", "original", "x", results=[mr("contains", True)]),
        _group("g", "typo", "x", results=[mr("contains", False)]),
        _group("g", "lowercase", "x", results=[mr("contains", True)]),
    ]
    r = run_dataset("perturbation_drop", rows, make_ctx(tmp_path, {"base_metrics": ["contains"]}))
    assert r.value == pytest.approx(0.5) and r.passed is False


# ------------------------------------------------------------------ classic


def test_unit_fake_llm_runs_pytest(tmp_path: Path) -> None:
    tests = tmp_path / "t"
    tests.mkdir()
    (tests / "test_x.py").write_text(
        "def test_ok():\n    assert 1\n\ndef test_bad():\n    assert 0\n", encoding="utf-8"
    )
    ctx = make_ctx(tmp_path, {"unit_tests": "t"}, suite="classic")
    r = run_dataset("unit_fake_llm", [], ctx)
    assert r.value == 0.5 and r.details["failed"] == ["t.test_x::test_bad"]


def test_contract_and_record_replay(tmp_path: Path) -> None:
    schemas = {"get_order": {"type": "object"}, "bad": {"type": 12}}
    ctx = make_ctx(
        tmp_path,
        {"response_schema": {"type": "object", "required": ["text"]}, "tool_schemas": schemas},
        suite="classic",
    )
    rows = [case_run(C(id="a", input="hi"), ["hi"])]
    r = run_dataset("contract", rows, ctx)
    assert r.passed is False and any("bad" in p for p in r.details["problems"])
    rr = run_dataset("record_replay", rows, ctx)
    assert rr.passed is True and rr.details["live_calls"] == 0
    assert run_dataset("api", rows, ctx).skipped == "n/a: target is not an http target"
    ui = run_dataset("ui_streaming", rows, ctx)
    assert ui.skipped in ("playwright not installed", "n/a: no tests configured")


def test_crosscutting(tmp_path: Path) -> None:
    rows = [case_run(C(id="a", tags=["x"]), ["t", "t"], [mr("m", True), mr("m", False)])]
    ctx = make_ctx(tmp_path)
    assert run_dataset("pass_rate_ci", rows, ctx).value == 0.5
    assert run_dataset("flaky_rate", rows, ctx).details["flaky"] == ["a"]
    assert run_dataset("bootstrap_ci", rows, ctx).value == 0.5
    assert run_dataset("slice_report", rows, ctx).details["by_tag"] == {"x": 0.5}
    assert run_dataset("judge_agreement", rows, ctx).skipped == "n/a: no judges used"
    assert run_dataset("sample_size", rows, ctx).value == 385.0


def test_tool_call_model() -> None:
    assert ToolCall(name="x").args == {}
