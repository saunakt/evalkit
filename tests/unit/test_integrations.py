"""M11: catalog, evidence, Label Studio, Langfuse (stubbed), judge pass-through, generate."""

from __future__ import annotations

import json
import socket
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from evalkit.catalog import stale_files
from evalkit.config import EvalkitConfig, ProviderConfig, RoleModel
from evalkit.engines.labelstudio import agreement, export_tasks, import_labels
from evalkit.registry import METRICS, SUITES, load_all
from evalkit.schema import Case, Chunk, Observation
from tests.conftest import ROOT, FakeServer, Reply, chat_reply
from tests.metric_helpers import make_ctx

PLAN_METRICS = [
    "exact_match",
    "token_f1",
    "contains",
    "not_contains",
    "regex",
    "json_schema",
    "numeric_tolerance",
    "length",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "bleu",
    "rouge",
    "meteor",
    "bertscore",
    "cosine_similarity",
    "completeness_checklist",
    "judge_binary",
    "judge_rubric",
    "judge_reference",
    "judge_pairwise",
    "g_eval",
    "tone",
    "readability",
    "self_consistency",
    "instruction_positive",
    "instruction_negative",
    "out_of_scope_redirect",
    "format_compliance",
    "instruction_conflict",
    "tool_definition_choice",
    "fixed_vs_broken",
    "pairwise_win_rate",
    "metamorphic_invariance",
    "metamorphic_change",
    "perturbation_drop",
    "rephrase_consistency",
    "knowledge_retention",
    "reference_resolution",
    "consistency",
    "role_adherence",
    "conversation_completeness",
    "turn_relevancy",
    "recovery",
    "turns_to_resolution",
    "context_overflow",
    "long_term_memory",
    "parse_fidelity",
    "chunk_integrity",
    "metadata_completeness",
    "duplicate_rate",
    "index_stats",
    "freshness",
    "access_control",
    "recall_at_k",
    "hit_rate_at_k",
    "precision_at_k",
    "mrr",
    "ndcg",
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "citation_accuracy",
    "noise_sensitivity",
    "unanswerable_handling",
    "config_compare",
    "tool_unit",
    "single_step_choice",
    "tool_selection",
    "argument_accuracy",
    "trajectory_exact",
    "trajectory_in_order",
    "trajectory_any_order",
    "trajectory_subset",
    "trajectory",
    "forbidden_tools",
    "step_efficiency",
    "end_state",
    "task_success",
    "plan_quality",
    "error_recovery",
    "policy_compliance",
    "unsafe_action_rate",
    "confirmation_compliance",
    "pass_at_k",
    "pass_hat_k",
    "cost_per_success",
    "time_per_task",
    "simulated_user_success",
    "mcp_contract",
    "handoff_accuracy",
    "hitl_approval",
    "browser_task_success",
    "resume_after_interrupt",
    "resolution_rate",
    "regression_rate",
    "code_quality",
    "diff_discipline",
    "rules_adherence",
    "security_findings",
    "command_safety",
    "cost_per_resolved",
    "seeded_bug_recall",
    "comment_precision",
    "severity_accuracy",
    "actionability",
    "review_consistency",
    "test_coverage",
    "mutation_score",
    "trigger_rate",
    "false_trigger_rate",
    "execution_steps",
    "output_quality",
    "with_without_delta",
    "skill_conflict",
    "third_party_scan",
    "attack_success_rate",
    "direct_injection",
    "indirect_injection",
    "jailbreak",
    "prompt_leakage",
    "sensitive_disclosure",
    "excessive_agency",
    "output_handling",
    "poisoning",
    "cross_tenant_retrieval",
    "unbounded_consumption",
    "garak_scan",
    "unanswerable",
    "false_premise",
    "fake_entity",
    "citation_verification",
    "number_grounding",
    "bias_gap",
    "tone_gap",
    "toxicity_rate",
    "harmful_refusal_rate",
    "over_refusal_rate",
    "pii_leakage_rate",
    "guard_precision",
    "guard_recall",
    "guard_f1",
    "false_block_rate",
    "added_latency",
    "fail_mode",
    "bypass_rate",
    "requirement_coverage",
    "ai_disclosure",
    "data_rights",
    "evidence_completeness",
    "ttft",
    "tokens_per_second",
    "latency_p50",
    "latency_p95",
    "latency_p99",
    "component_latency",
    "throughput",
    "error_rate",
    "cost_per_request",
    "cache_hit_rate",
    "retry_backoff",
    "timeout_handling",
    "fallback_quality",
    "graceful_degradation",
    "stream_interruption",
    "slice_delta",
    "mcnemar",
    "latency_delta",
    "cost_delta",
    "refusal_delta",
    "wer",
    "entity_accuracy",
    "voice_latency",
    "barge_in",
    "turn_detection",
    "tts_quality",
    "field_accuracy",
    "image_classification_f1",
    "refuses_unreadable",
    "visual_injection",
    "visual_hallucination",
    "image_token_cost",
    "prompt_adherence",
    "media_safety",
    "media_bias",
    "provenance",
    "gain_vs_prompted_base",
    "heldout_generalisation",
    "train_test_overlap",
    "forgetting_regression",
    "safety_regression",
    "memorisation_canary",
    "language_match",
    "script_match",
    "correctness_parity",
    "retrieval_parity",
    "entity_preservation",
    "safety_parity",
    "guardrail_indic",
    "cost_per_language",
    "latency_per_language",
    "native_tone",
    "unit_fake_llm",
    "record_replay",
    "contract",
    "api",
    "ui_streaming",
    "pass_rate_ci",
    "flaky_rate",
    "bootstrap_ci",
    "slice_report",
    "judge_agreement",
    "sample_size",
]
PLAN_SUITES = [
    "agent",
    "classic",
    "coding_agent",
    "compliance",
    "conversation",
    "finetune",
    "guardrails",
    "migration",
    "multilingual",
    "multimodal",
    "performance",
    "prompt",
    "quality",
    "rag",
    "responsible_ai",
    "robustness",
    "security",
    "skills",
]


def test_registry_has_every_plan_metric_and_suite() -> None:
    load_all()
    assert len(PLAN_METRICS) == len(set(PLAN_METRICS)) == 217
    assert sorted(m for m in METRICS.names() if not m.startswith("_")) == sorted(PLAN_METRICS)
    assert SUITES.names() == PLAN_SUITES


def test_catalog_is_up_to_date() -> None:
    assert stale_files(ROOT) == [], "run `evalctl catalog`"


# ------------------------------------------------------------------ Label Studio


def test_label_studio_roundtrip(tmp_path: Path) -> None:
    paths = export_tasks([{"id": "a", "input": "q", "output": "o"}], tmp_path, "Be polite")
    tasks = json.loads(paths["tasks"].read_text(encoding="utf-8"))
    assert tasks[0]["data"]["criterion"] == "Be polite" and "PASS" in paths["config"].read_text(
        encoding="utf-8"
    )

    def ann(who: int, label: str) -> dict[str, Any]:
        return {"completed_by": who, "result": [{"value": {"choices": [label]}}]}

    export = [
        {
            "id": 1,
            "data": {"item_id": "a", "output": "o"},
            "annotations": [ann(1, "PASS"), ann(2, "PASS")],
        },
        {
            "id": 2,
            "data": {"item_id": "b", "output": "o"},
            "annotations": [ann(1, "FAIL"), ann(2, "PASS")],
        },
    ]
    (tmp_path / "export.json").write_text(json.dumps(export), encoding="utf-8")
    stats = import_labels(tmp_path / "export.json", tmp_path / "labels.jsonl")
    assert stats["rows"] == 2 and stats["annotators"] == 2 and "cohen_kappa" in stats
    rows = [
        json.loads(line)
        for line in (tmp_path / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0] == {"id": "a", "input": "", "output": "o", "label": "PASS", "split": "dev"}
    assert agreement({"x": {1: "A"}}) == {"annotators": 1}


# ------------------------------------------------------------------ Langfuse (stub SDK)


class _StubLangfuse:
    def __init__(self, **kw: Any) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        trace = types.SimpleNamespace(id="t-1234567890abcdef", input="Mail me at a@b.in")
        self.api = types.SimpleNamespace(
            trace=types.SimpleNamespace(list=lambda **k: types.SimpleNamespace(data=[trace]))
        )

    def __getattr__(self, name: str) -> Any:
        return lambda **kw: self.calls.append((name, kw))


def test_langfuse_push_and_pull(tmp_path: Path) -> None:
    from evalkit.engines.langfuse_engine import pull_traces, push_run
    from evalkit.reporting.store import ResultStore, load_run
    from evalkit.schema import ResultRow, RunInfo

    row = ResultRow(
        run_id="r",
        suite="quality",
        case_id="c1",
        repeat=0,
        metric="contains",
        value=1.0,
        passed=True,
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
        trace_id="t1",
    )
    st = ResultStore(tmp_path / "r")
    st.write_info(RunInfo(run_id="r", started="x", suites={"quality": ["contains"]}))
    st.write_rows([row])
    ctx = make_ctx(tmp_path)
    lf = _StubLangfuse()
    counts = push_run(ctx.suite.project, load_run(tmp_path / "r"), lf)
    assert counts == {"datasets": 1, "items": 1, "scores": 1}
    assert [c[0] for c in lf.calls] == [
        "create_dataset",
        "create_dataset_item",
        "create_score",
        "flush",
    ]
    out = pull_traces(ctx.suite.project, "quality", 10, None, lf)
    draft = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert draft["input"] == "Mail me at <EMAIL>" and draft["source"].startswith("trace:")


# ------------------------------------------------------------------ judge pass-through


class _Verdicts(BaseModel):
    verdicts: list[str]


def _stub_deepeval() -> dict[str, types.ModuleType]:
    """Minimal stand-in for deepeval: a metric that asks the model for structured output."""
    root, models, metrics, tc = (
        types.ModuleType(n)
        for n in ("deepeval", "deepeval.models", "deepeval.metrics", "deepeval.test_case")
    )

    class DeepEvalBaseLLM:
        def __init__(self, model: str | None = None) -> None:
            self.name = model
            self.model = self.load_model()  # type: ignore[attr-defined]

    class FaithfulnessMetric:
        def __init__(self, threshold: float, model: Any, async_mode: bool) -> None:
            self.model, self.threshold = model, threshold

        def measure(self, test_case: Any) -> None:
            out = self.model.generate(f"Claims in: {test_case.actual_output}", _Verdicts)
            self.score = sum(v == "yes" for v in out.verdicts) / len(out.verdicts)
            self.reason = "stub"

    models.DeepEvalBaseLLM = DeepEvalBaseLLM  # type: ignore[attr-defined]
    metrics.FaithfulnessMetric = FaithfulnessMetric  # type: ignore[attr-defined]
    tc.LLMTestCase = lambda **kw: types.SimpleNamespace(**kw)  # type: ignore[attr-defined]
    return {
        "deepeval": root,
        "deepeval.models": models,
        "deepeval.metrics": metrics,
        "deepeval.test_case": tc,
    }


def test_deepeval_calls_only_the_judge_endpoint(
    tmp_path: Path, http_server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evalkit.engines import deepeval_engine
    from evalkit.llm.roles import LLMRuntime

    for name, mod in _stub_deepeval().items():
        monkeypatch.setitem(sys.modules, name, mod)
    http_server.default = Reply(
        body=chat_reply(
            '{"verdicts": ["yes", "no"]}', {"prompt_tokens": 50, "completion_tokens": 5}
        )
    )
    port = int(http_server.url.rsplit(":", 1)[1])
    real_connect = socket.create_connection

    def guarded(address: Any, *a: Any, **kw: Any) -> Any:
        assert tuple(address)[:2] in (("127.0.0.1", port),), f"unexpected network: {address}"
        return real_connect(address, *a, **kw)

    monkeypatch.setattr(socket, "create_connection", guarded)
    ctx = make_ctx(tmp_path)
    cfg = EvalkitConfig(
        providers={
            "judgehost": ProviderConfig(
                type="openai_compatible", base_url=http_server.url, free=False
            )
        },
        models={"judge": RoleModel(provider="judgehost", model="judge-x")},
    )
    ctx.suite.judge.runtime = LLMRuntime(cfg, tmp_path)
    case = Case(id="c", input="q")
    obs = Observation(text="a. b.", retrieved=[Chunk(id="x", text="ctx")])
    res = deepeval_engine.faithfulness(case, obs, ctx)
    assert res.value == 0.5
    assert all(r["path"] == "/chat/completions" for r in http_server.requests)
    led = ctx.suite.judge.runtime.ledger.summary()["judge"]
    assert led["paid_calls"] == 1  # counted under the judge role, as paid


def test_real_deepeval_if_installed() -> None:
    pytest.importorskip("deepeval")


def test_engines_skip_cleanly_when_missing(tmp_path: Path) -> None:
    from evalkit.execution import run_case_metric

    ctx = make_ctx(tmp_path)
    for eng in ("ragas", "deepeval"):
        if eng in sys.modules:
            continue
        res = run_case_metric(
            METRICS.get("faithfulness"),
            eng,
            Case(id="c", input="q"),
            Observation(text="x", retrieved=[Chunk(id="a", text="b")]),
            ctx,
        )
        assert res.skipped == f"{eng} not installed"


# ------------------------------------------------------------------ generate


def test_generate_drafts_into_drafts_folder(tmp_path: Path) -> None:
    from evalkit.config import Overrides, load_project
    from evalkit.generate import generate_drafts

    project = load_project(ROOT / "evalkit.yaml", None, Overrides(provider="fake"))
    project.root = tmp_path
    out = generate_drafts(project, "rag", ROOT / "examples/shopease/docs", 3)
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert out.parent.name == "drafts" and len(rows) == 3
    assert all("draft" in r["tags"] and r["relevant_ids"] for r in rows)


def test_evidence_pack_has_every_item(tmp_path: Path) -> None:
    from evalkit.config import Overrides, load_project
    from evalkit.reporting.evidence import EVIDENCE_ITEMS, build_evidence
    from evalkit.reporting.store import ResultStore, load_run
    from evalkit.schema import RunInfo

    project = load_project(ROOT / "evalkit.yaml", None, Overrides(provider="fake"))
    st = ResultStore(tmp_path / "run-x")
    st.write_info(RunInfo(run_id="run-x", started="x", suites={"quality": ["contains"]}))
    st.write_rows([])
    project.root = tmp_path
    for d in ("datasets/cards", "judges/calibration"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "datasets/cards/quality.yaml").write_text("name: quality\n", encoding="utf-8")
    report_dir = tmp_path / "reports" / "run-x"
    (report_dir / "security").mkdir(parents=True)
    (report_dir / "security" / "report.md").write_text("# security\n", encoding="utf-8")
    out = build_evidence(project, load_run(tmp_path / "run-x"), report_dir)
    assert all((out / item).exists() for item in EVIDENCE_ITEMS)
    assert "# security" in (out / "red_team_report.md").read_text(encoding="utf-8")
    assert "R1" in (out / "risk_register.md").read_text(encoding="utf-8")
