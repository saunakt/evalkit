"""Unit tests for retrieval, RAG and conversation metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalkit.schema import Case, Chunk, Observation
from evalkit.targets.base import load_target
from tests.metric_helpers import ROOT, case_run, make_ctx, run_case, run_dataset


def C(**kw: object) -> Case:
    return Case.model_validate({"id": "c1", **kw})


def chunks(*ids: str) -> list[Chunk]:
    return [
        Chunk(id=i, text=f"text of {i}", metadata={"source": f"{i}.md", "access": "public"})
        for i in ids
    ]


def shopease() -> object:
    return load_target(ROOT / "targets" / "shopease_agent.yaml", ROOT)


# ------------------------------------------------------------------ retrieval


def test_retrieval_metrics(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, {"k": 3})
    case = C(relevant_ids=["b", "z"])
    obs = Observation(retrieved=chunks("a", "b", "c", "z"))
    assert run_case("recall_at_k", case, obs, ctx).value == 0.5
    assert run_case("hit_rate_at_k", case, obs, ctx).passed is True
    assert run_case("precision_at_k", case, obs, ctx).value == pytest.approx(1 / 3)
    assert run_case("mrr", case, obs, ctx).value == 0.5
    graded = C(relevance={"a": 3, "b": 1})
    n = run_case("ndcg", graded, Observation(retrieved=chunks("a", "b")), ctx).value
    assert n == pytest.approx(1.0)
    rev = run_case("ndcg", graded, Observation(retrieved=chunks("b", "a")), ctx).value
    assert rev is not None and rev < 1.0
    assert run_case("recall_at_k", C(), obs, ctx).skipped == "n/a: needs relevant_ids or relevance"


# ------------------------------------------------------------------ ingestion (ShopEase docs)


def test_ingestion_metrics(tmp_path: Path) -> None:
    ctx = make_ctx(
        tmp_path,
        {"docs_dir": "examples/shopease/docs", "required_metadata": ["source", "access", "tenant"]},
        target=shopease(),
        suite="rag",
    )
    from evalkit.metrics.rag import ingested

    obs = Observation(retrieved=ingested(ctx))
    ok = C(extra={"expected_snippets": ["₹10,000"], "keep_together": ["5-7 business days"]})
    assert run_case("parse_fidelity", ok, obs, ctx).passed is True
    assert run_case("chunk_integrity", ok, obs, ctx).passed is True
    bad = C(extra={"expected_snippets": ["₹99,999"]})
    assert run_case("parse_fidelity", bad, obs, ctx).details["missing"] == ["₹99,999"]
    assert run_dataset("metadata_completeness", [], ctx).passed is True
    assert run_dataset("duplicate_rate", [], ctx).value == 0.0
    stats = run_dataset("index_stats", [], ctx)
    assert stats.details["chunks"] == 13 and stats.passed is None


def test_ingestion_needs_hook(tmp_path: Path) -> None:
    r = run_dataset("metadata_completeness", [], make_ctx(tmp_path))
    assert r.skipped == "n/a: ingest returned no chunks"


# ------------------------------------------------------------------ freshness / access / noise


def test_freshness_access_noise_on_shopease(tmp_path: Path) -> None:
    tgt = shopease()
    ctx = make_ctx(tmp_path, {"role_levels": {"customer": ["public"]}}, target=tgt, suite="rag")
    fresh = C(
        input="What is the COD limit?",
        extra={
            "freshness": {
                "doc_id": "cod",
                "text": "COD is available for orders up to ₹15,000.",
                "new_fact": "₹15,000",
                "old_fact": "₹10,000",
            }
        },
    )
    assert run_case("freshness", fresh, Observation(), ctx).passed is True
    gone = C(
        input="What is the COD limit?",
        extra={"freshness": {"doc_id": "cod", "text": None, "old_fact": "₹10,000"}},
    )
    assert run_case("freshness", gone, Observation(), ctx).passed is True
    access = C(input="How much goodwill credit can team leads approve?", extra={"role": "customer"})
    assert run_case("access_control", access, Observation(), ctx).passed is True
    agent_ctx = make_ctx(tmp_path, {"role_levels": {"agent": ["public"]}}, target=tgt, suite="rag")
    leak = run_case(
        "access_control", C(input=access.input, extra={"role": "agent"}), Observation(), agent_ctx
    )
    assert leak.passed is False and leak.details["leaked"] == ["escalation-internal"]
    noise = C(input="Is shipping free above 499?", must_include=["free"])
    clean = tgt.call(noise, ctx.suite.target_ctx(noise))
    assert run_case("noise_sensitivity", noise, clean, ctx).passed is True


# ------------------------------------------------------------------ generation judges


def test_generation_judges(tmp_path: Path) -> None:
    ctx = make_ctx(
        tmp_path,
        script={
            "judge_script": [
                {"judge": "faithfulness", "step": "verify", "supported": [True, False]},
                {"judge": "context_relevance", "match": "Chunk\\ntext of b", "verdict": "FAIL"},
            ]
        },
    )
    obs = Observation(
        text="Returns take 30 days. Shipping is free. [a]", retrieved=chunks("a", "b", "c")
    )
    case = C(input="q", reference="Returns take 30 days.")
    f = run_case("faithfulness", case, obs, ctx)
    assert f.value == 0.5 and f.details["unsupported"] == ["Shipping is free."]
    assert run_case("faithfulness", case, Observation(text="x"), ctx).skipped.startswith("n/a")
    cp = run_case("context_precision", case, obs, ctx)
    assert cp.details["relevant"] == [True, False, True]
    assert cp.value == pytest.approx((1 + 2 / 3) / 2)
    assert run_case("context_recall", case, obs, ctx).value == 1.0  # one reference claim
    assert run_case("answer_relevancy", case, obs, ctx).passed is True
    assert run_case("answer_correctness", case, obs, ctx).passed is True
    cite = run_case("citation_accuracy", case, obs, ctx)
    assert cite.passed is True and cite.details["citations"][0]["id"] == "a"
    wrong = run_case(
        "citation_accuracy", case, Observation(text="x [zz]", retrieved=chunks("a")), ctx
    )
    assert wrong.passed is False
    un = run_case("unanswerable_handling", C(tags=["unanswerable"]), obs, ctx)
    assert un.passed is True


def test_config_compare(tmp_path: Path) -> None:
    ctx = make_ctx(
        tmp_path,
        {"retriever_configs": {"k5": {"retriever_k": 5}, "k1": {"retriever_k": 1}}},
        target=shopease(),
        suite="rag",
    )
    rows = [
        case_run(C(id="r", input="Is shipping free above 499?", relevant_ids=["shipping", "cod"]))
    ]
    r = run_dataset("config_compare", rows, ctx)
    assert r.details["configs"]["k5"]["recall_at_k"] == 1.0
    assert r.details["configs"]["k1"]["recall_at_k"] == 0.5 and r.details["best"] == "k5"


# ------------------------------------------------------------------ conversation


def _turns_obs() -> Observation:
    return Observation(
        text="b",
        messages=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        raw={
            "turns": [
                {
                    "turn": 0,
                    "user": "a",
                    "reply": "b",
                    "passed": True,
                    "missing": [],
                    "forbidden": [],
                },
                {
                    "turn": 1,
                    "user": "c",
                    "reply": "d",
                    "passed": False,
                    "missing": ["x"],
                    "forbidden": [],
                },
            ]
        },
    )


def test_conversation_judges(tmp_path: Path) -> None:
    ctx = make_ctx(
        tmp_path, script={"judge_script": [{"judge": "conv_consistency", "verdict": "FAIL"}]}
    )
    obs = _turns_obs()
    for m in (
        "knowledge_retention",
        "reference_resolution",
        "role_adherence",
        "conversation_completeness",
        "recovery",
    ):
        assert run_case(m, C(), obs, ctx).passed is True, m
    assert run_case("consistency", C(), obs, ctx).passed is False
    assert run_case("consistency", C(), Observation(text="x"), ctx).skipped.startswith("n/a")
    tr = run_case("turn_relevancy", C(), obs, ctx)
    assert tr.value == 0.5 and tr.passed is False


def test_turns_to_resolution(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    case = C(hidden_goal="g", max_turns=4)
    done = Observation(raw={"simulated": True, "user_turns": 2, "done": True})
    assert run_case("turns_to_resolution", case, done, ctx).value == 2.0
    stuck = Observation(raw={"simulated": True, "user_turns": 4, "done": False})
    assert run_case("turns_to_resolution", case, stuck, ctx).passed is False


def test_context_overflow_and_memory_on_shopease(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, {"pad_turns": 10}, target=shopease(), suite="conversation")
    case = C(
        turns=[
            {"user": "My name is Priya."},
            {"user": "What's my name?", "expect": {"must_include": ["Priya"]}},
        ],
        extra={"memory": {"tell": "My name is Priya.", "ask": "What's my name?", "fact": "Priya"}},
    )
    assert run_case("context_overflow", case, Observation(), ctx).passed is True
    assert run_case("long_term_memory", case, Observation(), ctx).passed is True
