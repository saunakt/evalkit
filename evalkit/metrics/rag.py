"""RAG ingestion, freshness, access and generation metrics (Plan.md §9.5)."""

from __future__ import annotations

import re
from typing import Any

from evalkit.judges.runner import METRIC_JUDGES, inline_judge
from evalkit.metrics.common import (
    boolean,
    case_sections,
    contains_all,
    contains_any,
    context_text,
    from_verdict,
    na,
    result,
)
from evalkit.metrics.retrieval import recall_at_k_value, top_k
from evalkit.registry import metric
from evalkit.schema import Case, CaseRun, Chunk, MetricResult, Observation
from evalkit.suites.context import MetricContext
from evalkit.util import jaccard, mean, normalize_text, percentile, tokens

RAG = ("rag",)
NOISE = [
    "The cafeteria menu changes every Monday and includes vegetarian options.",
    "Our office plants are watered twice a week by the facilities team.",
    "The 2019 annual picnic was held at a lakeside resort near Pune.",
]
inline_judge(
    "context_relevance",
    "PASS if this retrieved chunk is relevant to answering the user's question. FAIL otherwise.",
    metric="context_precision",
)
inline_judge(
    "answer_correctness",
    "PASS if the answer is factually consistent with the "
    "reference answer and does not contradict it.",
    style="reference",
    metric="answer_correctness",
)
inline_judge(
    "citation_support",
    "PASS if the cited chunk supports the sentence that cites it.",
    metric="citation_accuracy",
)
inline_judge(
    "unanswerable",
    "PASS if the reply says it cannot find or does not know the "
    "answer and invents nothing. FAIL if it makes up an answer.",
    metric="unanswerable_handling",
)
METRIC_JUDGES.update(
    {
        "faithfulness": "faithfulness",
        "context_recall": "faithfulness",
        "answer_relevancy": "answer_relevancy",
    }
)


def ingested(ctx: MetricContext) -> list[Chunk]:
    """Chunks from target.ingest(params.docs_dir), cached per suite run."""
    memo = ctx.suite.memo
    if "chunks" not in memo:
        raw = ctx.target.ingest(str(ctx.p("docs_dir", "docs")))
        memo["chunks"] = [c if isinstance(c, Chunk) else Chunk.model_validate(c) for c in raw]
    chunks: list[Chunk] = memo["chunks"]
    return chunks


# --------------------------------------------------------------------------- ingestion


@metric("parse_fidelity", suites=RAG, kind="deterministic", needs=("extra.expected_snippets",))
def parse_fidelity(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Every extra.expected_snippets string appears in some chunk."""
    texts = [c.text for c in obs.retrieved]
    snippets = list(case.extra["expected_snippets"])
    missing = [
        s for s in snippets if not any(normalize_text(s) in normalize_text(t) for t in texts)
    ]
    return result((len(snippets) - len(missing)) / len(snippets), not missing, missing=missing)


@metric("chunk_integrity", suites=RAG, kind="deterministic", needs=("extra.keep_together",))
def chunk_integrity(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Every extra.keep_together string appears whole inside one chunk."""
    items = list(case.extra["keep_together"])
    split = [
        s
        for s in items
        if not any(normalize_text(s) in normalize_text(c.text) for c in obs.retrieved)
    ]
    return result((len(items) - len(split)) / len(items), not split, split=split)


@metric(
    "metadata_completeness",
    suites=RAG,
    kind="deterministic",
    scope="dataset",
    needs=("hook:ingest",),
)
def metadata_completeness(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Share of chunks carrying all params.required_metadata keys."""
    keys = list(ctx.p("required_metadata", ["source"]))
    chunks = ingested(ctx)
    if not chunks:
        return na("ingest returned no chunks")
    bad = [c.id for c in chunks if any(c.metadata.get(k) in (None, "") for k in keys)]
    value = 1 - len(bad) / len(chunks)
    return result(value, not bad, 1.0, incomplete=bad, required=keys)


@metric(
    "duplicate_rate",
    suites=RAG,
    kind="deterministic",
    scope="dataset",
    needs=("hook:ingest",),
    direction="lower",
)
def duplicate_rate(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Share of chunks with Jaccard ≥ 0.9 (normalised tokens) to another chunk."""
    chunks = ingested(ctx)
    sets = [set(tokens(c.text)) for c in chunks]
    dup = {
        chunks[i].id
        for i in range(len(sets))
        for j in range(len(sets))
        if i != j and jaccard(sets[i], sets[j]) >= 0.9
    }
    value = len(dup) / len(chunks) if chunks else 0.0
    limit = float(ctx.p("max_duplicate_rate", 0.05))
    return result(value, value <= limit, limit, duplicates=sorted(dup))


@metric(
    "index_stats",
    suites=RAG,
    kind="statistical",
    scope="dataset",
    needs=("hook:ingest",),
    unit="count",
)
def index_stats(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Informational: chunk count, mean / p95 chunk length (words), docs count."""
    chunks = ingested(ctx)
    lengths = [len(c.text.split()) for c in chunks]
    docs = {c.metadata.get("source", c.id) for c in chunks}
    return result(
        float(len(chunks)),
        None,
        chunks=len(chunks),
        docs=len(docs),
        mean_words=mean([float(x) for x in lengths]),
        p95_words=percentile([float(x) for x in lengths], 95),
    )


# --------------------------------------------------------------------------- freshness / access


@metric("freshness", suites=RAG, kind="deterministic", needs=("extra.freshness", "hook:update_doc"))
def freshness(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """update_doc(id, new_text) → ask again → new fact present, old fact gone (text null =
    deletion: the deleted fact must be gone)."""
    spec = dict(case.extra["freshness"])
    ctx.suite.reset(ctx.seed)
    ctx.target.update_doc(str(spec["doc_id"]), spec.get("text"))
    try:
        fresh = ctx.suite.call(case)
    finally:
        ctx.suite.reset(ctx.seed)
    missing = contains_all(fresh.text, [spec["new_fact"]]) if spec.get("new_fact") else []
    stale = contains_any(fresh.text, [spec["old_fact"]]) if spec.get("old_fact") else []
    ok = not missing and not stale
    return boolean(ok, answer=fresh.text[:300], missing_new=missing, stale_present=stale)


@metric("access_control", suites=RAG + ("security",), kind="deterministic", needs=("extra.role",))
def access_control(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Call with ctx.role; no retrieved chunk's metadata.access exceeds the role."""
    role = str(case.extra["role"])
    allowed = set(dict(ctx.p("role_levels", {})).get(role, ["public"]))
    res = ctx.suite.recall(case, role=role)
    leaks = [c.id for c in res.retrieved if c.metadata.get("access", "public") not in allowed]
    return boolean(not leaks, role=role, leaked=leaks, retrieved=[c.id for c in res.retrieved])


# --------------------------------------------------------------------------- generation


@metric("context_precision", suites=RAG, kind="judge", needs=("obs.retrieved",))
def context_precision(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Judge relevance of each retrieved chunk; average precision over relevant positions."""
    hits, precisions, flags = 0, [], []
    for rank, chunk in enumerate(obs.retrieved[: int(ctx.p("k", 5))], start=1):
        v = ctx.judge.binary(
            "context_relevance",
            sections={"User question": case.text_input(), "Chunk": chunk.text},
            case_id=case.id,
        )
        flags.append(bool(v.passed))
        if v.passed:
            hits += 1
            precisions.append(hits / rank)
    value = mean(precisions) or 0.0
    return result(value, value >= float(ctx.p("min_context_precision", 0.5)), relevant=flags)


@metric("context_recall", suites=RAG, kind="judge", needs=("reference",))
def context_recall(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Claims of the reference that the retrieved context supports (claims judge)."""
    v = ctx.judge.claims(
        "faithfulness",
        text=str(case.reference),
        context=context_text(obs),
        case_id=case.id,
        threshold=float(ctx.p("min_context_recall", 0.8)),
    )
    return from_verdict(v)


@metric(
    "faithfulness",
    suites=RAG + ("responsible_ai",),
    kind="judge",
    engines=("native", "ragas", "deepeval"),
)
def faithfulness(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Claims of the output supported by the retrieved context (claims judge)."""
    if not obs.retrieved and not obs.tool_calls:
        return na("no retrieved context")
    v = ctx.judge.claims(
        "faithfulness",
        text=obs.text,
        context=context_text(obs),
        case_id=case.id,
        threshold=float(ctx.p("min_faithfulness", 0.8)),
    )
    return from_verdict(v)


@metric(
    "answer_relevancy",
    suites=RAG + ("conversation", "multilingual"),
    kind="judge",
    engines=("native", "ragas", "deepeval"),
)
def answer_relevancy(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Judge: the answer addresses the question."""
    return from_verdict(
        ctx.judge.binary("answer_relevancy", sections=case_sections(case, obs), case_id=case.id)
    )


@metric("answer_correctness", suites=RAG, kind="judge", needs=("reference",))
def answer_correctness(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Reference judge: consistent with the reference answer."""
    secs = {**case_sections(case, obs), "Reference answer": case.reference}
    return from_verdict(ctx.judge.binary("answer_correctness", sections=secs, case_id=case.id))


def cited_sentences(text: str, pattern: str) -> list[tuple[str, str]]:
    out = []
    for sentence in re.split(r"(?<=[.!?\]])\s+", text):
        for cid in re.findall(pattern, sentence):
            out.append((cid, sentence))
    return out


@metric("citation_accuracy", suites=RAG, kind="judge", needs=("obs.retrieved",))
def citation_accuracy(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Each [cited-id] is a retrieved chunk and the judge confirms it supports its sentence."""
    pattern = str(ctx.p("citation_regex", r"\[([\w\-#.]+)\]"))
    cites = cited_sentences(obs.text, pattern)
    if not cites:
        return boolean(False, reason="no citations found")
    by_id = {c.id: c for c in obs.retrieved}
    checks = []
    for cid, sentence in cites:
        chunk = by_id.get(cid)
        ok = chunk is not None and bool(
            ctx.judge.binary(
                "citation_support",
                sections={"Sentence": sentence, "Cited chunk": chunk.text},
                case_id=case.id,
            ).passed
        )
        checks.append({"id": cid, "retrieved": chunk is not None, "supported": ok})
    good = sum(1 for c in checks if c["supported"])
    return result(good / len(checks), good == len(checks), citations=checks)


@metric("noise_sensitivity", suites=RAG, kind="deterministic", needs=("must_include",))
def noise_sensitivity(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Re-call with irrelevant chunks injected (ctx.extra_context); correctness must not drop."""
    noise = ctx.p("noise_chunks") or NOISE
    extra = [
        Chunk(id=f"noise-{i}", text=t, metadata={"access": "public"})
        for i, t in enumerate(noise if isinstance(noise, list) else NOISE)
    ]
    noisy = ctx.suite.recall(case, extra_context=extra)
    before = not contains_all(obs.text, case.must_include)
    after = not contains_all(noisy.text, case.must_include)
    ok = after or not before
    return boolean(ok, clean_correct=before, noisy_correct=after, noisy_answer=noisy.text[:300])


@metric(
    "unanswerable_handling",
    suites=RAG + ("responsible_ai",),
    kind="judge",
    needs=("tag:unanswerable",),
)
def unanswerable_handling(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Judge: says it cannot find the answer and invents nothing."""
    return from_verdict(
        ctx.judge.binary("unanswerable", sections=case_sections(case, obs), case_id=case.id)
    )


@metric(
    "config_compare",
    suites=RAG,
    kind="statistical",
    scope="dataset",
    needs=("params.retriever_configs",),
)
def config_compare(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """For each params.retriever_configs entry re-run retrieval; recall@k per config."""
    configs: dict[str, Any] = dict(ctx.p("retriever_configs"))
    table: dict[str, dict[str, float | None]] = {}
    for name, params in sorted(configs.items()):
        recalls, rr = [], []
        for cr in rows:
            if not (cr.case.relevant_ids or cr.case.relevance):
                continue
            res = ctx.suite.recall(cr.case, params=dict(params or {}))
            ids = top_k(res, ctx)
            recalls.append(recall_at_k_value(cr.case, ids))
            rel = set(cr.case.relevant_ids)
            rr.append(next((1 / i for i, c in enumerate(ids, 1) if c in rel), 0.0))
        table[name] = {"recall_at_k": mean(recalls), "mrr": mean(rr), "n": float(len(recalls))}
    best = max(table, key=lambda n: table[n]["recall_at_k"] or 0.0) if table else None
    return result(None, None, configs=table, best=best)
