"""Reference-overlap metrics (Plan.md §9.1). Heavy libraries are lazy (`[textmetrics]`).

These are informational unless `params.<metric>.min_score` is set. They miss negation
("refund" vs "no refund"), which the report states (ch. 6.2).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from evalkit.metrics.common import result, scored
from evalkit.registry import metric
from evalkit.schema import Case, MetricResult, Observation
from evalkit.suites.context import MetricContext
from evalkit.util import optional_import, tokens

SIM_SUITES = ("quality", "migration", "finetune", "multilingual")
NEGATION_NOTE = "overlap metrics miss negation; pair them with a judge or contains checks"
_MODELS: dict[str, Any] = {}


def _maybe_threshold(value: float, ctx: MetricContext, **details: Any) -> MetricResult:
    minimum = ctx.p("min_score")
    details["note"] = NEGATION_NOTE
    if minimum is None:
        return result(value, None, **details)
    return scored(value, float(minimum), **details)


def token_f1_score(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


@metric("token_f1", suites=SIM_SUITES, kind="deterministic", needs=("reference",))
def token_f1(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """SQuAD-style token overlap F1 against the reference."""
    return _maybe_threshold(token_f1_score(obs.text, str(case.reference)), ctx)


@metric(
    "bleu",
    requires_extra="textmetrics",
    suites=SIM_SUITES,
    kind="deterministic",
    needs=("reference", "pkg:sacrebleu"),
)
def bleu(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """sacrebleu sentence BLEU / 100."""
    sacrebleu = optional_import("sacrebleu")
    score = sacrebleu.sentence_bleu(obs.text, [str(case.reference)]).score / 100.0
    return _maybe_threshold(float(score), ctx)


@metric(
    "rouge",
    requires_extra="textmetrics",
    suites=SIM_SUITES,
    kind="deterministic",
    needs=("reference", "pkg:rouge_score=rouge-score"),
)
def rouge(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """ROUGE-L F-measure (rouge-score)."""
    rs = optional_import("rouge_score.rouge_scorer", "rouge-score")
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(str(case.reference), obs.text)["rougeL"].fmeasure
    return _maybe_threshold(float(score), ctx)


@metric(
    "meteor",
    requires_extra="textmetrics",
    suites=SIM_SUITES,
    kind="deterministic",
    needs=("reference", "pkg:nltk"),
)
def meteor(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """NLTK METEOR (needs the wordnet corpus)."""
    mod = optional_import("nltk.translate.meteor_score", "nltk")
    score = mod.meteor_score([tokens(case.reference)], tokens(obs.text))
    return _maybe_threshold(float(score), ctx)


@metric(
    "bertscore",
    requires_extra="textmetrics",
    suites=SIM_SUITES,
    kind="deterministic",
    needs=("reference", "pkg:bert_score=bert-score"),
)
def bertscore(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """BERTScore F1 (bert-score, default English model)."""
    bs = optional_import("bert_score", "bert-score")
    _, _, f1 = bs.score([obs.text], [str(case.reference)], lang="en", verbose=False)
    return _maybe_threshold(float(f1[0]), ctx)


@metric(
    "cosine_similarity",
    requires_extra="textmetrics",
    suites=SIM_SUITES,
    kind="deterministic",
    needs=("reference", "pkg:sentence_transformers=sentence-transformers"),
)
def cosine_similarity(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Cosine similarity of all-MiniLM-L6-v2 embeddings."""
    st = optional_import("sentence_transformers", "sentence-transformers")
    model = _MODELS.get("minilm")
    if model is None:
        model = _MODELS["minilm"] = st.SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode([obs.text, str(case.reference)], normalize_embeddings=True)
    return _maybe_threshold(float((emb[0] * emb[1]).sum()), ctx)
