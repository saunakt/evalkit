"""Langfuse push / pull (extra `[observe]`, lazy import). Checked against langfuse 5.2.0:
`Langfuse(public_key, secret_key, host)`, `create_dataset(name=)`,
`create_dataset_item(dataset_name=, input=, expected_output=, metadata=, id=)`,
`create_score(name=, value=, trace_id=, comment=)`, `api.trace.list(...)`, `flush()`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from evalkit.config import Project
from evalkit.metrics.pii import detect
from evalkit.registry import engine
from evalkit.reporting.store import StoredRun
from evalkit.util import optional_import, write_jsonl


@engine("langfuse", requires_extra="observe")
class LangfuseEngine:
    """Engine marker: `evalctl push` / `evalctl pull-traces`."""

    package = "langfuse"


def client(project: Project, sdk: Any = None) -> Any:
    sdk = sdk or optional_import("langfuse")
    host = project.config.observe.langfuse_host or os.environ.get("LANGFUSE_HOST")
    return sdk.Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        host=host,
    )


def push_run(project: Project, stored: StoredRun, lf: Any = None) -> dict[str, int]:
    """One dataset per suite, one item per case, one score per case × metric."""
    lf = lf or client(project)
    counts = {"datasets": 0, "items": 0, "scores": 0}
    for suite in stored.suites():
        name = f"{project.config.project}/{suite}"
        lf.create_dataset(name=name)
        counts["datasets"] += 1
        seen: set[str] = set()
        for row in stored.suite_rows(suite):
            if row.case_id and row.case_id not in seen:
                seen.add(row.case_id)
                case = stored.cases.get(f"{suite}/{row.case_id}")
                lf.create_dataset_item(
                    dataset_name=name,
                    id=f"{suite}:{row.case_id}",
                    input=case.text_input() if case else row.case_id,
                    expected_output=case.reference if case else None,
                    metadata={"tags": row.tags, "run_id": stored.run_id},
                )
                counts["items"] += 1
            if row.value is not None and row.skipped is None:
                lf.create_score(
                    name=f"{suite}.{row.metric}",
                    value=float(row.value),
                    trace_id=row.trace_id,
                    comment=f"run {stored.run_id}",
                )
                counts["scores"] += 1
    lf.flush()
    return counts


def scrub(text: str) -> str:
    """Replace detected PII with typed placeholders before writing drafts."""
    for ent in detect(text)[0]:
        text = text.replace(ent["value"], f"<{ent['type']}>")
    return text


def pull_traces(
    project: Project, suite: str, limit: int, filter_expr: str | None, lf: Any = None
) -> Path:
    """Flagged/low-scored traces → PII-scrubbed draft cases in datasets/drafts/."""
    lf = lf or client(project)
    kwargs: dict[str, Any] = {"limit": limit}
    if filter_expr:
        kwargs["tags"] = [filter_expr]
    traces = getattr(lf.api.trace.list(**kwargs), "data", [])
    drafts = []
    for t in traces:
        text = str(getattr(t, "input", "") or "")
        drafts.append(
            {
                "id": f"trace-{str(getattr(t, 'id', ''))[:12]}",
                "input": scrub(text),
                "tags": ["draft", "from_trace"],
                "source": f"trace:{getattr(t, 'id', '')}",
                "suite": suite,
            }
        )
    out = project.path("datasets/drafts") / f"{suite}_traces.jsonl"
    write_jsonl(out, drafts)
    return out
