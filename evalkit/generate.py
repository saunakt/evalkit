"""`evalctl generate`: draft cases from documents on the generator role (never into golden)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from evalkit.config import Project
from evalkit.llm.roles import LLMRuntime, set_runtime
from evalkit.util import extract_json, write_jsonl

PROMPT = (
    "Write ONE question a customer might ask that the text below answers, plus the "
    'answer, as JSON {"question": str, "answer": str}. Use only the text.'
)


def doc_chunks(docs: Path) -> list[tuple[str, str]]:
    out = []
    for path in sorted(p for p in docs.rglob("*") if p.suffix in (".md", ".txt")):
        text = path.read_text(encoding="utf-8")
        body = text.split("---", 2)[2] if text.startswith("---") else text
        for i, para in enumerate(p.strip() for p in re.split(r"\n\s*\n", body)):
            if len(para.split()) >= 8:
                out.append((f"{path.stem}" + (f"#{i}" if i else ""), para))
    return out


def draft_from_chunk(client: Any, doc_id: str, text: str, n: int) -> dict[str, Any]:
    resp = client.complete(
        system=PROMPT,
        messages=[{"role": "user", "content": text}],
        response_format="json",
        meta={"step": "generate", "doc": doc_id},
    )
    data = extract_json(resp.text)
    if not isinstance(data, dict) or "question" not in data:
        first = re.split(r"(?<=[.!?])\s+", text.strip())[0]
        data = {"question": f"Can you tell me about this: {first[:120]}", "answer": first}
    return {
        "id": f"draft-{doc_id.replace('#', '-')}-{n:02d}",
        "input": str(data["question"]),
        "reference": str(data.get("answer", "")),
        "relevant_ids": [doc_id.split("#")[0]],
        "tags": ["draft", "generated"],
        "source": f"generated:{doc_id}",
    }


def generate_drafts(project: Project, suite: str, docs: Path, n: int) -> Path:
    """Native generator on the `generator` role (local, free); writes datasets/drafts/."""
    rt = LLMRuntime(project.config, project.root)
    set_runtime(rt)
    try:
        client = rt.client("generator")
        chunks = doc_chunks(docs)[:n]
        drafts = [
            draft_from_chunk(client, doc_id, text, i) for i, (doc_id, text) in enumerate(chunks)
        ]
    finally:
        set_runtime(None)
        rt.close()
    out = project.path("datasets/drafts") / f"{suite}_generated.jsonl"
    write_jsonl(out, drafts)
    return out
