"""Judge calibration against human labels (Plan.md §8.3, `evalctl calibrate`)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from evalkit.config import Project
from evalkit.judges.runner import JudgeRunner, calibration_path
from evalkit.llm.roles import LLMRuntime, set_runtime
from evalkit.stats.agreement import judge_agreement
from evalkit.util import DataError, read_jsonl

LABELS = ("PASS", "FAIL")


def load_labels(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DataError(f"no human labels at {path}")
    rows = read_jsonl(path)
    for i, r in enumerate(rows, start=1):
        if str(r.get("label", "")).upper() not in LABELS or "output" not in r:
            raise DataError(f"{path}:{i}: rows need output and label PASS|FAIL")
    return rows


def judge_row(jr: JudgeRunner, judge_id: str, row: dict[str, Any]) -> str:
    sections = {
        "User input": row.get("input", ""),
        "Output": row["output"],
        "Context": row.get("context"),
        "Reference": row.get("reference"),
    }
    v = jr.binary(judge_id, sections=sections, case_id=str(row.get("id")), enforce=False)
    return "PASS" if v.passed else "FAIL"


def split_stats(rows: list[dict[str, Any]], verdicts: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("dev", "heldout", "all"):
        idx = [i for i, r in enumerate(rows) if split == "all" or r.get("split", "dev") == split]
        if not idx:
            continue
        human = [str(rows[i]["label"]).upper() for i in idx]
        judge = [verdicts[i] for i in idx]
        stats = judge_agreement(human, judge)
        stats["disagreements"] = [
            {"id": rows[i].get("id"), "human": h, "judge": j}
            for i, h, j in zip(idx, human, judge, strict=True)
            if h != j
        ]
        out[split] = stats
    return out


def calibrate(project: Project, judge_id: str) -> dict[str, Any]:
    """Run the judge on every labelled row; write the JSON record + Markdown summary."""
    cfg = project.config
    rt = LLMRuntime(cfg, project.root)
    set_runtime(rt)
    try:
        judges_dir = project.path(cfg.judges_dir)
        jr = JudgeRunner(rt, judges_dir, cfg.calibration)
        judge = jr.get(judge_id)
        rows = load_labels(project.path(f"datasets/human_labels/{judge_id}.jsonl"))
        verdicts = [judge_row(jr, judge_id, r) for r in rows]
        provider, model = jr.resolved_model(judge)
    finally:
        set_runtime(None)
        rt.close()
    splits = split_stats(rows, verdicts)
    headline = splits.get("heldout") or splits["all"]
    record = {
        "judge_id": judge.id,
        "version": judge.version,
        "provider": provider,
        "model": model,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n": len(rows),
        "kappa": round(headline["kappa"], 4),
        "fail_recall": round(headline["fail_recall"], 4),
        "basis": "heldout" if "heldout" in splits else "all",
        "thresholds": {
            "min_kappa": cfg.calibration.min_kappa,
            "min_fail_recall": cfg.calibration.min_fail_recall,
        },
        "splits": splits,
    }
    record["passed"] = (
        record["kappa"] >= cfg.calibration.min_kappa
        and record["fail_recall"] >= cfg.calibration.min_fail_recall
    )
    path = calibration_path(judges_dir, judge)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    path.with_suffix(".md").write_text(markdown_summary(record), encoding="utf-8")
    return record


def markdown_summary(rec: dict[str, Any]) -> str:
    lines = [
        f"# Calibration: {rec['judge_id']} v{rec['version']}",
        "",
        f"Judge model: `{rec['provider']}:{rec['model']}` · rows: {rec['n']} · "
        f"created {rec['created']}",
        "",
        f"**Status: {'PASS' if rec['passed'] else 'FAIL'}** — kappa {rec['kappa']:.3f} "
        f"(min {rec['thresholds']['min_kappa']}), FAIL recall {rec['fail_recall']:.3f} "
        f"(min {rec['thresholds']['min_fail_recall']}), basis: {rec['basis']} split",
        "",
        "| split | n | accuracy | kappa | FAIL precision | FAIL recall | FAIL F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for split, s in rec["splits"].items():
        lines.append(
            f"| {split} | {s['n']} | {s['accuracy']:.3f} | {s['kappa']:.3f} | "
            f"{s['fail_precision']:.3f} | {s['fail_recall']:.3f} | {s['fail_f1']:.3f} |"
        )
    lines += ["", "## Confusion matrix (positive class = FAIL)", ""]
    c = rec["splits"]["all"]["confusion"]
    lines += [
        "| | judge FAIL | judge PASS |",
        "|---|---|---|",
        f"| human FAIL | {c['tp']} | {c['fn']} |",
        f"| human PASS | {c['fp']} | {c['tn']} |",
        "",
        "## Disagreements",
        "",
    ]
    dis = rec["splits"]["all"]["disagreements"]
    lines += [f"- `{d['id']}`: human {d['human']}, judge {d['judge']}" for d in dis] or ["None."]
    return "\n".join(lines) + "\n"
