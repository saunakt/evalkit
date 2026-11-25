"""Label Studio export / import (no SDK needed) (Plan.md §12.2)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evalkit.registry import engine
from evalkit.stats.agreement import cohen_kappa, krippendorff_alpha_nominal
from evalkit.util import write_jsonl

BINARY_CONFIG = """<View>
  <Header value="Criterion"/>
  <Text name="criterion" value="$criterion"/>
  <Header value="User input"/>
  <Text name="input" value="$input"/>
  <Header value="Output"/>
  <Text name="output" value="$output"/>
  <Choices name="label" toName="output" choice="single" required="true">
    <Choice value="PASS"/>
    <Choice value="FAIL"/>
  </Choices>
</View>
"""
MOS_CONFIG = """<View>
  <Text name="output" value="$output"/>
  <Rating name="mos" toName="output" maxRating="5" required="true"/>
</View>
"""


@engine("labelstudio")
class LabelStudioEngine:
    """Engine marker: `evalctl label export|import`."""

    package = "none (plain JSON)"


def export_tasks(
    items: list[dict[str, Any]], out_dir: Path, criterion: str, mos: bool = False
) -> dict[str, Path]:
    """Write tasks.json (Label Studio import format), config.xml and guidelines.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "id": i,
            "data": {
                "item_id": str(it.get("id")),
                "input": it.get("input", ""),
                "output": it.get("output", ""),
                "criterion": criterion,
            },
        }
        for i, it in enumerate(items, start=1)
    ]
    paths = {
        "tasks": out_dir / "tasks.json",
        "config": out_dir / "config.xml",
        "guidelines": out_dir / "guidelines.md",
    }
    paths["tasks"].write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["config"].write_text(MOS_CONFIG if mos else BINARY_CONFIG, encoding="utf-8")
    paths["guidelines"].write_text(
        f"# Labelling guidelines\n\n{criterion}\n\nLabel PASS when the output meets the "
        "criterion, FAIL otherwise. Judge only this criterion; ignore length and style.\n",
        encoding="utf-8",
    )
    return paths


def _choice(annotation: dict[str, Any]) -> str | None:
    for res in annotation.get("result", []):
        val = res.get("value", {})
        if val.get("choices"):
            return str(val["choices"][0])
        if "rating" in val:
            return str(val["rating"])
    return None


def import_labels(export_path: Path, out_path: Path, split: str = "dev") -> dict[str, Any]:
    """Label Studio JSON export → human_labels JSONL (majority label) + agreement stats."""
    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    by_annotator: defaultdict[str, dict[Any, str]] = defaultdict(dict)
    for task in tasks:
        labels = []
        for ann in task.get("annotations", []):
            label = _choice(ann)
            if label is not None:
                labels.append(label)
                by_annotator[str(ann.get("completed_by", "?"))][task["id"]] = label
        if not labels:
            continue
        majority = max(sorted(set(labels)), key=labels.count)
        data = task.get("data", {})
        rows.append(
            {
                "id": data.get("item_id", task["id"]),
                "input": data.get("input", ""),
                "output": data.get("output", ""),
                "label": majority,
                "split": split,
            }
        )
    write_jsonl(out_path, rows)
    return {"rows": len(rows), **agreement(by_annotator)}


def agreement(by_annotator: dict[str, dict[Any, str]]) -> dict[str, Any]:
    """Cohen's kappa for two annotators; Krippendorff's alpha for any number."""
    names = sorted(by_annotator)
    if len(names) < 2:
        return {"annotators": len(names)}
    items = sorted({t for a in names for t in by_annotator[a]})
    matrix = [[by_annotator[a].get(t) for t in items] for a in names]
    out: dict[str, Any] = {
        "annotators": len(names),
        "krippendorff_alpha": krippendorff_alpha_nominal(matrix),
    }
    if len(names) == 2:
        shared = [t for t in items if all(t in by_annotator[a] for a in names)]
        if shared:
            out["cohen_kappa"] = cohen_kappa(
                [by_annotator[names[0]][t] for t in shared],
                [by_annotator[names[1]][t] for t in shared],
            )
    return out
