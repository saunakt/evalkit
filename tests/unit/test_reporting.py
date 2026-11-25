from __future__ import annotations

import csv
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from evalkit.config import Bound, Thresholds
from evalkit.reporting.render import render_run, svg_bar
from evalkit.reporting.store import ResultStore, load_run
from evalkit.schema import ResultRow, RunInfo

SNAPSHOTS = Path(__file__).parent.parent / "fixtures" / "snapshots"


def row(
    suite: str,
    case: str | None,
    metric: str,
    passed: bool | None,
    value: float | None = None,
    repeat: int = 0,
    must: bool = False,
    **details: Any,
) -> ResultRow:
    return ResultRow(
        run_id="run-1",
        suite=suite,
        case_id=case,
        repeat=repeat if case else None,
        metric=metric,
        value=value if value is not None else (None if passed is None else float(passed)),
        passed=passed,
        threshold=None,
        skipped=details.pop("skipped", None),
        details=details,
        tags=["refund"] if case and case.endswith("1") else ["shipping"],
        language="en",
        split="dev",
        must_pass=must,
        latency_s=0.5,
        tokens=100,
        cost_usd=0.0,
        trace_id=None,
    )


@pytest.fixture
def stored(tmp_path: Path) -> Any:
    rows = [
        row("quality", "q1", "contains", True),
        row("quality", "q1", "contains", True, repeat=1),
        row("quality", "q2", "contains", False, missing=["30 days"]),
        row("quality", "q2", "contains", True, repeat=1),
        row("quality", "q3", "contains", True),
        row("quality", "q3", "contains", True, repeat=1),
        row("quality", "q1", "bleu", None, skipped="sacrebleu not installed"),
        row("security", "s1", "prompt_leakage", False, must=True, reason="canary leaked"),
        row("security", None, "attack_success_rate", None, value=0.25),
    ]
    st = ResultStore(tmp_path / "results" / "run-1")
    st.write_info(
        RunInfo(
            run_id="run-1",
            started="2026-11-25T10:00:00+05:30",
            profile="nightly",
            target="shopease",
            suites={
                "quality": ["contains", "bleu"],
                "security": ["prompt_leakage", "attack_success_rate"],
            },
            repeats=2,
            seed=42,
        )
    )
    st.write_rows(rows)
    st.write_panels(
        {
            "security": [
                {
                    "title": "OWASP LLM Top 10",
                    "kind": "table",
                    "columns": ["id", "attempts", "successes", "ASR"],
                    "rows": [["LLM07", 4, 1, 0.25]],
                }
            ]
        }
    )
    return load_run(tmp_path / "results" / "run-1")


THR = Thresholds(
    defaults={"pass_rate": 0.8}, suites={"security": {"attack_success_rate": Bound(max=0.02)}}
)


def test_render_writes_per_suite_folders_and_summary(stored: Any, tmp_path: Path) -> None:
    out = tmp_path / "reports"
    render_run(stored, THR, out)
    for suite in ("quality", "security"):
        for name in ("report.html", "report.md", "report.json", "junit.xml", "per_case.csv"):
            assert (out / suite / name).exists(), f"{suite}/{name}"
    assert (out / "summary.html").exists() and (out / "summary.md").exists()
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "Gate FAILED" in summary
    assert "attack_success_rate value 0.250 > 0.02" in summary
    assert "must-pass security/s1 failed prompt_leakage" in summary


def test_html_is_self_contained(stored: Any, tmp_path: Path) -> None:
    render_run(stored, THR, tmp_path / "r")
    for html in (tmp_path / "r").rglob("*.html"):
        text = html.read_text(encoding="utf-8")
        assert not re.search(r"(src|href)=[\"']?(https?:)?//", text), html
        assert "<link" not in text and "<script src" not in text


def test_markdown_snapshot(stored: Any, tmp_path: Path) -> None:
    from evalkit.registry import load_all

    load_all()  # descriptions/scopes come from the registry: load it for a stable snapshot
    render_run(stored, THR, tmp_path / "r", formats=["md"])
    text = (tmp_path / "r" / "quality" / "report.md").read_text(encoding="utf-8")
    snap = SNAPSHOTS / "quality_report.md"
    if os.environ.get("UPDATE_SNAPSHOTS") or not snap.exists():
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(text, encoding="utf-8", newline="\n")
    assert text == snap.read_text(encoding="utf-8")


def test_junit_and_csv(stored: Any, tmp_path: Path) -> None:
    render_run(stored, THR, tmp_path / "r")
    ts = ET.parse(tmp_path / "r" / "security" / "junit.xml").getroot().find("testsuite")
    assert ts is not None
    names = [tc.get("name") for tc in ts.iter("testcase")]
    assert "s1::prompt_leakage" in names and "[dataset] attack_success_rate" in names
    assert ts.get("failures") == "2"
    q = ET.parse(tmp_path / "r" / "quality" / "junit.xml").getroot().find("testsuite")
    assert q is not None and [tc.get("name") for tc in q.iter("testcase")] == ["q1", "q2", "q3"]
    with open(tmp_path / "r" / "quality" / "per_case.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 7 and rows[0]["case_id"] == "q1"


def test_subset_rerender(stored: Any, tmp_path: Path) -> None:
    out = tmp_path / "sub"
    render_run(stored, THR, out, suites=["quality"], metrics=["contains"], formats=["md"])
    assert not (out / "security").exists()
    text = (out / "quality" / "report.md").read_text(encoding="utf-8")
    assert "`contains`" in text and "bleu" not in text


def test_not_tested_and_svg() -> None:
    assert svg_bar(0.5).count("<rect") == 2
    assert svg_bar(None) == "" and svg_bar(3.2) == ""
