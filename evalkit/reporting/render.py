"""Render reports: html, md, json, junit, csv + summary and compare (Plan.md §11.3)."""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from evalkit.config import Thresholds
from evalkit.reporting.model import compute, suite_report, summary_report
from evalkit.reporting.store import StoredRun
from evalkit.schema import ResultRow

TEMPLATES = Path(__file__).parent / "templates"
ALL_FORMATS = ("html", "md", "json", "junit", "csv")


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".") if value % 1 else f"{value:.0f}"
    return str(value)


def pct(value: Any) -> str:
    return "–" if value is None else f"{100 * float(value):.1f}%"


def svg_bar(value: Any, lo: Any = None, hi: Any = None, width: int = 120) -> str:
    """Inline SVG bar for a [0,1] value with an optional CI whisker (server-side, no JS)."""
    if value is None or not isinstance(value, int | float) or not 0 <= value <= 1:
        return ""
    w = round(width * float(value), 1)
    ci = ""
    if isinstance(lo, int | float) and isinstance(hi, int | float):
        x1, x2 = round(width * float(lo), 1), round(width * float(hi), 1)
        ci = f'<line x1="{x1}" y1="7" x2="{x2}" y2="7" stroke="#222" stroke-width="1.5"/>'
    return (
        f'<svg width="{width}" height="14" role="img" aria-label="{pct(value)}">'
        f'<rect width="{width}" height="14" fill="#e8ecf1"/>'
        f'<rect width="{w}" height="14" fill="#4a78c2"/>{ci}</svg>'
    )


def env() -> Environment:
    e = Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    e.filters.update(fmt=fmt, pct=pct, bar=svg_bar, tojson=lambda v: json.dumps(v, default=str))
    return e


def write_junit(path: Path, suite: str, rows: list[ResultRow], report: dict[str, Any]) -> None:
    """One testcase per case × metric for must-pass cases, per case otherwise."""
    root = ET.Element("testsuites")
    ts = ET.SubElement(root, "testsuite", name=suite)
    per_case: dict[str, list[ResultRow]] = defaultdict(list)
    for r in rows:
        if r.case_id is not None:
            per_case[r.case_id].append(r)
    n = fails = skips = 0
    for cid in sorted(per_case):
        crow = per_case[cid]
        groups = (
            [
                (f"{cid}::{m}", [r for r in crow if r.metric == m])
                for m in sorted({r.metric for r in crow})
            ]
            if any(r.must_pass for r in crow)
            else [(cid, crow)]
        )
        for name, grp in groups:
            n += 1
            tc = ET.SubElement(ts, "testcase", classname=f"evalkit.{suite}", name=name)
            bad = [r for r in grp if r.skipped is None and r.passed is False]
            if bad:
                fails += 1
                msg = "; ".join(f"{r.metric} (repeat {r.repeat})" for r in bad[:10])
                ET.SubElement(tc, "failure", message=f"failed: {msg}").text = json.dumps(
                    [r.details for r in bad[:3]], default=str
                )[:2000]
            elif all(r.skipped for r in grp):
                skips += 1
                ET.SubElement(tc, "skipped", message=grp[0].skipped or "skipped")
    for m in report["metrics"]:
        if m["scope"] != "dataset":
            continue
        n += 1
        tc = ET.SubElement(
            ts, "testcase", classname=f"evalkit.{suite}", name=f"[dataset] {m['metric']}"
        )
        if m["status"] == "FAIL":
            fails += 1
            ET.SubElement(
                tc, "failure", message=f"{m['metric']}={fmt(m['value'])} target {m['target']}"
            )
        elif m["status"] == "NOT TESTED":
            skips += 1
            ET.SubElement(tc, "skipped", message=m["skipped_summary"] or "not tested")
    ts.set("tests", str(n))
    ts.set("failures", str(fails))
    ts.set("skipped", str(skips))
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_csv(path: Path, rows: list[ResultRow]) -> None:
    cols = [
        "case_id",
        "repeat",
        "metric",
        "value",
        "passed",
        "skipped",
        "must_pass",
        "tags",
        "language",
        "split",
        "latency_s",
        "tokens",
        "cost_usd",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in sorted(rows, key=lambda r: (r.case_id or "", r.repeat or 0, r.metric)):
            w.writerow(
                [
                    r.case_id,
                    r.repeat,
                    r.metric,
                    r.value,
                    r.passed,
                    r.skipped,
                    r.must_pass,
                    ";".join(r.tags),
                    r.language,
                    r.split,
                    r.latency_s,
                    r.tokens,
                    r.cost_usd,
                ]
            )


def render_run(
    stored: StoredRun,
    thresholds: Thresholds,
    out_dir: Path,
    *,
    baseline: StoredRun | None = None,
    suites: list[str] | None = None,
    metrics: list[str] | None = None,
    formats: list[str] | tuple[str, ...] = ALL_FORMATS,
    trace_url_template: str | None = None,
) -> list[Path]:
    """Render selected suites (and metrics) from the store. Returns written files."""
    sel, summaries, gate, rows = compute(stored, thresholds, baseline, suites, metrics)
    e = env()
    written: list[Path] = []
    for suite, ids in sel.items():
        d = out_dir / suite
        d.mkdir(parents=True, exist_ok=True)
        rep = suite_report(suite, ids, summaries[suite], gate.suites[suite], stored, baseline)
        rep["trace_url_template"] = trace_url_template
        srows = [r for r in rows if r.suite == suite]
        written += _write_formats(e, d, "suite", "report", rep, formats)
        if "junit" in formats:
            write_junit(d / "junit.xml", suite, srows, rep)
            written.append(d / "junit.xml")
        if "csv" in formats:
            write_csv(d / "per_case.csv", srows)
            written.append(d / "per_case.csv")
    summ = summary_report(stored, sel, summaries, gate, baseline)
    written += _write_formats(
        e, out_dir, "summary", "summary", summ, [f for f in formats if f in ("html", "md")]
    )
    if baseline is not None and "html" in formats:
        html = e.get_template("compare.html").render(r=summ, gate=gate.as_dict())
        (out_dir / "compare.html").write_text(html, encoding="utf-8")
        written.append(out_dir / "compare.html")
    return written


def _write_formats(
    e: Environment,
    d: Path,
    template: str,
    stem: str,
    data: dict[str, Any],
    formats: list[str] | tuple[str, ...],
) -> list[Path]:
    out = []
    if "html" in formats:
        (d / f"{stem}.html").write_text(
            e.get_template(f"{template}.html").render(r=data), encoding="utf-8"
        )
        out.append(d / f"{stem}.html")
    if "md" in formats:
        (d / f"{stem}.md").write_text(
            e.get_template(f"{template}.md").render(r=data), encoding="utf-8"
        )
        out.append(d / f"{stem}.md")
    if "json" in formats:
        (d / f"{stem}.json").write_text(
            json.dumps(data, indent=2, default=str, sort_keys=True), encoding="utf-8"
        )
        out.append(d / f"{stem}.json")
    return out
