"""Locust headless load test for http targets (extra `[perf]`), run as a subprocess.

Flags follow Locust's documented headless CLI (`-f -H --headless -u -r -t --csv`); Locust is
not installed in the build environment, so re-check with `locust --help` (docs/DECISIONS.md).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from evalkit.registry import engine
from evalkit.util import is_installed

LOCUSTFILE = """from locust import HttpUser, task, between
import json

BODY = json.loads({body!r})


class EvalkitUser(HttpUser):
    wait_time = between(0, 0.1)

    @task
    def ask(self):
        self.client.request({method!r}, "", json=BODY, headers={headers!r})
"""


@engine("locust", requires_extra="perf")
class LocustEngine:
    """Engine marker: performance load mode with `engine: locust`."""

    package = "locust"


def run_locust(target: Any, users: int, duration_s: int, out_dir: Path) -> dict[str, Any]:
    if not is_installed("locust"):
        return {"skipped": "locust not installed"}
    if not hasattr(target, "url"):
        return {"skipped": "n/a: locust needs an http target"}
    out_dir.mkdir(parents=True, exist_ok=True)
    lf = out_dir / "locustfile.py"
    lf.write_text(
        LOCUSTFILE.format(
            body=json.dumps({"input": "ping"}), method=target.method, headers=target.headers
        ),
        encoding="utf-8",
    )
    prefix = out_dir / "stats"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "locust",
            "-f",
            str(lf),
            "-H",
            target.url,
            "--headless",
            "-u",
            str(users),
            "-r",
            str(users),
            "-t",
            f"{duration_s}s",
            "--csv",
            str(prefix),
        ],
        capture_output=True,
        text=True,
        timeout=duration_s + 120,
    )
    stats = Path(f"{prefix}_stats.csv")
    if not stats.exists():
        return {"skipped": "locust produced no stats"}
    with open(stats, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    total = next((r for r in rows if r.get("Name") == "Aggregated"), rows[-1] if rows else {})
    return {
        "requests": int(total.get("Request Count", 0)),
        "failures": int(total.get("Failure Count", 0)),
        "throughput": float(total.get("Requests/s", 0.0)),
        "p50": float(total.get("50%", 0)) / 1000,
        "p95": float(total.get("95%", 0)) / 1000,
        "p99": float(total.get("99%", 0)) / 1000,
    }
