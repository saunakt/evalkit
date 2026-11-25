"""Judge runner: binary, rubric, reference, pairwise, geval, claims (Plan.md §8)."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalkit.config import CalibrationConfig, JudgeConfig, load_judge
from evalkit.judges.prompts import REMINDER, SYSTEM, build_prompt
from evalkit.llm.client import RoleClient
from evalkit.llm.roles import LLMRuntime
from evalkit.util import ConfigError, EvalkitError, extract_json

CITATION = re.compile(r"\s*\[[\w\-#.]+\]")
INLINE_JUDGES: dict[str, JudgeConfig] = {}
METRIC_JUDGES: dict[str, str] = {}  # metric id -> judge id (for planning / calibration)


class JudgeBlocked(EvalkitError):
    """Judge may not run (uncalibrated and allow_uncalibrated is false)."""


class JudgeParseError(EvalkitError):
    """The judge did not return valid JSON after one retry."""


@dataclass
class Verdict:
    passed: bool | None = None
    score: float | None = None  # normalised to [0, 1]
    reasoning: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def inline_judge(
    id: str,
    criterion: str,
    *,
    style: str = "binary",
    rubric: dict[int, str] | None = None,
    metric: str | None = None,
) -> str:
    """Register a code-defined judge (a YAML file with the same id overrides it)."""
    INLINE_JUDGES[id] = JudgeConfig(id=id, criterion=criterion, style=style, rubric=rubric)  # type: ignore[arg-type]
    if metric:
        METRIC_JUDGES[metric] = id
    return id


def calibration_path(judges_dir: Path, judge: JudgeConfig) -> Path:
    return judges_dir / "calibration" / f"{judge.id}@v{judge.version}.json"


class JudgeRunner:
    def __init__(self, runtime: LLMRuntime, judges_dir: Path, cal: CalibrationConfig) -> None:
        self.runtime = runtime
        self.judges_dir = judges_dir
        self.cal = cal
        self.used: dict[str, dict[str, Any]] = {}
        self._judges: dict[str, JudgeConfig] = {}
        self._steps: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- lookup

    def get(self, judge_id: str) -> JudgeConfig:
        with self._lock:
            if judge_id not in self._judges:
                path = self.judges_dir / f"{judge_id}.yaml"
                if path.exists():
                    self._judges[judge_id] = load_judge(path)
                elif judge_id in INLINE_JUDGES:
                    self._judges[judge_id] = INLINE_JUDGES[judge_id]
                else:
                    raise ConfigError(f"unknown judge {judge_id!r} (no {path})")
            return self._judges[judge_id]

    def resolved_model(self, judge: JudgeConfig) -> tuple[str, str]:
        prov, model = self.runtime.role_model(judge.role)
        return (judge.provider or prov, judge.model or model)

    def calibration_status(self, judge: JudgeConfig) -> dict[str, Any]:
        """{calibrated: bool, reason, record?} for the judge's current version + model."""
        path = calibration_path(self.judges_dir, judge)
        if not path.exists():
            return {"calibrated": False, "reason": "no calibration record"}
        rec = json.loads(path.read_text(encoding="utf-8"))
        prov, model = self.resolved_model(judge)
        if (rec.get("provider"), rec.get("model")) != (prov, model):
            return {
                "calibrated": False,
                "reason": f"record is for {rec.get('provider')}:"
                f"{rec.get('model')}, judge resolves to {prov}:{model}",
            }
        ok = rec.get("kappa", 0) >= self.cal.min_kappa and (
            rec.get("fail_recall", 0) >= self.cal.min_fail_recall
        )
        reason = "ok" if ok else f"kappa={rec.get('kappa')} fail_recall={rec.get('fail_recall')}"
        return {
            "calibrated": ok,
            "reason": reason,
            "kappa": rec.get("kappa"),
            "fail_recall": rec.get("fail_recall"),
        }

    def _client(self, judge: JudgeConfig, *, enforce: bool = True) -> tuple[RoleClient, bool]:
        status = self.calibration_status(judge)
        prov, model = self.resolved_model(judge)
        with self._lock:
            self.used[judge.id] = {
                "version": judge.version,
                "provider": prov,
                "model": model,
                **status,
            }
        if enforce and not status["calibrated"] and not self.cal.allow_uncalibrated:
            raise JudgeBlocked(
                f"judge {judge.id}@v{judge.version} uncalibrated ({status['reason']})"
            )
        client = self.runtime.client(judge.role, provider=judge.provider, model=judge.model)
        return client, not status["calibrated"]

    # ----------------------------------------------------------------- calling

    def call_json(
        self,
        judge: JudgeConfig,
        fmt: str,
        prompt: str,
        required: str,
        meta: dict[str, Any],
        enforce: bool = True,
    ) -> tuple[dict[str, Any], bool]:
        client, uncal = self._client(judge, enforce=enforce)
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        meta = {"judge_id": judge.id, "style": judge.style, **meta}
        for attempt in range(2):
            resp = client.complete(
                system=SYSTEM,
                messages=messages,
                temperature=judge.temperature,
                max_tokens=judge.max_tokens,
                response_format="json",
                meta=meta,
            )
            data = extract_json(resp.text)
            if isinstance(data, dict) and required in data:
                return data, uncal
            if attempt == 0:
                messages = [
                    *messages,
                    {"role": "assistant", "content": resp.text},
                    {"role": "user", "content": REMINDER},
                ]
        raise JudgeParseError("judge_parse_error")

    def binary(
        self,
        judge_id: str,
        *,
        criterion: str | None = None,
        sections: dict[str, Any] | None = None,
        case_id: str | None = None,
        enforce: bool = True,
    ) -> Verdict:
        judge = self.get(judge_id)
        fmt = "reference" if judge.style == "reference" else "binary"
        prompt = build_prompt(judge, fmt, criterion=criterion, sections=sections)
        data, uncal = self.call_json(
            judge, fmt, prompt, "verdict", {"case_id": case_id, "criterion": criterion}, enforce
        )
        ok = str(data.get("verdict", "")).strip().upper() == "PASS"
        return Verdict(
            ok,
            1.0 if ok else 0.0,
            str(data.get("reasoning", "")),
            {"uncalibrated": uncal} if uncal else {},
        )

    def rubric(
        self,
        judge_id: str,
        *,
        criterion: str | None = None,
        sections: dict[str, Any] | None = None,
        case_id: str | None = None,
        threshold: float = 0.5,
    ) -> Verdict:
        judge = self.get(judge_id)
        prompt = build_prompt(judge, "rubric", criterion=criterion, sections=sections)
        data, uncal = self.call_json(
            judge, "rubric", prompt, "score", {"case_id": case_id, "style": "rubric"}
        )
        return _scored(data, uncal, threshold)

    def geval(
        self,
        judge_id: str,
        *,
        criterion: str | None = None,
        sections: dict[str, Any] | None = None,
        case_id: str | None = None,
        threshold: float = 0.5,
    ) -> Verdict:
        """Step 1: evaluation steps (cached per judge version + criterion); step 2: score."""
        judge = self.get(judge_id)
        key = f"{judge.id}@{judge.version}:{criterion or judge.criterion}"
        if key not in self._steps:
            prompt = build_prompt(judge, "steps", criterion=criterion)
            data, _ = self.call_json(
                judge,
                "steps",
                prompt,
                "steps",
                {"case_id": case_id, "step": "steps", "style": "geval"},
            )
            self._steps[key] = [str(s) for s in data.get("steps", [])]
        secs = {
            "Evaluation steps": "\n".join(f"{i + 1}. {s}" for i, s in enumerate(self._steps[key])),
            **(sections or {}),
        }
        prompt = build_prompt(judge, "geval", criterion=criterion, sections=secs)
        data, uncal = self.call_json(
            judge, "geval", prompt, "score", {"case_id": case_id, "style": "geval"}
        )
        verdict = _scored(data, uncal, threshold)
        verdict.details["steps"] = self._steps[key]
        return verdict

    def pairwise(
        self,
        judge_id: str,
        *,
        a: str,
        b: str,
        criterion: str | None = None,
        sections: dict[str, Any] | None = None,
        case_id: str | None = None,
    ) -> Verdict:
        """Run (A,B) and (B,A); inconsistent winners → tie with position_unstable."""
        judge = self.get(judge_id)
        winners, uncal = [], False
        for first, second, swapped in ((a, b, False), (b, a, True)):
            secs = {**(sections or {}), "Response A": first, "Response B": second}
            prompt = build_prompt(judge, "pairwise", criterion=criterion, sections=secs)
            data, u = self.call_json(
                judge,
                "pairwise",
                prompt,
                "winner",
                {"case_id": case_id, "style": "pairwise", "swapped": swapped},
            )
            uncal = uncal or u
            w = str(data.get("winner", "tie")).strip().upper()
            winners.append(
                {"A": "B", "B": "A"}.get(w, "TIE") if swapped else (w if w in ("A", "B") else "TIE")
            )
        unstable = winners[0] != winners[1]
        final = "TIE" if unstable else winners[0]
        score = {"A": 1.0, "TIE": 0.5, "B": 0.0}[final]
        details: dict[str, Any] = {
            "winner": final.lower() if final == "TIE" else final,
            "runs": winners,
            "position_unstable": unstable,
        }
        if uncal:
            details["uncalibrated"] = True
        return Verdict(score > 0.5 if final != "TIE" else None, score, "", details)

    def claims(
        self,
        judge_id: str,
        *,
        text: str,
        context: str,
        case_id: str | None = None,
        threshold: float = 0.5,
    ) -> Verdict:
        """Extract atomic claims from `text`, verify each against `context`."""
        judge = self.get(judge_id)
        text = CITATION.sub("", text).strip()  # citation markers are not claims
        p1 = build_prompt(
            judge,
            "extract",
            sections={"Text": text},
            criterion="Split the text into short atomic factual claims.",
        )
        data, uncal = self.call_json(
            judge, "extract", p1, "claims", {"case_id": case_id, "step": "extract", "text": text}
        )
        claims = [str(c) for c in data.get("claims", []) if str(c).strip()]
        if not claims:
            return Verdict(True, 1.0, "no claims", {"claims": [], "uncalibrated": uncal})
        p2 = build_prompt(judge, "verify", sections={"Claims": claims, "Context": context})
        data, u2 = self.call_json(
            judge,
            "verify",
            p2,
            "supported",
            {"case_id": case_id, "step": "verify", "n_claims": len(claims)},
        )
        sup = [bool(x) for x in data.get("supported", [])][: len(claims)]
        sup += [False] * (len(claims) - len(sup))
        score = sum(sup) / len(claims)
        details: dict[str, Any] = {
            "claims": claims,
            "supported": sup,
            "unsupported": [c for c, s in zip(claims, sup, strict=True) if not s],
        }
        if uncal or u2:
            details["uncalibrated"] = True
        return Verdict(score >= threshold, score, "", details)


def _scored(data: dict[str, Any], uncal: bool, threshold: float) -> Verdict:
    try:
        raw = float(data.get("score", 1))
    except (TypeError, ValueError):
        raw = 1.0
    raw = min(5.0, max(1.0, raw))
    score = (raw - 1) / 4
    details: dict[str, Any] = {"raw_score": raw}
    if uncal:
        details["uncalibrated"] = True
    return Verdict(score >= threshold, score, str(data.get("reasoning", "")), details)
