from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from evalkit.config import (
    CalibrationConfig,
    EvalkitConfig,
    Overrides,
    ProviderConfig,
    RoleModel,
    load_project,
)
from evalkit.judges.calibration import calibrate
from evalkit.judges.runner import JudgeBlocked, JudgeParseError, JudgeRunner, inline_judge
from evalkit.llm.roles import LLMRuntime
from tests.conftest import ROOT


def runner(tmp_path: Path, script: dict[str, Any], allow: bool = True) -> JudgeRunner:
    (tmp_path / "script.yaml").write_text(yaml.safe_dump(script), encoding="utf-8")
    cfg = EvalkitConfig(
        providers={"fake": ProviderConfig(type="fake", script="script.yaml")},
        models={"judge": RoleModel(provider="fake", model="fj")},
    )
    jdir = tmp_path / "judges"
    jdir.mkdir(exist_ok=True)
    return JudgeRunner(LLMRuntime(cfg, tmp_path), jdir, CalibrationConfig(allow_uncalibrated=allow))


def test_binary_verdict_and_uncalibrated_flag(tmp_path: Path) -> None:
    jid = inline_judge("t_binary", "Is it polite?")
    jr = runner(tmp_path, {"judge_script": [{"case": "c1", "verdict": "FAIL"}]})
    v = jr.binary(jid, sections={"Output": "go away"}, case_id="c1")
    assert v.passed is False and v.details["uncalibrated"] is True
    assert jr.binary(jid, sections={"Output": "hello"}, case_id="c2").passed is True
    assert jr.used["t_binary"]["calibrated"] is False


def test_blocked_when_uncalibrated_and_not_allowed(tmp_path: Path) -> None:
    jid = inline_judge("t_block", "x")
    jr = runner(tmp_path, {}, allow=False)
    with pytest.raises(JudgeBlocked):
        jr.binary(jid, sections={"Output": "y"})


def test_parse_retry_then_error(tmp_path: Path) -> None:
    jid = inline_judge("t_parse", "x")
    jr = runner(tmp_path, {"judge_script": [{"judge": "t_parse", "response": "not json"}]})
    with pytest.raises(JudgeParseError):
        jr.binary(jid, sections={"Output": "y"})
    fake = jr.runtime.provider_client("fake")
    assert len(fake.calls) == 2 and "Return ONLY the JSON" in fake.calls[1]["prompt"]  # type: ignore[attr-defined]


def test_pairwise_position_swap(tmp_path: Path) -> None:
    jid = inline_judge("t_pair", "better?", style="pairwise")
    consistent = runner(
        tmp_path,
        {
            "judge_script": [
                {"judge": "t_pair", "match": "Response A\\nGOOD", "winner": "A"},
                {"judge": "t_pair", "match": "Response A\\nBAD", "winner": "B"},
            ]
        },
    )
    v = consistent.pairwise(jid, a="GOOD", b="BAD")
    assert v.score == 1.0 and v.details["position_unstable"] is False
    biased = runner(tmp_path, {"judge_script": [{"judge": "t_pair", "winner": "A"}]})
    v2 = biased.pairwise(jid, a="GOOD", b="BAD")
    assert v2.score == 0.5 and v2.details["position_unstable"] is True


def test_rubric_geval_and_claims(tmp_path: Path) -> None:
    rid = inline_judge("t_rubric", "tone", style="rubric", rubric={1: "rude", 5: "warm"})
    jr = runner(tmp_path, {"judge_script": [{"judge": "t_rubric", "score": 4}]})
    assert jr.rubric(rid, sections={"Output": "hi"}).score == pytest.approx(0.75)
    gid = inline_judge("t_geval", "coherent", style="geval")
    jr.geval(gid, sections={"Output": "a"})
    jr.geval(gid, sections={"Output": "b"})
    steps = [
        c
        for c in jr.runtime.provider_client("fake").calls  # type: ignore[attr-defined]
        if c["meta"].get("step") == "steps"
    ]
    assert len(steps) == 1  # evaluation steps cached per judge version
    cid = inline_judge("t_claims", "supported?", style="claims")
    cj = runner(
        tmp_path,
        {"judge_script": [{"judge": "t_claims", "step": "verify", "supported": [True, False]}]},
    )
    v = cj.claims(cid, text="Returns take 30 days. Shipping is free.", context="30 days")
    assert v.score == 0.5 and v.details["unsupported"] == ["Shipping is free."]


def test_calibration_record_and_gating(tmp_path: Path) -> None:
    for name in ("evalkit.yaml", "thresholds.yaml"):
        shutil.copy(ROOT / name, tmp_path / name)
    shutil.copytree(ROOT / "judges", tmp_path / "judges", ignore=shutil.ignore_patterns("calib*"))
    (tmp_path / "datasets" / "human_labels").mkdir(parents=True)
    shutil.copy(
        ROOT / "datasets/human_labels/policy_compliance.jsonl",
        tmp_path / "datasets/human_labels/policy_compliance.jsonl",
    )
    (tmp_path / "examples/shopease").mkdir(parents=True)
    shutil.copy(
        ROOT / "examples/shopease/fake_script.yaml", tmp_path / "examples/shopease/fake_script.yaml"
    )
    project = load_project(tmp_path / "evalkit.yaml", None, Overrides(provider="fake"))
    rec = calibrate(project, "policy_compliance")
    assert rec["passed"] and rec["kappa"] == pytest.approx(0.8235, abs=1e-3)
    assert rec["splits"]["dev"]["disagreements"] == [
        {"id": "pc-08", "human": "FAIL", "judge": "PASS"}
    ]
    path = tmp_path / "judges/calibration/policy_compliance@v3.json"
    assert json.loads(path.read_text(encoding="utf-8"))["model"] == "fake"
    jr = JudgeRunner(
        LLMRuntime(project.config, tmp_path), tmp_path / "judges", project.config.calibration
    )
    assert jr.calibration_status(jr.get("policy_compliance"))["calibrated"] is True
    other = load_project(
        tmp_path / "evalkit.yaml",
        None,
        Overrides(provider="fake", roles=["judge=fake:other-model"]),
    )
    jr2 = JudgeRunner(
        LLMRuntime(other.config, tmp_path), tmp_path / "judges", other.config.calibration
    )
    st = jr2.calibration_status(jr2.get("policy_compliance"))
    assert st["calibrated"] is False and "record is for fake:fake" in st["reason"]
