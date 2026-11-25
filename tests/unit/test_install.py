"""install.yaml + `evalctl setup`: run_cmd, HTTP and importlib.metadata are mocked."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from evalkit.config import Project, load_project
from evalkit.install import apply as apply_mod
from evalkit.install import external as external_mod
from evalkit.install.apply import clean_lock, setup
from evalkit.install.config import (
    EXTRA_NAMES,
    InstallConfig,
    core_packages,
    extra_packages,
    load_install,
)
from evalkit.install.resolve import needed_extras, resolve
from evalkit.util import ConfigError
from tests.conftest import ROOT

PROVIDERS = {
    "fake": {"type": "fake"},
    "local": {"type": "openai_compatible", "base_url": "http://localhost:11434/v1"},
    "claude": {"type": "anthropic"},
}


class FakeRun:
    """Stand-in for run_cmd: records calls, returns canned output."""

    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.stdout, self.returncode = stdout, returncode

    def __call__(self, cmd: list[str], cwd: Path | None = None, timeout: float = 0) -> Any:
        self.calls.append(cmd)
        return SimpleNamespace(returncode=self.returncode, stdout=self.stdout, stderr="")


def project(
    tmp_path: Path,
    suites: dict[str, Any],
    judge: str = "fake",
    engines: dict[str, str] | None = None,
) -> Project:
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    cfg = {
        "project": "t",
        "target": "targets/none.yaml",
        "providers": PROVIDERS,
        "models": {"judge": {"provider": judge, "model": "m"}},
        "suites": suites,
        "engines": engines or {},
    }
    (tmp_path / "evalkit.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return load_project(tmp_path / "evalkit.yaml")


def install_cfg(tmp_path: Path, data: dict[str, Any]) -> InstallConfig:
    path = tmp_path / "install.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_install(path)


@pytest.fixture(autouse=True)
def nothing_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apply_mod, "is_dist_installed", lambda name: False)
    monkeypatch.setattr(external_mod, "http_ok", lambda url, timeout=3.0: False)


# ------------------------------------------------------------------ validation


def test_unknown_key_suggests(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="did you mean 'deepeval'"):
        install_cfg(tmp_path, {"extras": {"deepevl": True}})


def test_bad_action_value(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"external\.docker\.action"):
        install_cfg(tmp_path, {"external": {"docker": {"action": "launch"}}})


def test_install_on_check_only_tool_is_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="check-only") as exc:
        install_cfg(tmp_path, {"external": {"gitleaks": {"action": "install"}}})
    assert exc.value.exit_code == 2
    with pytest.raises(ConfigError, match="'start' is only supported"):
        install_cfg(tmp_path, {"external": {"docker": {"action": "start"}}})


def test_yaml_off_is_accepted(tmp_path: Path) -> None:
    (tmp_path / "install.yaml").write_text(
        "external:\n  docker: { action: off }\n", encoding="utf-8"
    )
    assert load_install(tmp_path / "install.yaml").external.docker.action == "off"


def test_shipped_install_yaml_matches_pyproject() -> None:
    cfg = load_install(ROOT / "install.yaml")
    assert set(extra_packages(ROOT)) == set(EXTRA_NAMES)
    assert cfg.core == core_packages(ROOT)
    assert cfg.extras.textmetrics is False and cfg.extras.security is False
    assert cfg.external.local_model.action == "check" and cfg.external.wiremock.action == "off"


# ------------------------------------------------------------------ resolution


CASES = {
    "core_only": ({"quality": {"enabled": True, "metrics": ["contains"]}}, "fake", {}, set()),
    "rag_ragas": (
        {"rag": {"enabled": True, "metrics": ["faithfulness"]}},
        "fake",
        {"faithfulness": "ragas"},
        {"ragas"},
    ),
    "ml_rai": (
        {
            "multilingual": {"enabled": True, "metrics": "all"},
            "responsible_ai": {"enabled": True, "metrics": "all"},
        },
        "fake",
        {},
        {"rai", "textmetrics"},
    ),  # multilingual includes the text-similarity metrics
    "anthropic_judge": (
        {"quality": {"enabled": True, "metrics": ["contains"]}},
        "claude",
        {},
        {"anthropic"},
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_modes(tmp_path: Path, case: str) -> None:
    suites, judge, engines, needed = CASES[case]
    proj = project(tmp_path, suites, judge, engines)
    assert set(needed_extras(proj)) == needed
    choose = resolve(InstallConfig(), proj)  # defaults: textmetrics false, the rest `auto`
    assert set(choose.selected) == needed - {"textmetrics"}
    assert ("textmetrics" in needed) == any("'textmetrics'" in w for w in choose.warnings)
    auto = resolve(InstallConfig.model_validate({"extras": {"mode": "auto", "voice": True}}), proj)
    assert set(auto.selected) == needed and "ignores the extra switches" in auto.notes[0]
    every = resolve(InstallConfig.model_validate({"extras": {"mode": "all"}}), proj)
    assert set(every.selected) == set(EXTRA_NAMES) - {"security"}


def test_auto_switch_installs_only_when_needed(tmp_path: Path) -> None:
    proj = project(tmp_path, CASES["core_only"][0])
    res = resolve(InstallConfig.model_validate({"extras": {"voice": "auto", "agent": True}}), proj)
    assert res.selected == ["agent"]


def test_needed_but_false_warns_and_is_not_installed(tmp_path: Path) -> None:
    proj = project(tmp_path, {"quality": {"enabled": True, "metrics": ["bleu", "rouge"]}})
    cfg = InstallConfig.model_validate({"extras": {"textmetrics": False, "agent": True}})
    run = FakeRun()
    report = setup(proj, cfg, no_external=True, run=run)
    warning = next(w for w in report.warnings if "textmetrics" in w)
    assert "quality.bleu" in warning and "quality.rouge" in warning and "Not tested" in warning
    sync = next(c for c in run.calls if c[:2] == ["uv", "sync"])
    assert "textmetrics" not in sync and sync[-2:] == ["--extra", "agent"]


def test_off_tool_needed_by_feature_warns(tmp_path: Path) -> None:
    proj = project(
        tmp_path,
        {
            "security": {
                "enabled": True,
                "params": {"generate": "promptfoo"},
                "metrics": ["attack_success_rate"],
            }
        },
    )
    res = resolve(InstallConfig(), proj)
    assert any("'promptfoo' is off" in w for w in res.warnings)


def test_conflicting_extras_in_choose_mode(tmp_path: Path) -> None:
    proj = project(tmp_path, CASES["core_only"][0])
    cfg = InstallConfig.model_validate({"extras": {"security": True, "anthropic": True}})
    with pytest.raises(ConfigError, match="cannot be installed together"):
        resolve(cfg, proj)


# ------------------------------------------------------------------ setup


def test_dry_run_makes_zero_run_cmd_calls(tmp_path: Path) -> None:
    proj = project(tmp_path, CASES["rag_ragas"][0], engines=CASES["rag_ragas"][2])
    cfg = InstallConfig.model_validate(
        {"external": {"promptfoo": {"action": "install"}, "wiremock": {"action": "start"}}}
    )
    run = FakeRun()
    report = setup(proj, cfg, dry_run=True, run=run)
    assert run.calls == []
    flat = [" ".join(c) for c in report.commands]
    assert any("uv sync --inexact --no-install-project --extra ragas" in c for c in flat)
    assert "npm install" in flat and any("up -d wiremock" in c for c in flat)
    assert report.exit_code == 0


def test_check_exits_1_when_needed_check_tool_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proj = project(tmp_path, CASES["core_only"][0], judge="local")
    cfg = InstallConfig()  # local_model: check
    report = setup(proj, cfg, check=True, run=FakeRun())
    tool = next(t for t in report.tools if t.name == "local_model")
    assert tool.state == "missing" and tool.needed_by and report.exit_code == 1
    monkeypatch.setattr(external_mod, "http_ok", lambda url, timeout=3.0: True)
    assert setup(proj, cfg, check=True, run=FakeRun()).exit_code == 0


def test_check_ignores_missing_tool_nobody_needs(tmp_path: Path) -> None:
    proj = project(tmp_path, CASES["core_only"][0])
    report = setup(proj, InstallConfig(), check=True, run=FakeRun())
    assert next(t for t in report.tools if t.name == "local_model").state == "missing"
    assert report.exit_code == 0


def test_check_reports_selected_extra_not_installed(tmp_path: Path) -> None:
    proj = project(tmp_path, CASES["rag_ragas"][0], engines=CASES["rag_ragas"][2])
    report = setup(proj, InstallConfig(), check=True, no_external=True, run=FakeRun())
    assert report.extras["ragas"] == "missing" and report.exit_code == 1


def test_already_installed_extras_are_not_reinstalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(apply_mod, "is_dist_installed", lambda name: name == "ragas")
    proj = project(tmp_path, CASES["rag_ragas"][0], engines=CASES["rag_ragas"][2])
    run = FakeRun()
    report = setup(proj, InstallConfig(), no_external=True, run=run)
    assert not any(c[:2] == ["uv", "sync"] for c in run.calls)
    assert report.extras["ragas"] == "already"


def test_requirements_lock_excludes_editable_evalkit(tmp_path: Path) -> None:
    proj = project(tmp_path, CASES["core_only"][0])
    out = (
        "-e file:///C:/work/evalkit\nevalkit @ file:///C:/work/evalkit\n"
        "httpx==0.28.1 \\\n    --hash=sha256:abc\npydantic==2.14.4\n"
    )
    report = setup(proj, InstallConfig(), no_external=True, run=FakeRun(stdout=out))
    text = (tmp_path / "requirements.lock").read_text(encoding="utf-8")
    assert report.lock_file == "requirements.lock"
    assert "evalkit" not in text and "-e " not in text and "httpx==0.28.1" in text
    assert clean_lock("file:./x\nrich==1\n") == "rich==1\n"


def test_failed_external_install_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(external_mod.shutil, "which", lambda name: f"/bin/{name}")
    proj = project(tmp_path, CASES["core_only"][0])
    cfg = InstallConfig.model_validate({"external": {"playwright_browsers": {"action": "install"}}})
    run = FakeRun(returncode=1)
    report = setup(proj, cfg, no_lock=True, run=run)
    tool = next(t for t in report.tools if t.name == "playwright_browsers")
    assert tool.state == "failed" and run.calls[-1][-2:] == ["install", "chromium"]


def test_report_json_written(tmp_path: Path) -> None:
    from evalkit.install.report import write_report

    proj = project(tmp_path, CASES["core_only"][0])
    report = setup(proj, InstallConfig(), dry_run=True, run=FakeRun())
    path = write_report(report, tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["mode"] == "dry-run" and "required_missing" in data and "tools" in data
