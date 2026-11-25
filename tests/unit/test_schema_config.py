from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evalkit.config import Overrides, interpolate, load_project
from evalkit.registry import Registry, metric
from evalkit.schema import Case, MetricResult
from evalkit.util import ConfigError, DataError, deep_merge, extract_json, normalize_text


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------ cases


def test_case_from_jsonl_reports_every_bad_line(tmp_path: Path) -> None:
    p = write(
        tmp_path / "c.jsonl",
        '{"id": "a", "input": "hi"}\n'
        "not json\n"
        '{"id": "b", "split": "train"}\n'
        '{"input": "no id"}\n'
        '{"id": "a", "input": "dup"}\n',
    )
    with pytest.raises(DataError) as exc:
        Case.from_jsonl(p)
    msg = str(exc.value)
    assert f"{p}:2: invalid JSON" in msg
    assert f"{p}:3: split" in msg
    assert f"{p}:4: id" in msg
    assert f"{p}:5: duplicate id 'a'" in msg


def test_case_schema_alias_and_extra_fields(tmp_path: Path) -> None:
    p = write(tmp_path / "c.jsonl", '{"id": "x", "schema": {"type": "object"}, "custom": 1}\n')
    (case,) = Case.from_jsonl(p)
    assert case.schema_ == {"type": "object"}
    assert case.model_extra == {"custom": 1}


def test_metric_result_defaults() -> None:
    r = MetricResult(metric="m")
    assert r.value is None and r.passed is None and r.skipped is None


# ------------------------------------------------------------------ config


def _project(tmp_path: Path, cfg: dict[str, object]) -> Path:
    write(tmp_path / "evalkit.yaml", yaml.safe_dump(cfg))
    write(tmp_path / "thresholds.yaml", "defaults: {pass_rate: 0.9}\n")
    return tmp_path / "evalkit.yaml"


BASE = {
    "project": "t",
    "providers": {"fake": {"type": "fake"}, "local": {"type": "openai_compatible"}},
    "models": {"judge": {"provider": "local", "model": "m1"}, "sut": {"provider": "local"}},
    "run": {"repeats": 3, "seed": 1},
    "suites": {"quality": {"enabled": True, "params": {"a": 1, "b": {"c": 2}}}},
}


def test_profile_deep_merge_and_cli_overrides(tmp_path: Path) -> None:
    path = _project(tmp_path, BASE)
    write(
        tmp_path / "profiles" / "p.yaml",
        "run: {repeats: 5}\nsuites: {quality: {params: {b: {d: 3}}}}\n",
    )
    proj = load_project(path, "p", Overrides(sets=["run.seed=9"], suites=["rag"]))
    assert proj.config.run.repeats == 5
    assert proj.config.run.seed == 9
    assert proj.config.suites["quality"].params == {"a": 1, "b": {"c": 2, "d": 3}}
    assert proj.enabled_suites() == ["rag"]


def test_provider_override_maps_every_role(tmp_path: Path) -> None:
    proj = load_project(_project(tmp_path, BASE), None, Overrides(provider="fake"))
    assert proj.config.models["judge"] is not None
    assert proj.config.models["judge"].provider == "fake"
    assert proj.config.models["simulator"] is not None


def test_role_override(tmp_path: Path) -> None:
    proj = load_project(_project(tmp_path, BASE), None, Overrides(roles=["candidate=fake:fake-b"]))
    cand = proj.config.models["candidate"]
    assert cand is not None and (cand.provider, cand.model) == ("fake", "fake-b")


def test_unknown_key_suggestion(tmp_path: Path) -> None:
    cfg = {**BASE, "run": {"repeat": 3}}
    with pytest.raises(ConfigError) as exc:
        load_project(_project(tmp_path, cfg))
    assert "unknown key 'repeat'" in str(exc.value)
    assert "did you mean 'repeats'" in str(exc.value)


def test_unknown_profile(tmp_path: Path) -> None:
    write(tmp_path / "profiles" / "nightly.yaml", "{}\n")
    with pytest.raises(ConfigError, match="did you mean 'nightly'"):
        load_project(_project(tmp_path, BASE), "nighty")


def test_unknown_provider_reference(tmp_path: Path) -> None:
    cfg = {**BASE, "models": {"judge": {"provider": "openroutr", "model": "x"}}}
    with pytest.raises(ConfigError, match="unknown provider 'openroutr'"):
        load_project(_project(tmp_path, cfg))


def test_interpolation_env_and_config_refs() -> None:
    raw = {"models": {"judge": {"provider": "p", "model": "m"}}, "x": "${models.judge}"}
    out = interpolate(raw, raw, env={"KEY": "secret"})
    assert out["x"] == "p:m"
    assert interpolate("k=${KEY} m=${MISSING}", raw, env={"KEY": "v"}) == "k=v m="


# ------------------------------------------------------------------ util + registry


def test_deep_merge_replaces_lists() -> None:
    assert deep_merge({"a": [1], "b": {"c": 1}}, {"a": [2], "b": {"d": 2}}) == {
        "a": [2],
        "b": {"c": 1, "d": 2},
    }


def test_normalize_and_extract_json() -> None:
    assert normalize_text("  Hello   World!! ") == "hello world"
    assert extract_json('text ```json\n{"a": 1}\n``` more') == {"a": 1}
    assert extract_json('prefix {"b": [1, 2]} suffix') == {"b": [1, 2]}
    assert extract_json("no json") is None


def test_registry_lookup_suggests() -> None:
    reg: Registry[int] = Registry("thing")
    reg.add("alpha", 1)
    with pytest.raises(ConfigError, match="did you mean 'alpha'"):
        reg.get("alpah")


def test_metric_decorator_registers_spec() -> None:
    from evalkit.registry import METRICS

    @metric("_test_only_metric", suites=("quality",), kind="deterministic")
    def fn(case: Case, obs: object, ctx: object) -> MetricResult:
        """A test metric."""
        return MetricResult(metric="_test_only_metric")

    try:
        spec = METRICS.get("_test_only_metric")
        assert spec.description == "A test metric."
        assert spec.applies_to("quality") and not spec.applies_to("rag")
    finally:
        METRICS._items.pop("_test_only_metric", None)


def test_set_keeps_yaml11_words_as_strings(tmp_path: Path) -> None:
    # regression: `--set run.cache=off` became False and failed validation
    proj = load_project(
        _project(tmp_path, BASE), None, Overrides(sets=["run.cache=off", "run.short_circuit=false"])
    )
    assert proj.config.run.cache == "off" and proj.config.run.short_circuit is False
