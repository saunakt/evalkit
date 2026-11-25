"""Pydantic config models + loading / merging / interpolation (Plan.md §13)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from evalkit.util import ConfigError, deep_merge, did_you_mean, get_path, set_path

ROLES = ("judge", "simulator", "perturb", "generator", "sut", "candidate", "fallback")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Price(Strict):
    input_per_mtok: float
    output_per_mtok: float


class ProviderConfig(Strict):
    type: str
    base_url: str | None = None
    api_key: str | None = None
    free: bool | None = None
    supports_json: bool = False
    prices: dict[str, Price] = {}
    headers: dict[str, str] = {}
    timeout_s: float = 60.0
    max_retries: int = 5
    retry_base_s: float = 0.5
    max_concurrency: int | None = None
    default_model: str | None = None
    script: str | None = None


class RoleModel(Strict):
    provider: str
    model: str = ""


class RunConfig(Strict):
    repeats: int = 1
    parallel: int = 4
    seed: int = 42
    cache: Literal["off", "read_write", "replay_only"] = "read_write"
    cache_path: str = ".evalkit/cache.sqlite"
    now: str | None = None
    reset_between_cases: bool = True
    short_circuit: bool = True
    max_cost_usd: float | None = 20.0


class SuiteConfig(Strict):
    enabled: bool = False
    dataset: str | None = None
    metrics: list[str] | Literal["all", "default"] = "default"
    exclude: list[str] = []
    tags: list[str] = []
    params: dict[str, Any] = {}
    target: str | None = None


class CalibrationConfig(Strict):
    min_kappa: float = 0.6
    min_fail_recall: float = 0.8
    allow_uncalibrated: bool = False


class ReportsConfig(Strict):
    formats: list[Literal["html", "md", "json", "junit", "csv"]] = [
        "html",
        "md",
        "json",
        "junit",
        "csv",
    ]
    per_suite: bool = True
    evidence_pack: bool = False
    dir: str = "reports"
    release: str | None = None


class ObserveConfig(Strict):
    langfuse: bool = False
    trace_url_template: str | None = None
    langfuse_host: str | None = None


class EvalkitConfig(Strict):
    project: str = "evalkit"
    system: dict[str, Any] = {}
    risks: list[dict[str, Any]] = []
    target: str = "targets/fake.yaml"
    providers: dict[str, ProviderConfig] = {}
    models: dict[str, RoleModel | None] = {}
    run: RunConfig = RunConfig()
    splits: list[Literal["dev", "heldout"]] = ["dev"]
    engines: dict[str, str] = {}
    calibration: CalibrationConfig = CalibrationConfig()
    suites: dict[str, SuiteConfig] = {}
    reports: ReportsConfig = ReportsConfig()
    observe: ObserveConfig = ObserveConfig()
    thresholds: str = "thresholds.yaml"
    requirements: str = "requirements.yaml"
    results_dir: str = "results"
    judges_dir: str = "judges"


class Bound(Strict):
    min: float | None = None
    max: float | None = None


class RegressionConfig(Strict):
    baseline: str | None = "auto"
    max_slice_drop: float = 0.05
    require_significance: bool = True


class Thresholds(Strict):
    defaults: dict[str, float] = {"pass_rate": 0.9}
    must_pass_tags: list[str] = []
    suites: dict[str, dict[str, float | Bound]] = {}
    regression: RegressionConfig = RegressionConfig()


class Requirement(Strict):
    id: str
    regulation: str = ""
    text: str = ""
    cases: list[str] = []
    metrics: list[str] = []
    owner: str = ""


class JudgeConfig(Strict):
    id: str
    version: int = 1
    style: Literal["binary", "rubric", "reference", "pairwise", "geval", "claims"] = "binary"
    criterion: str
    role: str = "judge"
    provider: str | None = None
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    few_shot: list[dict[str, Any]] = []
    rubric: dict[int, str] | None = None
    description: str = ""


_STRICT_MODELS: tuple[type[BaseModel], ...] = (
    Price,
    ProviderConfig,
    RoleModel,
    RunConfig,
    SuiteConfig,
    CalibrationConfig,
    ReportsConfig,
    ObserveConfig,
    EvalkitConfig,
    Bound,
    RegressionConfig,
    Thresholds,
    Requirement,
    JudgeConfig,
)
KNOWN_KEYS = sorted({name for m in _STRICT_MODELS for name in m.model_fields})


# --------------------------------------------------------------------------- overrides


@dataclass
class Overrides:
    """CLI flags applied after evalkit.yaml and the profile."""

    suites: list[str] | None = None
    metrics: list[str] | None = None
    tags: list[str] | None = None
    split: str | None = None
    repeats: int | None = None
    provider: str | None = None
    roles: list[str] = field(default_factory=list)
    sets: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in (None, [])}


@dataclass
class Project:
    """A fully loaded configuration plus its location."""

    config: EvalkitConfig
    root: Path
    profile: str | None
    overrides: Overrides
    thresholds: Thresholds
    raw: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def path(self, rel: str | Path) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def enabled_suites(self) -> list[str]:
        return [name for name, sc in self.config.suites.items() if sc.enabled]


# --------------------------------------------------------------------------- loading


def read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: expected a mapping at top level")
    return data


_VAR = re.compile(r"\$\{([^}]+)\}")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def interpolate(value: Any, raw: dict[str, Any], env: dict[str, str] | None = None) -> Any:
    """Replace `${ENV_VAR}` (missing → "") and `${dotted.config.path}` recursively."""
    env = dict(os.environ) if env is None else env
    if isinstance(value, dict):
        return {k: interpolate(v, raw, env) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate(v, raw, env) for v in value]
    if not isinstance(value, str):
        return value

    def repl(m: re.Match[str]) -> str:
        name = m.group(1).strip()
        if _ENV_NAME.match(name):
            return env.get(name, "")
        ref = get_path(raw, name)
        if isinstance(ref, dict) and "provider" in ref:
            return f"{ref['provider']}:{ref.get('model', '')}"
        return "" if ref is None else str(ref)

    return _VAR.sub(repl, value)


def validation_message(
    source: str, exc: ValidationError, known_keys: list[str] | None = None
) -> str:
    keys = KNOWN_KEYS if known_keys is None else known_keys
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        msg = err["msg"]
        if err["type"] == "extra_forbidden":
            key = str(err["loc"][-1])
            msg = f"unknown key {key!r}{did_you_mean(key, keys)}"
        lines.append(f"{source}: {loc}: {msg}")
    return "\n".join(lines)


def parse_model(
    model: type[Any], data: Any, source: str, known_keys: list[str] | None = None
) -> Any:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(validation_message(source, exc, known_keys)) from exc


def _parse_value(text: str) -> Any:
    """YAML scalar from --set, but YAML 1.1 words (on/off/yes/no) stay strings."""
    if text.strip().lower() in ("on", "off", "yes", "no", "y", "n"):
        return text.strip()
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(raw: dict[str, Any], ov: Overrides) -> tuple[dict[str, Any], list[str]]:
    """Apply CLI overrides to the merged raw config; returns (raw, notes)."""
    raw = deep_merge(raw, {})
    notes: list[str] = []
    for item in ov.sets:
        key, sep, val = item.partition("=")
        if not sep:
            raise ConfigError(f"--set expects key.path=value, got {item!r}")
        set_path(raw, key.strip(), _parse_value(val))
    if ov.repeats is not None:
        set_path(raw, "run.repeats", ov.repeats)
    if ov.split:
        raw["splits"] = [s.strip() for s in ov.split.split(",") if s.strip()]
    if ov.provider:
        _map_all_roles(raw, ov.provider)
    for spec in ov.roles:
        _override_role(raw, spec)
    if ov.suites is not None:
        suites = raw.setdefault("suites", {})
        for name in list(suites):
            suites[name] = {**(suites[name] or {}), "enabled": name in ov.suites}
        for name in ov.suites:
            suites.setdefault(name, {"enabled": True})
    for name, sc in (raw.get("suites") or {}).items():
        if sc is None:
            raw["suites"][name] = sc = {}
        if ov.tags is not None:
            sc["tags"] = list(ov.tags)
    return raw, notes


def _map_all_roles(raw: dict[str, Any], provider: str) -> None:
    providers = raw.setdefault("providers", {})
    if provider not in providers:
        if provider == "fake":
            providers["fake"] = {"type": "fake"}
        else:
            hint = did_you_mean(provider, sorted(providers))
            raise ConfigError(f"--provider {provider!r} is not configured{hint}")
    default_model = providers[provider].get("default_model")
    models = raw.setdefault("models", {})
    for role in ROLES:
        cur = models.get(role)
        if cur is None and role in ("candidate", "fallback"):
            continue
        model = default_model or (cur or {}).get("model") or provider
        if provider == "fake":
            model = (cur or {}).get("model") if role in ("candidate", "fallback") else "fake"
            model = model or "fake"
        models[role] = {"provider": provider, "model": model}


def _override_role(raw: dict[str, Any], spec: str) -> None:
    role, sep, target = spec.partition("=")
    prov, sep2, model = target.partition(":")
    if not sep or not sep2 or role not in ROLES:
        raise ConfigError(f"--role expects <role>=<provider>:<model> with role in {ROLES}")
    providers = raw.setdefault("providers", {})
    if prov == "fake" and "fake" not in providers:
        providers["fake"] = {"type": "fake"}
    raw.setdefault("models", {})[role] = {"provider": prov, "model": model}


def load_project(
    config_path: str | Path = "evalkit.yaml",
    profile: str | None = None,
    overrides: Overrides | None = None,
) -> Project:
    """evalkit.yaml → profiles/<name>.yaml (deep merge) → CLI flags; then validate."""
    ov = overrides or Overrides()
    cfg_path = Path(config_path).resolve()
    root = cfg_path.parent
    raw = read_yaml(cfg_path)
    if profile:
        prof_path = root / "profiles" / f"{profile}.yaml"
        if not prof_path.exists():
            names = sorted(p.stem for p in (root / "profiles").glob("*.yaml"))
            raise ConfigError(f"unknown profile {profile!r}{did_you_mean(profile, names)}")
        raw = deep_merge(raw, read_yaml(prof_path))
    raw, notes = apply_overrides(raw, ov)
    resolved = interpolate(raw, raw)
    config = parse_model(EvalkitConfig, resolved, str(cfg_path.name))
    thr_path = root / config.thresholds
    thresholds = (
        parse_model(Thresholds, read_yaml(thr_path), config.thresholds)
        if thr_path.exists()
        else Thresholds()
    )
    _check_references(config)
    return Project(config, root, profile, ov, thresholds, resolved, notes)


def _check_references(cfg: EvalkitConfig) -> None:
    errors = []
    for role, rm in cfg.models.items():
        if role not in ROLES:
            errors.append(f"models: unknown role {role!r}{did_you_mean(role, ROLES)}")
        elif rm is not None and rm.provider not in cfg.providers:
            hint = did_you_mean(rm.provider, sorted(cfg.providers))
            errors.append(f"models.{role}: unknown provider {rm.provider!r}{hint}")
    if errors:
        raise ConfigError("\n".join(errors))


def load_judge(path: str | Path) -> JudgeConfig:
    cfg: JudgeConfig = parse_model(JudgeConfig, read_yaml(path), str(path))
    return cfg


def load_requirements(path: str | Path) -> list[Requirement]:
    p = Path(path)
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    items = data.get("requirements", []) if isinstance(data, dict) else data
    return [parse_model(Requirement, item, str(p)) for item in items]
