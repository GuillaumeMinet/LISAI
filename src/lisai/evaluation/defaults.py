from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Mapping

from lisai.config import load_yaml, settings
from lisai.config.io import deep_merge
from lisai.config.models.inference_defaults import InferenceConfig, InferenceDefaults

INFERENCE_CONFIG_SUFFIXES = (".yml", ".yaml")


class UnsetType:
    def __repr__(self) -> str:
        return "UNSET"


UNSET = UnsetType()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _candidate_paths(path: Path) -> tuple[Path, ...]:
    candidates = [path]
    if not path.suffix:
        candidates.extend(path.with_suffix(suffix) for suffix in INFERENCE_CONFIG_SUFFIXES)
    return tuple(candidates)


def _first_existing_path(candidates: list[Path] | tuple[Path, ...]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _search_roots(*, cwd: Path) -> tuple[Path, ...]:
    roots = [cwd / "configs" / "inference"]
    repo_inference = _repo_root() / "configs" / "inference"
    if repo_inference not in roots:
        roots.append(repo_inference)
    return tuple(roots)


def _available_inference_configs(search_roots: tuple[Path, ...]) -> list[str]:
    available: set[str] = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        for suffix in INFERENCE_CONFIG_SUFFIXES:
            available.update(path.name for path in root.glob(f"*{suffix}") if path.is_file())
    return sorted(available)


def _missing_config_error(config_arg: str, *, search_roots: tuple[Path, ...]) -> FileNotFoundError:
    available = _available_inference_configs(search_roots)
    lines = [f"Inference config not found: {config_arg}"]
    if available:
        lines.append("Available configs:")
        lines.extend(f"  - {config_name}" for config_name in available)
    else:
        lines.append("No inference configs were found under configs/inference.")
    return FileNotFoundError("\n".join(lines))


def resolve_inference_config_path(config_arg: str | Path | None, *, cwd: Path | None = None) -> Path | None:
    base_cwd = Path.cwd() if cwd is None else Path(cwd)
    search_roots = _search_roots(cwd=base_cwd)

    if config_arg is None:
        for root in search_roots:
            resolved = _first_existing_path(_candidate_paths(root / "defaults"))
            if resolved is not None:
                return resolved
        return None

    config_path = Path(config_arg).expanduser()
    resolved = _first_existing_path(_candidate_paths(config_path))
    if resolved is not None:
        return resolved

    if not config_path.is_absolute():
        for root in search_roots:
            resolved = _first_existing_path(_candidate_paths(root / config_path))
            if resolved is not None:
                return resolved

    raise _missing_config_error(str(config_arg), search_roots=search_roots)


def load_inference_config(
    config_arg: str | Path | None = None,
    *,
    cwd: Path | None = None,
) -> tuple[InferenceConfig, Path | None]:
    cfg_path = resolve_inference_config_path(config_arg, cwd=cwd)
    if cfg_path is None:
        return InferenceConfig(), None
    return InferenceConfig.model_validate(load_yaml(cfg_path)), cfg_path


def _merge_value(default: Any, override: Any) -> Any:
    if override is UNSET:
        return deepcopy(default)
    if isinstance(default, Mapping) and isinstance(override, Mapping):
        return deep_merge(dict(default), dict(override))
    return override


def _resolve_task_options(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise KeyError(f"Unknown inference default override(s): {', '.join(unknown)}")
    return {key: _merge_value(default, overrides.get(key, UNSET)) for key, default in defaults.items()}


def _resolve_section_defaults(
    section: Literal["apply", "evaluate"],
    *,
    config: str | Path | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    resolved = InferenceDefaults().model_dump()
    defaults_cfg, _ = load_inference_config(None, cwd=cwd)
    resolved = deep_merge(resolved, defaults_cfg.model_dump(exclude_unset=True))

    if config is None:
        return dict(resolved[section])

    named_cfg, cfg_path = load_inference_config(config, cwd=cwd)
    named_cfg_dict = named_cfg.model_dump(exclude_unset=True)
    if section not in named_cfg_dict:
        raise ValueError(
            f"Inference config '{cfg_path}' does not define a '{section}' section."
        )
    resolved = deep_merge(resolved, named_cfg_dict)
    return dict(resolved[section])


def resolve_apply_options(
    *,
    defaults: InferenceDefaults | None = None,
    defaults_path: str | Path | None = None,
    config: str | Path | None = None,
    cwd: Path | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    if defaults is not None or defaults_path is not None:
        loaded_defaults = load_inference_defaults(defaults_path) if defaults is None else defaults
        section_defaults = loaded_defaults.apply.model_dump()
    else:
        section_defaults = _resolve_section_defaults("apply", config=config, cwd=cwd)
    return _resolve_task_options(section_defaults, overrides)


def resolve_evaluate_options(
    *,
    defaults: InferenceDefaults | None = None,
    defaults_path: str | Path | None = None,
    config: str | Path | None = None,
    cwd: Path | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    if defaults is not None or defaults_path is not None:
        loaded_defaults = load_inference_defaults(defaults_path) if defaults is None else defaults
        section_defaults = loaded_defaults.evaluate.model_dump()
    else:
        section_defaults = _resolve_section_defaults("evaluate", config=config, cwd=cwd)
    return _resolve_task_options(section_defaults, overrides)


def load_inference_defaults(path: str | Path | None = None) -> InferenceDefaults:
    resolved = InferenceDefaults().model_dump()
    cfg_path = Path(path) if path is not None else resolve_inference_config_path(None)
    if cfg_path is not None:
        raw = InferenceConfig.model_validate(load_yaml(cfg_path)).model_dump(exclude_unset=True)
        resolved = deep_merge(resolved, raw)
    return InferenceDefaults.model_validate(resolved)


__all__ = [
    "INFERENCE_CONFIG_SUFFIXES",
    "UNSET",
    "UnsetType",
    "InferenceConfig",
    "InferenceDefaults",
    "load_inference_config",
    "load_inference_defaults",
    "resolve_apply_options",
    "resolve_evaluate_options",
    "resolve_inference_config_path",
]
