from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .models import DataConfig, ProjectConfig
from .io.yaml import load_yaml, save_yaml


class AttrDict(dict):
    """dict with attribute access, so '{paths.roots.data_dir}' works with str.format."""
    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any):
        self[key] = value


def _to_attr(x: Any) -> Any:
    """
    Recursively convert dictionaries into AttrDict to allow attribute-style
    access required by ``str.format`` templates (e.g. ``{paths.roots.data_dir}``).

    - dict  → AttrDict with recursive conversion
    - list  → elements recursively converted
    - other → returned unchanged

    Used to build the formatting context for project path templates.
    """
    if isinstance(x, dict):
        ad = AttrDict()
        for k, v in x.items():
            ad[k] = _to_attr(v)
        return ad
    if isinstance(x, list):
        return [_to_attr(v) for v in x]
    return x


class Settings:
    """
    Single source of truth for project configuration.

    It loads and validates project/data YAML files via Pydantic models,
    builds the template formatting context, and exposes resolved
    configuration needed by infrastructure components such as Paths.
    """
    def __init__(self):
        self.PROJECT_ROOT = self._find_project_root(anchor="configs")
        self.CONFIGS_ROOT = self.PROJECT_ROOT / "configs"

        self._local_yaml_path = self.CONFIGS_ROOT / "local_config.yml"
        self._project_yaml_path = self.CONFIGS_ROOT / "project_config.yml"
        self._data_yaml_path = self.CONFIGS_ROOT / "data_config.yml"

        self._infra_cfg = self._load_or_setup_infrastructure()

        project_raw = self._load_required(self._project_yaml_path)
        data_raw = self._load_required(self._data_yaml_path)

        self.project: ProjectConfig = ProjectConfig.model_validate(project_raw)
        self.data: DataConfig = DataConfig.model_validate(data_raw)

        self._ctx = self._build_context()

    def _find_project_root(self, anchor: str = "configs") -> Path:
        current_path = Path(__file__).resolve()
        for parent in [current_path] + list(current_path.parents):
            if (parent / anchor).exists():
                return parent
        raise FileNotFoundError(f"Could not find project root from {current_path} (missing '{anchor}' folder).")

    def _load_required(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Critical config missing: {path}")
        return load_yaml(path)

    def _load_or_setup_infrastructure(self) -> dict:
        if self._local_yaml_path.exists():
            return load_yaml(self._local_yaml_path)

        print("\n" + "=" * 60)
        print(" LISAI - FIRST TIME SETUP")
        print("=" * 60)
        default_root = "E:/dl_monalisa" if os.name == "nt" else "/data/dl_monalisa"
        user_input = input(f"Enter absolute path to Data Root [default: {default_root}]: ").strip()
        data_root = user_input if user_input else default_root

        new_config = {"infrastructure": {"data_root": str(Path(data_root).resolve())}}
        self._local_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        save_yaml(new_config, self._local_yaml_path)
        print(f"✅ Saved to {self._local_yaml_path}\n")
        return new_config

    def _build_context(self) -> AttrDict:
        infra = self._infra_cfg.get("infrastructure", {})
        data_root = Path(infra.get("data_root")).resolve()
        code_dir = self.PROJECT_ROOT.resolve()

        ctx = AttrDict(
            data_root=str(data_root),
            code_dir=str(code_dir),
            paths=AttrDict(
                roots=AttrDict(),
                templates=AttrDict(),
            ),
        )

        # Provide code_dir for templates
        ctx.paths.roots.code_dir = str(code_dir)

        # Resolve roots (only depend on infra)
        for key, tmpl in (self.project.paths.roots or {}).items():
            value = tmpl.format(**ctx)
            value = str(Path(os.path.normpath(value)).resolve())
            ctx.paths.roots[key] = value

        # Store templates as-is (experiment-dependent keys can't be resolved yet)
        for key, tmpl in (self.project.paths.templates or {}).items():
            ctx.paths.templates[key] = tmpl

        return ctx


    # ==========================================
    # PUBLIC API
    # ==========================================
    
    @property
    def NAMING(self):
        # expose naming conventions
        return self.project.naming

    def resolve_path(self, template: str, **kwargs) -> Path:
        merged = AttrDict(self._ctx)
        for k, v in kwargs.items():
            merged[k] = _to_attr(v)
        s = template.format(**merged)
        return Path(os.path.normpath(s))

    def get_template_path(self, key: str, **kwargs) -> Path:
        tmpl = self._ctx.paths.templates.get(key)
        if tmpl is None:
            raise KeyError(f"Template key '{key}' not found in project.paths.templates")
        return self.resolve_path(tmpl, **kwargs)

    def get_data_filename(self, fmt: str, data_type: str, **kwargs) -> Path:
        fmt_cfg = self.data.format.get(fmt)
        if not fmt_cfg:
            raise ValueError(f"Unknown format: {fmt}")
        template = fmt_cfg.get(data_type)
        if not template:
            raise ValueError(f"Unknown type '{data_type}' for format '{fmt}'")

        filename = self.resolve_path(template, **kwargs)
        return filename


settings = Settings()
