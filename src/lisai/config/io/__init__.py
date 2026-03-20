from __future__ import annotations

from .merge import deep_merge
from .yaml import load_yaml, save_yaml


def resolve_config(*args, **kwargs):
    from .resolver import resolve_config as _resolve_config

    return _resolve_config(*args, **kwargs)


def resolve_config_dict(*args, **kwargs):
    from .resolver import resolve_config_dict as _resolve_config_dict

    return _resolve_config_dict(*args, **kwargs)


def prune_config_for_saving(*args, **kwargs):
    from .resolver import prune_config_for_saving as _prune_config_for_saving

    return _prune_config_for_saving(*args, **kwargs)


__all__ = [
    "deep_merge",
    "resolve_config",
    "resolve_config_dict",
    "prune_config_for_saving",
    "load_yaml",
    "save_yaml",
]