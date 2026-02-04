from pathlib import Path
from lisai.config.io import load_yaml, save_yaml
from lisai.config.merge import deep_merge
from lisai.lib.utils import get_paths, misc


def resolve_config(
    experiment_cfg_path: str | Path,
    project_cfg_path: str | Path = "configs/project/project.yaml",
    data_cfg_path: str | Path = "configs/data/data.yaml",
) -> dict:
    """
    Resolve final experiment config:
      project → data → experiment → mode logic
    """

    # Load base configs
    project_cfg = load_yaml(project_cfg_path)
    data_cfg = load_yaml(data_cfg_path)
    exp_cfg = load_yaml(experiment_cfg_path)

    # Merge hierarchy
    cfg = deep_merge(project_cfg, data_cfg)
    cfg = deep_merge(cfg, exp_cfg)

    # Resolve mode logic
    cfg = _resolve_mode(cfg)

    return cfg


def _resolve_mode(cfg: dict) -> dict:
    """
    Apply train / continue_training / retrain logic.
    YAML-only: previous configs are also YAML.
    """
    exp = cfg.setdefault("experiment", {})
    mode = exp.get("mode", "train")
    local = exp.get("local", True)

    if mode not in {"train", "continue_training", "retrain"}:
        raise ValueError(f"Unknown mode: {mode}")

    if mode == "train":
        return cfg

    # Load previous experiment config (YAML!)
    origin_folder = get_paths.get_model_folder(
        local=local, **cfg["load_model"]
    )
    origin_cfg_path = origin_folder / "config_train.yaml"

    origin_cfg = load_yaml(origin_cfg_path)

    if mode == "continue_training":
        exceptions = [
            "experiment",
            "training_prm",
            "load_model",
        ]
    else:  # retrain
        exceptions = [
            "experiment",
            "training_prm",
        ]

    # Override selected keys
    for key in exceptions:
        val = misc.nested_get(cfg, key)
        if val is not None:
            misc.nested_replace(origin_cfg, key, val)

    origin_cfg["experiment"]["mode"] = mode
    origin_cfg["experiment"]["local"] = local

    return origin_cfg
