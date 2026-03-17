from __future__ import annotations

from pathlib import Path

import lisai.evaluation.cli as evaluation_cli
from lisai.cli import build_parser


def test_apply_cli_parses_run_ref_config_and_overrides(monkeypatch):
    captured = {}

    def fake_run_apply_model(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(evaluation_cli, "run_apply_model", fake_run_apply_model)

    parser = build_parser()
    args = parser.parse_args(
        [
            "apply",
            "Gag/Upsamp/my_model",
            "/data/images",
            "--config",
            "fast_upsamp",
            "--tiling-size",
            "512",
            "--crop-size",
            "200",
        ]
    )
    result = args.handler(args)

    assert result == 0
    assert captured["model_dataset"] == "Gag"
    assert captured["model_subfolder"] == "Upsamp"
    assert captured["model_name"] == "my_model"
    assert captured["data_path"] == Path("/data/images")
    assert captured["config"] == "fast_upsamp"
    assert captured["tiling_size"] == 512
    assert captured["crop_size"] == 200


def test_evaluate_cli_parses_metrics_and_split(monkeypatch):
    captured = {}

    def fake_run_evaluate(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(evaluation_cli, "run_evaluate", fake_run_evaluate)

    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "Gag/Upsamp/my_model",
            "--config",
            "benchmark",
            "--split",
            "val",
            "--metrics",
            "psnr,ssim",
        ]
    )
    result = args.handler(args)

    assert result == 0
    assert captured["dataset_name"] == "Gag"
    assert captured["model_subfolder"] == "Upsamp"
    assert captured["model_name"] == "my_model"
    assert captured["config"] == "benchmark"
    assert captured["split"] == "val"
    assert captured["metrics_list"] == ["psnr", "ssim"]
