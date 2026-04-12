from __future__ import annotations

import io
from datetime import timedelta
from pathlib import Path

import lisai.evaluation.cli as evaluation_cli
from lisai.cli import build_parser
from lisai.infra.fs.run_naming import parse_run_dir_name
from lisai.runs.io import write_run_metadata_atomic
from lisai.runs.scanner import scan_runs
from lisai.runs.schema import RunMetadata, utc_now


class InteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return True


class NonInteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return False


def _write_metadata(
    run_dir: Path,
    *,
    run_id: str,
    dataset: str,
    model_subfolder: str,
    status: str = "completed",
    last_heartbeat_at=None,
):
    run_name, run_index = parse_run_dir_name(run_dir.name)
    now = utc_now()
    heartbeat = last_heartbeat_at if last_heartbeat_at is not None else now
    created_at = heartbeat - timedelta(minutes=5)
    updated_at = heartbeat
    ended_at = None if status == "running" else heartbeat

    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "run_name": run_name,
        "run_index": run_index,
        "dataset": dataset,
        "model_subfolder": model_subfolder,
        "status": status,
        "closed_cleanly": status != "running",
        "created_at": created_at,
        "updated_at": updated_at,
        "ended_at": ended_at,
        "last_heartbeat_at": heartbeat,
        "last_epoch": 3,
        "max_epoch": 10,
        "best_val_loss": 0.4,
        "path": f"datasets/{dataset}/models/{model_subfolder}/{run_dir.name}",
        "group_path": None if "/" not in model_subfolder else model_subfolder.split("/", 1)[1],
    }
    write_run_metadata_atomic(run_dir, RunMetadata.model_validate(payload))


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


def test_apply_cli_accepts_best_or_last_both(monkeypatch):
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
            "--best-or-last",
            "both",
        ]
    )
    result = args.handler(args)

    assert result == 0
    assert captured["best_or_last"] == "both"


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


def test_evaluate_cli_accepts_best_or_last_both(monkeypatch):
    captured = {}

    def fake_run_evaluate(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(evaluation_cli, "run_evaluate", fake_run_evaluate)

    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "Gag/Upsamp/my_model",
            "--best-or-last",
            "both",
        ]
    )
    result = args.handler(args)

    assert result == 0
    assert captured["best_or_last"] == "both"


def test_evaluate_cli_accepts_run_name_and_index_selector(monkeypatch, tmp_path):
    captured = {}
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "models" / "Upsamp" / "resume_me_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G7AAA",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    monkeypatch.setattr(evaluation_cli, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "run_evaluate", lambda **kwargs: captured.update(kwargs))

    parser = build_parser()
    args = parser.parse_args(["evaluate", "resume_me", "0", "--config", "benchmark"])
    result = args.handler(args)

    assert result == 0
    assert captured["dataset_name"] == "Gag"
    assert captured["model_subfolder"] == "Upsamp"
    assert captured["model_name"] == "resume_me_00"
    assert captured["config"] == "benchmark"


def test_evaluate_cli_ambiguous_selector_allows_interactive_line_selection(monkeypatch, tmp_path):
    captured = {}
    datasets_root = tmp_path / "datasets"
    now = utc_now()
    _write_metadata(
        datasets_root / "Actin" / "models" / "HDN" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G7AAB",
        dataset="Actin",
        model_subfolder="HDN",
        last_heartbeat_at=now,
    )
    _write_metadata(
        datasets_root / "Gag" / "models" / "Upsamp" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G7AAC",
        dataset="Gag",
        model_subfolder="Upsamp",
        last_heartbeat_at=now - timedelta(minutes=1),
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(evaluation_cli, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "run_evaluate", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(evaluation_cli.sys, "stdin", InteractiveInput("02\n"))
    monkeypatch.setattr(evaluation_cli.sys, "stdout", stdout)
    monkeypatch.setattr(evaluation_cli.sys, "stderr", stderr)

    parser = build_parser()
    args = parser.parse_args(["evaluate", "duplicate", "0"])
    result = args.handler(args)

    assert result == 0
    assert captured["dataset_name"] == "Gag"
    assert captured["model_subfolder"] == "Upsamp"
    assert captured["model_name"] == "duplicate_00"
    assert "Multiple matching runs found:" in stdout.getvalue()
    assert "Select run number from '#'" in stdout.getvalue()


def test_evaluate_cli_ambiguous_selector_requires_extra_filters_when_non_interactive(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    _write_metadata(
        datasets_root / "Actin" / "models" / "HDN" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G7AAD",
        dataset="Actin",
        model_subfolder="HDN",
    )
    _write_metadata(
        datasets_root / "Gag" / "models" / "Upsamp" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G7AAE",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    called = {"value": False}
    monkeypatch.setattr(evaluation_cli, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "run_evaluate", lambda **kwargs: called.update({"value": True}))
    monkeypatch.setattr(evaluation_cli.sys, "stdin", NonInteractiveInput(""))
    monkeypatch.setattr(evaluation_cli.sys, "stdout", stdout)
    monkeypatch.setattr(evaluation_cli.sys, "stderr", stderr)

    parser = build_parser()
    args = parser.parse_args(["evaluate", "duplicate", "0"])
    result = args.handler(args)

    assert result == 1
    assert called["value"] is False
    assert "Multiple matching runs found:" in stdout.getvalue()
    assert "Rerun with --dataset/--subfolder or with --run-id to disambiguate." in stderr.getvalue()
