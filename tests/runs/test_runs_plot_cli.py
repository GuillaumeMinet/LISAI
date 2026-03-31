from __future__ import annotations

import io
from datetime import timedelta
from pathlib import Path

import lisai.cli as root_cli
import lisai.runs.cli as runs_cli
from lisai.infra.fs.run_naming import parse_run_dir_name
from lisai.runs.io import write_run_metadata_atomic
from lisai.runs.scanner import scan_runs
from lisai.runs.schema import RunMetadata, utc_now


class NonInteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return False


def _write_metadata(
    run_dir: Path,
    *,
    run_id: str,
    dataset: str,
    model_subfolder: str,
    architecture: str | None = None,
):
    run_name, run_index = parse_run_dir_name(run_dir.name)
    now = utc_now()
    created_at = now - timedelta(minutes=5)
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "run_name": run_name,
        "run_index": run_index,
        "dataset": dataset,
        "model_subfolder": model_subfolder,
        "status": "completed",
        "closed_cleanly": True,
        "created_at": created_at,
        "updated_at": now,
        "ended_at": now,
        "last_heartbeat_at": now,
        "last_epoch": 3,
        "max_epoch": 10,
        "best_val_loss": 0.4,
        "path": f"datasets/{dataset}/models/{model_subfolder}/{run_dir.name}",
        "group_path": None if "/" not in model_subfolder else model_subfolder.split("/", 1)[1],
    }
    if architecture is not None:
        payload["training_signature"] = {"architecture": architecture}
    write_run_metadata_atomic(run_dir, RunMetadata.model_validate(payload))


def test_runs_plot_uses_paths_loss_file_and_passes_parsed_data(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "models" / "Upsamp" / "resume_me_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69GB100",
        dataset="Gag",
        model_subfolder="Upsamp",
        architecture="upsamp",
    )

    custom_loss = run_dir / "custom_loss.dat"
    custom_loss.write_text(
        "Epoch Train_loss Val_loss\n"
        "0 1.2 1.5\n"
        "1 0.9 1.1\n",
        encoding="utf-8",
    )

    class _FakePaths:
        def loss_file_path(self, *, run_dir):
            return Path(run_dir) / "custom_loss.dat"

    captured = {}
    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(runs_cli, "Paths", _FakePaths)

    def _fake_plot(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runs_cli, "_plot_loss_history", _fake_plot)

    exit_code = root_cli.main(["runs", "plot", "resume_me", "0", "--dataset", "Gag"])

    assert exit_code == 0
    assert captured["loss_file"] == custom_loss
    assert captured["hdn_layout"] is False
    assert tuple(captured["headers"]) == ("Epoch", "Train_loss", "Val_loss")
    assert captured["data"].shape == (2, 3)


def test_runs_plot_accepts_dataset_subfolder_runref_selector(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "models" / "Upsamp_base" / "SubA" / "resume_me_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69GB105",
        dataset="Gag",
        model_subfolder="Upsamp_base/SubA",
        architecture="upsamp",
    )
    (run_dir / "loss.txt").write_text(
        "Epoch Train_loss Val_loss\n"
        "0 1.2 1.5\n",
        encoding="utf-8",
    )

    captured = {}
    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))

    def _fake_plot(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runs_cli, "_plot_loss_history", _fake_plot)

    exit_code = root_cli.main(["runs", "plot", "Gag/Upsamp_base/SubA/resume_me_00"])

    assert exit_code == 0
    assert captured["run"].run_dir == run_dir.resolve()
    assert captured["data"].shape == (1, 3)


def test_runs_plot_detects_hdn_and_enables_two_subplot_layout(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "models" / "HDN" / "hdn_run_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69GB101",
        dataset="Gag",
        model_subfolder="HDN",
        architecture="hdn",
    )
    (run_dir / "loss.txt").write_text(
        "Epoch Train_loss Val_loss Recons_Loss KL_Loss\n"
        "0 1.2 1.5 1.1 0.1\n"
        "1 0.9 1.1 0.8 0.1\n",
        encoding="utf-8",
    )

    captured = {}
    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))

    def _fake_plot(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runs_cli, "_plot_loss_history", _fake_plot)

    exit_code = root_cli.main(["runs", "plot", "hdn_run", "0", "--dataset", "Gag"])

    assert exit_code == 0
    assert captured["hdn_layout"] is True
    assert captured["data"].shape == (2, 5)


def test_runs_plot_ambiguous_selector_requires_disambiguation_when_non_interactive(
    monkeypatch,
    tmp_path,
    capsys,
):
    datasets_root = tmp_path / "datasets"
    _write_metadata(
        datasets_root / "Actin" / "models" / "HDN" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69GB102",
        dataset="Actin",
        model_subfolder="HDN",
    )
    _write_metadata(
        datasets_root / "Gag" / "models" / "Upsamp" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69GB103",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    called = {"value": False}
    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(runs_cli.sys, "stdin", NonInteractiveInput(""))

    def _fake_plot(**_kwargs):
        called["value"] = True
        return 0

    monkeypatch.setattr(runs_cli, "_plot_loss_history", _fake_plot)

    exit_code = root_cli.main(["runs", "plot", "duplicate", "0"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert called["value"] is False
    assert "Multiple matching runs found:" in captured.out
    assert "Rerun with --dataset/--subfolder or with --run-id to disambiguate." in captured.err


def test_runs_plot_reports_missing_loss_file(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "models" / "Upsamp" / "resume_me_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69GB104",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(runs_cli, "_plot_loss_history", lambda **_kwargs: 0)

    exit_code = root_cli.main(["runs", "plot", "resume_me", "0", "--dataset", "Gag"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Loss file not found:" in captured.err


def test_is_non_interactive_backend_marks_agg_and_inline():
    assert runs_cli._is_non_interactive_backend("Agg") is True
    assert runs_cli._is_non_interactive_backend("module://matplotlib_inline.backend_inline") is True
    assert runs_cli._is_non_interactive_backend("QtAgg") is False


def test_plot_loss_history_saves_when_backend_is_non_interactive(tmp_path, monkeypatch):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "models" / "Upsamp" / "resume_me_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69GB106",
        dataset="Gag",
        model_subfolder="Upsamp",
    )
    run = scan_runs(datasets_root).runs[0]
    loss_file = run_dir / "loss.txt"
    loss_file.write_text(
        "Epoch Train_loss Val_loss\n"
        "0 1.0 1.2\n",
        encoding="utf-8",
    )

    class _FakeAxes:
        def plot(self, *_args, **_kwargs):
            return None

        def set_title(self, *_args, **_kwargs):
            return None

        def set_xlabel(self, *_args, **_kwargs):
            return None

        def set_ylabel(self, *_args, **_kwargs):
            return None

        def grid(self, *_args, **_kwargs):
            return None

        def legend(self, *_args, **_kwargs):
            return None

    class _FakeFigure:
        def __init__(self):
            self.saved_path = None

        def suptitle(self, *_args, **_kwargs):
            return None

        def tight_layout(self):
            return None

        def savefig(self, path, **_kwargs):
            self.saved_path = Path(path)
            self.saved_path.write_text("fake-png", encoding="utf-8")

    class _FakePyplot:
        def __init__(self):
            self.show_called = False
            self.closed = []
            self.figure = _FakeFigure()

        def subplots(self, *_args, **_kwargs):
            return self.figure, _FakeAxes()

        def show(self):
            self.show_called = True

        def close(self, fig):
            self.closed.append(fig)

    fake_plt = _FakePyplot()
    monkeypatch.setattr(
        runs_cli,
        "_prepare_matplotlib_for_plot",
        lambda stderr=None: (fake_plt, "Agg", False),
    )
    opened = {"value": False}
    monkeypatch.setattr(
        runs_cli,
        "_try_open_path",
        lambda _path: opened.update({"value": True}) or True,
    )

    import numpy as np

    stderr = io.StringIO()
    exit_code = runs_cli._plot_loss_history(
        run=run,
        loss_file=loss_file,
        headers=("Epoch", "Train_loss", "Val_loss"),
        data=np.array([[0.0, 1.0, 1.2]], dtype=float),
        hdn_layout=False,
        stderr=stderr,
    )

    assert exit_code == 0
    assert fake_plt.show_called is False
    assert fake_plt.figure.saved_path == run_dir / "loss_plot.png"
    assert fake_plt.figure.saved_path.exists()
    assert fake_plt.closed == [fake_plt.figure]
    assert "non-interactive" in stderr.getvalue().lower()
    assert "opened fallback plot image" in stderr.getvalue().lower()
    assert opened["value"] is True


def test_plot_loss_history_uses_show_with_interactive_backend(tmp_path, monkeypatch):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "models" / "Upsamp" / "resume_me_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69GB107",
        dataset="Gag",
        model_subfolder="Upsamp",
    )
    run = scan_runs(datasets_root).runs[0]
    loss_file = run_dir / "loss.txt"
    loss_file.write_text(
        "Epoch Train_loss Val_loss\n"
        "0 1.0 1.2\n",
        encoding="utf-8",
    )

    class _FakeAxes:
        def plot(self, *_args, **_kwargs):
            return None

        def set_title(self, *_args, **_kwargs):
            return None

        def set_xlabel(self, *_args, **_kwargs):
            return None

        def set_ylabel(self, *_args, **_kwargs):
            return None

        def grid(self, *_args, **_kwargs):
            return None

        def legend(self, *_args, **_kwargs):
            return None

    class _FakeFigure:
        def suptitle(self, *_args, **_kwargs):
            return None

        def tight_layout(self):
            return None

        def savefig(self, *_args, **_kwargs):
            return None

    class _FakePyplot:
        def __init__(self):
            self.show_called = False
            self.closed = []

        def subplots(self, *_args, **_kwargs):
            return _FakeFigure(), _FakeAxes()

        def show(self):
            self.show_called = True

        def close(self, fig):
            self.closed.append(fig)

    fake_plt = _FakePyplot()
    monkeypatch.setattr(
        runs_cli,
        "_prepare_matplotlib_for_plot",
        lambda stderr=None: (fake_plt, "QtAgg", True),
    )

    import numpy as np

    stderr = io.StringIO()
    exit_code = runs_cli._plot_loss_history(
        run=run,
        loss_file=loss_file,
        headers=("Epoch", "Train_loss", "Val_loss"),
        data=np.array([[0.0, 1.0, 1.2]], dtype=float),
        hdn_layout=False,
        stderr=stderr,
    )

    assert exit_code == 0
    assert fake_plt.show_called is True
    assert "non-interactive" not in stderr.getvalue().lower()
