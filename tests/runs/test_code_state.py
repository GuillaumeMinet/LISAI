from __future__ import annotations

import subprocess

from lisai.runs.code_state import collect_code_state


class _Completed:
    def __init__(self, stdout: str = "", returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


def test_collect_code_state_reads_git_and_package_version(monkeypatch, tmp_path):
    outputs = {
        ("rev-parse", "HEAD"): _Completed("abc1234def5678\n"),
        ("branch", "--show-current"): _Completed("main\n"),
        ("config", "--get", "remote.origin.url"): _Completed(
            "git@github.com:GuillaumeMinet/LISAI.git\n"
        ),
        ("status", "--porcelain"): _Completed(" M src/lisai/runs/schema.py\n"),
    }

    def fake_run(command, **kwargs):
        assert command[0] == "git"
        return outputs[tuple(command[1:])]

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("lisai.runs.code_state.version", lambda package: "0.1.0")

    code = collect_code_state(tmp_path)

    assert code.git_commit == "abc1234def5678"
    assert code.git_branch == "main"
    assert code.git_dirty is True
    assert code.git_remote == "git@github.com:GuillaumeMinet/LISAI.git"
    assert code.lisai_version == "0.1.0"


def test_collect_code_state_reports_clean_tree(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        if command[1:] == ["status", "--porcelain"]:
            return _Completed("")
        return _Completed("value\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("lisai.runs.code_state.version", lambda package: "0.1.0")

    code = collect_code_state(tmp_path)

    assert code.git_dirty is False


def test_collect_code_state_tolerates_git_failure(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("lisai.runs.code_state.version", lambda package: "0.1.0")

    code = collect_code_state(tmp_path)

    assert code.git_commit is None
    assert code.git_branch is None
    assert code.git_dirty is None
    assert code.git_remote is None
    assert code.lisai_version == "0.1.0"
