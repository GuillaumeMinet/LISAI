from __future__ import annotations

from types import SimpleNamespace

from lisai.infra.paths import Paths


def test_paths_exposes_project_root(tmp_path):
    settings = SimpleNamespace(PROJECT_ROOT=tmp_path)

    assert Paths(settings).project_root() == tmp_path.resolve()
