from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_local_cfg = ROOT / "configs" / "local_config.yml"
_created_local_cfg = False
if not _local_cfg.exists():
    _local_cfg.parent.mkdir(parents=True, exist_ok=True)
    _local_cfg.write_text("infrastructure:\n  data_root: .\n", encoding="utf-8")
    _created_local_cfg = True


def pytest_sessionfinish(session, exitstatus):
    if _created_local_cfg and _local_cfg.exists():
        _local_cfg.unlink()
