from __future__ import annotations

import subprocess


def parse_cuda_device_index(device: str) -> int | None:
    normalized = device.strip().lower()
    if not normalized or normalized == "cpu":
        return None
    if normalized == "cuda":
        return 0
    if normalized.startswith("cuda:"):
        index_text = normalized.split(":", 1)[1]
        if not index_text.isdigit():
            raise ValueError(f"Unsupported CUDA device specifier: {device!r}")
        index = int(index_text)
        if index < 0:
            raise ValueError(f"CUDA device index must be >= 0, got {index}.")
        return index
    raise ValueError(f"Unsupported device specifier: {device!r}")


def query_free_vram_mb(device: str) -> int | None:
    index = parse_cuda_device_index(device)
    if index is None:
        return None

    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if index >= len(values):
        raise ValueError(
            f"Requested CUDA device index {index}, but nvidia-smi returned {len(values)} devices."
        )
    return int(values[index])


__all__ = ["parse_cuda_device_index", "query_free_vram_mb"]
