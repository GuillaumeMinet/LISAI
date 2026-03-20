from __future__ import annotations

import re
import secrets
import time

_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def _encode_crockford_base32(value: int, length: int) -> str:
    if value < 0:
        raise ValueError("Cannot encode negative values.")

    chars = ["0"] * length
    for idx in range(length - 1, -1, -1):
        chars[idx] = _ULID_ALPHABET[value & 0x1F]
        value >>= 5
    return "".join(chars)


def generate_run_id(*, timestamp_ms: int | None = None) -> str:
    """
    Generate a ULID-like identifier (26 chars, Crockford base32, time-sortable).
    """
    ts_ms = int(time.time() * 1000) if timestamp_ms is None else int(timestamp_ms)
    if ts_ms < 0 or ts_ms > (1 << 48) - 1:
        raise ValueError("timestamp_ms must fit in 48 bits.")

    randomness = secrets.randbits(80)
    return _encode_crockford_base32(ts_ms, 10) + _encode_crockford_base32(randomness, 16)


def is_valid_run_id(value: str) -> bool:
    return bool(_ULID_RE.fullmatch(value.strip().upper()))


__all__ = ["generate_run_id", "is_valid_run_id"]
