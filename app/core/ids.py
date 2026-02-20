"""Application-wide identifier utilities."""

from __future__ import annotations

import secrets
import threading
import time

_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
_LAST_MILLIS = 0
_COUNTER = 0
_LOCK = threading.Lock()


def _to_base36(value: int) -> str:
    if value == 0:
        return "0"
    chars: list[str] = []
    current = value
    while current:
        current, remainder = divmod(current, 36)
        chars.append(_ALPHABET[remainder])
    return "".join(reversed(chars))


def _random_alnum(length: int) -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(max(length, 0)))


def generate_cuid(length: int = 24) -> str:
    """Generate a collision-resistant lowercase identifier with a `c` prefix."""
    global _LAST_MILLIS, _COUNTER

    now_millis = int(time.time() * 1000)
    with _LOCK:
        if now_millis == _LAST_MILLIS:
            _COUNTER += 1
        else:
            _LAST_MILLIS = now_millis
            _COUNTER = 0
        counter = _COUNTER

    time_part = _to_base36(now_millis)
    counter_part = _to_base36(counter).rjust(4, "0")
    body_len = max(length - 1, 8)
    static_part = f"{time_part}{counter_part}"
    random_part = _random_alnum(max(body_len - len(static_part), 0))
    body = f"{static_part}{random_part}"[:body_len]
    return f"c{body}"

