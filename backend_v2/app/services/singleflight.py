from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class _Entry:
    lock: threading.Lock
    users: int = 0


class KeyedLocks:
    """Deduplicate equal work inside one API process without leaking locks."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._entries: dict[str, _Entry] = {}

    @contextmanager
    def acquire(self, key: str) -> Iterator[None]:
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = _Entry(lock=threading.Lock())
                self._entries[key] = entry
            entry.users += 1
        entry.lock.acquire()
        try:
            yield
        finally:
            entry.lock.release()
            with self._guard:
                entry.users -= 1
                if entry.users == 0:
                    self._entries.pop(key, None)
