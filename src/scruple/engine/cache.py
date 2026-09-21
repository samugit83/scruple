"""The probability cache (plan §9.4).

Keyed per (item x code), not per item, so editing one code definition re-runs
only that code. That is what makes the `try` loop of §12 feel instant, and that
loop is the main daily UX.

**Hashes only -- never text.** `gold.jsonl` and `review.jsonl` hold raw text and
are personal data at rest; the cache deliberately is not, so `scruple purge
--gold` can remove the personal data while leaving the expensive work intact
(§6, §9.4).

**Failures are not cached.** A `None` means the backend could not answer, and
caching that would make `run --retry-failed` a no-op (§10.2).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS probabilities (
    key           TEXT PRIMARY KEY,
    probability   REAL NOT NULL,
    created_at    TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS meta (
    name  TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class CacheStats:
    """Hit/miss counters, reported in run output so users see why iteration is cheap."""

    hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


@dataclass
class Cache:
    """SQLite-backed probability store."""

    path: Path
    stats: CacheStats = field(default_factory=CacheStats)
    _connection: sqlite3.Connection | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.executescript(_SCHEMA)
        connection.execute(
            "INSERT OR IGNORE INTO meta (name, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        connection.commit()
        self._connection = connection

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("cache is closed")
        return self._connection

    def get(self, key: str) -> float | None:
        """The cached probability, or ``None`` when the key is absent.

        Absent and "failed" are the same here on purpose: failures are never
        stored, so a miss is always something worth asking the backend about.
        """
        row = self.connection.execute(
            "SELECT probability FROM probabilities WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return float(row[0])

    def get_many(self, keys: Iterable[str]) -> dict[str, float]:
        """Look up many keys in one statement. O(1) per key via the primary key."""
        wanted = list(keys)
        found: dict[str, float] = {}
        # SQLite caps variables per statement; chunk well below the usual 32k.
        for start in range(0, len(wanted), 500):
            window = wanted[start : start + 500]
            placeholders = ",".join("?" * len(window))
            rows = self.connection.execute(
                f"SELECT key, probability FROM probabilities WHERE key IN ({placeholders})",
                window,
            ).fetchall()
            found.update({str(k): float(v) for k, v in rows})
        self.stats.hits += len(found)
        self.stats.misses += len(wanted) - len(found)
        return found

    def put(self, key: str, probability: float | None, *, input_tokens: int = 0) -> None:
        """Store a probability. ``None`` is ignored -- see the module docstring."""
        if probability is None:
            return
        self.connection.execute(
            "INSERT OR REPLACE INTO probabilities (key, probability, created_at, input_tokens) "
            "VALUES (?, ?, ?, ?)",
            (
                key,
                float(probability),
                datetime.now(UTC).isoformat(timespec="seconds"),
                input_tokens,
            ),
        )
        self.stats.writes += 1

    def put_many(self, values: Mapping[str, float | None], *, input_tokens: int = 0) -> None:
        for key, probability in values.items():
            self.put(key, probability, input_tokens=input_tokens)
        self.commit()

    def commit(self) -> None:
        self.connection.commit()

    def __len__(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM probabilities").fetchone()
        return int(row[0])

    def purge(self) -> int:
        """Delete every cached probability, returning how many were removed."""
        removed = len(self)
        self.connection.execute("DELETE FROM probabilities")
        self.commit()
        return removed

    def close(self) -> None:
        if self._connection is not None:
            self._connection.commit()
            self._connection.close()
            self._connection = None

    def __enter__(self) -> Cache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
