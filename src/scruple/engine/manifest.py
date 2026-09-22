"""Run manifests (§7.3).

Every number in the validation report must be reproducible from files on disk
(§6, design rule 4), so the manifest records what actually ran -- including the
exact model version string the API returned, not the one that was requested.
The report quotes these verbatim.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import __version__


def new_run_id(now: datetime | None = None) -> str:
    """A sortable, human-readable run id."""
    moment = now or datetime.now(UTC)
    return moment.strftime("%Y%m%dT%H%M%SZ")


@dataclass
class Manifest:
    """The provenance record for one run."""

    run_id: str
    created_at: str
    scruple_version: str
    purpose: str
    """Why this run happened: "run" scored the whole corpus, "check" scored only
    the gold sample, "try" only a dev sample. Without it, `export` could pick up
    a check's partial probabilities and quietly write a mostly-empty coded.csv."""

    backend: str
    model_version: str
    codebook_hash: str
    code_hashes: dict[str, str]
    corpus_hash: str
    corpus_path: str
    corpus_file_hash: str
    splits_seed: int
    chunking: dict[str, Any]
    item_count: int
    code_count: int
    unit_count: int
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    failures: int
    retries: int
    cache_hits: int
    cache_misses: int
    wall_clock_seconds: float
    python_version: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        path.write_text(payload, encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Manifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def build_manifest(**kwargs: Any) -> Manifest:
    """Construct a manifest, filling in the fields scruple knows about itself."""
    kwargs.setdefault("run_id", new_run_id())
    kwargs.setdefault("created_at", datetime.now(UTC).isoformat(timespec="seconds"))
    kwargs.setdefault("scruple_version", __version__)
    kwargs.setdefault("purpose", "run")
    return Manifest(**kwargs)
