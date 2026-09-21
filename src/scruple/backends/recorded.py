"""A backend that replays recorded probabilities (no network).

Two jobs:

* `examples/` ships a synthetic corpus **and** its probabilities, so the whole
  pipeline runs offline with no API key (§13). First-run experience decides
  adoption, and "get an API key first" loses most of it.
* Tests get a deterministic backend without touching the network, which §14.2
  forbids in every tier.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from ..codebook import Code
from ..errors import BackendError
from ..hashing import item_hash
from .base import BaseBackend


class RecordedBackend(BaseBackend):
    """Replays a table of ``{item_hash: {code_id: probability}}``.

    Keying on the item hash rather than the item id means a recording stays
    valid when ids are regenerated, and that identical texts share an entry.
    """

    name = "recorded"

    def __init__(
        self,
        table: dict[str, dict[str, float | None]],
        *,
        version: str = "recorded-1",
        strict: bool = True,
    ) -> None:
        super().__init__()
        self.table = table
        self.version = version
        self.strict = strict

    @classmethod
    def from_path(cls, path: Path, *, strict: bool = True) -> RecordedBackend:
        if not path.exists():
            raise BackendError(
                f"no recorded probabilities at {path}",
                hint="The offline example ships them; a real run needs a live backend.",
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BackendError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise BackendError(f"{path} must map item hashes to code probabilities")
        return cls(
            table=data.get("probabilities", data),
            version=str(data.get("model_version", "recorded-1"))
            if "model_version" in data
            else "recorded-1",
            strict=strict,
        )

    def model_version(self) -> str:
        return self.version

    def _score_batch(self, state: str, codes: Sequence[Code]) -> dict[str, float | None]:
        self.usage.calls += 1
        entry = self.table.get(item_hash(state))
        if entry is None:
            if self.strict:
                raise BackendError(
                    "no recorded probability for an item the recording does not cover",
                    hint=(
                        "The recording was made for a different corpus. Re-record it, "
                        "or use a live backend."
                    ),
                )
            return {code.id: None for code in codes}

        missing = [code.id for code in codes if code.id not in entry]
        if missing and self.strict:
            # Returning None here would look like a backend failure and quietly
            # push every affected item into the abstention band.
            raise BackendError(
                f"the recording does not cover code(s): {', '.join(missing)}",
                hint="Codes added since the recording was made need a live backend.",
            )
        return {code.id: entry.get(code.id) for code in codes}
