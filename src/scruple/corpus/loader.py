"""Loading and normalising a corpus (plan §7.2).

Supported inputs: CSV, JSONL, and a folder of plain-text files. Users bring
text; transcription, OCR and PDF parsing are explicitly out of scope (§5).

The input file itself is never modified -- §6 keeps `data/corpus.csv` untouched
and all derived state under `.scruple/`.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ValidationError
from ..hashing import corpus_hash, item_hash, normalise_item_text

TEXT_SUFFIXES = {".txt", ".md"}
MAX_REPORTED_ROWS = 5


@dataclass(frozen=True)
class Item:
    """One coding unit: the text that receives one set of judgements (§4)."""

    id: str
    text: str
    row: Mapping[str, Any]

    @property
    def item_hash(self) -> str:
        """Content hash. Two items with identical text share a cache entry, which
        is why repeated answers like "n/a" cost the backend nothing twice."""
        return item_hash(self.text)


@dataclass(frozen=True)
class Corpus:
    """A validated, normalised collection of items."""

    items: tuple[Item, ...]
    source: str
    text_column: str
    id_column: str | None = None
    columns: tuple[str, ...] = ()

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.items)

    @property
    def hash(self) -> str:
        """Order-independent corpus hash for the run manifest (§7.3)."""
        return corpus_hash(item.item_hash for item in self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Item]:
        return iter(self.items)

    def by_id(self) -> dict[str, Item]:
        return {item.id: item for item in self.items}

    def subset(self, ids: Sequence[str]) -> Corpus:
        """The items with these ids, preserving corpus order."""
        wanted = set(ids)
        return Corpus(
            items=tuple(item for item in self.items if item.id in wanted),
            source=self.source,
            text_column=self.text_column,
            id_column=self.id_column,
            columns=self.columns,
        )

    def length_stats(self) -> dict[str, float]:
        """Character-length distribution, reported by `scruple load` (§12).

        A researcher who sees a p99 of 40,000 characters learns straight away
        that they are in the chunking regime of §10.3, before spending money.
        """
        if not self.items:
            return {}
        lengths = sorted(len(item.text) for item in self.items)
        return {
            "n": float(len(lengths)),
            "min": float(lengths[0]),
            "median": float(statistics.median(lengths)),
            "mean": float(statistics.fmean(lengths)),
            "p95": float(lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))]),
            "max": float(lengths[-1]),
        }


def _build_items(
    rows: Sequence[Mapping[str, Any]],
    *,
    text_column: str,
    id_column: str | None,
) -> tuple[Item, ...]:
    if not rows:
        raise ValidationError("corpus contains no rows")

    if text_column not in rows[0]:
        available = ", ".join(map(str, rows[0].keys()))
        raise ValidationError(
            f"no column {text_column!r} in the corpus",
            hint=f"Available columns: {available}",
        )
    if id_column is not None and id_column not in rows[0]:
        raise ValidationError(f"no id column {id_column!r} in the corpus")

    items: list[Item] = []
    blank: list[int] = []
    seen: dict[str, int] = {}
    width = len(str(len(rows)))

    for index, row in enumerate(rows):
        raw = row.get(text_column)
        text = normalise_item_text("" if raw is None else str(raw))
        if not text:
            blank.append(index + 1)
            continue

        if id_column is None:
            item_id = f"row_{index + 1:0{width}d}"
        else:
            item_id = str(row[id_column]).strip()
            if not item_id:
                raise ValidationError(f"row {index + 1} has an empty {id_column!r}")
        if item_id in seen:
            raise ValidationError(
                f"duplicate item id {item_id!r} at rows {seen[item_id]} and {index + 1}",
                hint="Item ids must be unique; every statistic is computed per item.",
            )
        seen[item_id] = index + 1
        items.append(Item(id=item_id, text=text, row=dict(row)))

    if blank:
        shown = ", ".join(str(r) for r in blank[:MAX_REPORTED_ROWS])
        hidden = len(blank) - MAX_REPORTED_ROWS
        more = f" (and {hidden} more)" if hidden > 0 else ""
        raise ValidationError(
            f"{len(blank)} row(s) have empty {text_column!r}: rows {shown}{more}",
            hint=(
                "Filter them out of the input first. scruple will not drop them silently, "
                "because dropping rows changes the denominator of every statistic it reports."
            ),
        )
    return tuple(items)


def load_csv(path: Path, *, text_column: str, id_column: str | None = None) -> Corpus:
    """Load a CSV or TSV corpus."""
    import pandas as pd

    separator = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    try:
        frame = pd.read_csv(path, sep=separator, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ValidationError(f"could not read {path}: {exc}") from exc

    rows: list[Mapping[str, Any]] = [
        {str(k): v for k, v in record.items()} for record in frame.to_dict(orient="records")
    ]
    return Corpus(
        items=_build_items(rows, text_column=text_column, id_column=id_column),
        source=str(path),
        text_column=text_column,
        id_column=id_column,
        columns=tuple(str(c) for c in frame.columns),
    )


def load_jsonl(path: Path, *, text_column: str, id_column: str | None = None) -> Corpus:
    """Load a JSON Lines corpus, one object per line."""
    rows: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{number} is not valid JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValidationError(f"{path}:{number} is not a JSON object")
            rows.append(parsed)

    columns = tuple(rows[0].keys()) if rows else ()
    return Corpus(
        items=_build_items(rows, text_column=text_column, id_column=id_column),
        source=str(path),
        text_column=text_column,
        id_column=id_column,
        columns=columns,
    )


def load_text_folder(path: Path, *, text_column: str = "text") -> Corpus:
    """Load a folder of plain-text files, one file per item.

    The filename stem becomes the item id, so interview transcripts keep the
    names the researcher already uses for them.
    """
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in TEXT_SUFFIXES)
    if not files:
        raise ValidationError(
            f"no .txt or .md files in {path}",
            hint="A text-folder corpus needs one file per coding unit.",
        )
    rows = [
        {"id": f.stem, text_column: f.read_text(encoding="utf-8"), "source_file": f.name}
        for f in files
    ]
    return Corpus(
        items=_build_items(rows, text_column=text_column, id_column="id"),
        source=str(path),
        text_column=text_column,
        id_column="id",
        columns=("id", text_column, "source_file"),
    )


def load_corpus(path: Path, *, text_column: str = "text", id_column: str | None = None) -> Corpus:
    """Load a corpus, dispatching on the path."""
    if not path.exists():
        raise ValidationError(f"no corpus at {path}")
    if path.is_dir():
        return load_text_folder(path, text_column=text_column)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".tab"}:
        return load_csv(path, text_column=text_column, id_column=id_column)
    if suffix in {".jsonl", ".ndjson"}:
        return load_jsonl(path, text_column=text_column, id_column=id_column)
    raise ValidationError(
        f"unsupported corpus format {suffix!r}",
        hint="Supported: .csv, .tsv, .jsonl, or a folder of .txt/.md files.",
    )
