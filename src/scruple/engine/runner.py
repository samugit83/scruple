"""Run orchestration (plan §10).

The engine is a pure function of (corpus, codebook, backend) -> probabilities
(§6, design rule 2). It makes no decisions: thresholds and abstention live
downstream in `stats/`, so re-thresholding never requires re-running the model.

Cache reads and writes happen on the calling thread, and only misses are handed
to the worker pool. That keeps SQLite single-threaded without a lock, and makes
"zero backend calls on a warm cache" structurally true rather than a hope.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..backends import Backend, question_tokens
from ..codebook import Code, Codebook
from ..config import ProjectConfig
from ..corpus import Corpus, Item
from ..corpus.chunking import aggregate, chunk_text, estimate_tokens
from ..errors import BackendError
from ..hashing import cache_key, file_hash, item_hash
from .budget import CostEstimate, estimate_cost, require_confirmation
from .cache import Cache
from .manifest import Manifest, build_manifest, new_run_id

logger = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]


@dataclass(frozen=True)
class Unit:
    """One piece of text actually sent to the backend.

    A short item is one unit. A long item becomes several, and their scores are
    aggregated back to one probability per (item, code) -- calibration is always
    fitted on the aggregate, never on passages (§10.3).
    """

    item_id: str
    index: int
    text: str

    @property
    def text_hash(self) -> str:
        return item_hash(self.text)


@dataclass
class RunResult:
    """Raw probabilities plus everything the manifest and report need."""

    run_id: str
    scores: dict[str, dict[str, float | None]]
    manifest: Manifest
    failures: list[tuple[str, str]] = field(default_factory=list)
    chunked_items: int = 0

    def probabilities_for(self, code_id: str) -> dict[str, float | None]:
        return {item_id: codes.get(code_id) for item_id, codes in self.scores.items()}

    @property
    def failed_item_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item_id for item_id, _ in self.failures))


def build_units(item: Item, config: ProjectConfig) -> list[Unit]:
    """Split one item into the units that will be scored (§10.3)."""
    if estimate_tokens(item.text) <= config.chunking.max_tokens:
        return [Unit(item_id=item.id, index=0, text=item.text)]
    chunks = chunk_text(
        item.text,
        max_tokens=config.chunking.max_tokens,
        overlap_tokens=config.chunking.overlap_tokens,
    )
    return [Unit(item_id=item.id, index=i, text=chunk) for i, chunk in enumerate(chunks)]


def plan_units(corpus: Corpus, config: ProjectConfig) -> list[Unit]:
    return [unit for item in corpus for unit in build_units(item, config)]


class Engine:
    """Scores a corpus against a codebook, with caching and bounded concurrency."""

    def __init__(
        self,
        *,
        backend: Backend,
        config: ProjectConfig,
        cache: Cache,
        progress: ProgressFn | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.cache = cache
        self.progress = progress

    def estimate(self, corpus: Corpus, codebook: Codebook) -> CostEstimate:
        """Project the cost of an uncached run over `corpus` (§10.1)."""
        units = plan_units(corpus, self.config)
        return estimate_cost(
            texts=[unit.text for unit in units],
            codebook=codebook,
            engine=self.config.engine,
            question_tokens=sum(question_tokens(code) for code in codebook),
        )

    def run(
        self,
        corpus: Corpus,
        codebook: Codebook,
        *,
        corpus_path: Path | None = None,
        splits_seed: int | None = None,
        approved: bool = True,
        run_id: str | None = None,
    ) -> RunResult:
        """Score every (item, code) pair, using the cache wherever possible."""
        started = time.monotonic()
        units = plan_units(corpus, self.config)
        codes = list(codebook)

        if not approved:
            require_confirmation(
                self.estimate(corpus, codebook), self.config.engine, approved=approved
            )

        model_version = self.backend.model_version()
        # unit_scores[(item_id, index)][code_id] -> probability or None
        unit_scores: dict[tuple[str, int], dict[str, float | None]] = {
            (unit.item_id, unit.index): {} for unit in units
        }

        # Phase 1: resolve everything the cache already knows, on this thread.
        pending: list[tuple[Unit, list[Code]]] = []
        for unit in units:
            keys = {
                code.id: cache_key(
                    backend=self.backend.name,
                    model_version=model_version,
                    item=unit.text_hash,
                    code_id=code.id,
                    code=code.code_hash,
                )
                for code in codes
            }
            cached = self.cache.get_many(keys.values())
            missing: list[Code] = []
            for code in codes:
                key = keys[code.id]
                if key in cached:
                    unit_scores[(unit.item_id, unit.index)][code.id] = cached[key]
                else:
                    missing.append(code)
            if missing:
                pending.append((unit, missing))

        # Phase 2: ask the backend only about the misses, and only once per
        # distinct text. A survey corpus is full of repeated short answers
        # ("n/a", "no comment"); paying for each copy is the user's money.
        # Units sharing a text hash share their cache keys exactly, so their
        # missing code sets are identical by construction.
        failures: list[tuple[str, str]] = []
        groups: dict[str, tuple[Unit, list[Code], list[Unit]]] = {}
        for unit, missing in pending:
            existing = groups.get(unit.text_hash)
            if existing is None:
                groups[unit.text_hash] = (unit, missing, [unit])
            else:
                existing[2].append(unit)

        total = len(groups)
        done = 0
        if groups:
            workers = min(self.config.engine.concurrency, total)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._score_unit, representative, missing): (
                        representative,
                        missing,
                        members,
                    )
                    for representative, missing, members in groups.values()
                }
                for future in futures:
                    representative, missing, members = futures[future]
                    try:
                        scored = future.result()
                    except BackendError as exc:
                        # §10.2: a run completes with partial results and reports
                        # the failure count. It never silently drops an item.
                        logger.warning("scoring %s failed: %s", representative.item_id, exc)
                        scored = {code.id: None for code in missing}
                    for code in missing:
                        probability = scored.get(code.id)
                        for member in members:
                            unit_scores[(member.item_id, member.index)][code.id] = probability
                            if probability is None:
                                failures.append((member.item_id, code.id))
                        if probability is not None:
                            self.cache.put(
                                cache_key(
                                    backend=self.backend.name,
                                    model_version=model_version,
                                    item=representative.text_hash,
                                    code_id=code.id,
                                    code=code.code_hash,
                                ),
                                probability,
                            )
                    done += 1
                    if self.progress is not None:
                        self.progress(done, total)
            self.cache.commit()

        # Phase 3: aggregate passages back to one probability per (item, code).
        scores: dict[str, dict[str, float | None]] = {}
        units_by_item: dict[str, list[int]] = {}
        for unit in units:
            units_by_item.setdefault(unit.item_id, []).append(unit.index)
        for item_id, indexes in units_by_item.items():
            scores[item_id] = {
                code.id: aggregate(
                    [unit_scores[(item_id, i)].get(code.id) for i in indexes],
                    rule=self.config.chunking.aggregation,
                )
                for code in codes
            }

        usage = getattr(self.backend, "usage", None)
        chunked = sum(1 for indexes in units_by_item.values() if len(indexes) > 1)
        manifest = build_manifest(
            run_id=run_id or new_run_id(),
            backend=self.backend.name,
            model_version=self.backend.model_version(),
            codebook_hash=codebook.hash,
            code_hashes=codebook.code_hashes,
            corpus_hash=corpus.hash,
            corpus_path=str(corpus_path or corpus.source),
            corpus_file_hash=(
                file_hash(corpus_path) if corpus_path and corpus_path.is_file() else ""
            ),
            splits_seed=splits_seed if splits_seed is not None else self.config.splits.seed,
            chunking={
                "max_tokens": self.config.chunking.max_tokens,
                "overlap_tokens": self.config.chunking.overlap_tokens,
                "aggregation": self.config.chunking.aggregation,
            },
            item_count=len(corpus),
            code_count=len(codes),
            unit_count=len(units),
            calls=usage.calls if usage else 0,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            cost_usd=usage.cost_usd if usage else 0.0,
            failures=len(failures),
            retries=usage.retries if usage else 0,
            cache_hits=self.cache.stats.hits,
            cache_misses=self.cache.stats.misses,
            wall_clock_seconds=round(time.monotonic() - started, 3),
        )

        return RunResult(
            run_id=manifest.run_id,
            scores=scores,
            manifest=manifest,
            failures=failures,
            chunked_items=chunked,
        )

    def _score_unit(self, unit: Unit, codes: Sequence[Code]) -> dict[str, float | None]:
        return self.backend.score(unit.text, codes)


def save_run(result: RunResult, runs_dir: Path) -> Path:
    """Persist raw probabilities and the manifest (§6, §7.3).

    Raw probabilities are stored, never just decisions, so every number in the
    report is reproducible and re-thresholding is free.
    """
    import json

    directory = runs_dir / result.run_id
    directory.mkdir(parents=True, exist_ok=True)
    result.manifest.save(directory / "manifest.json")
    (directory / "probabilities.json").write_text(
        json.dumps(result.scores, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if result.failures:
        (directory / "failures.json").write_text(
            json.dumps([{"item_id": i, "code_id": c} for i, c in result.failures], indent=2) + "\n",
            encoding="utf-8",
        )
    return directory
