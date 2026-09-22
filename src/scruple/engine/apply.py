"""Turning probabilities into the delivered dataset (§7.4, §8.9).

Three provenances end up in `coded.csv`, and the report must be able to say how
much of the data came from each (§8.9):

* ``auto``   -- the engine decided, inside a certified band;
* ``human``  -- a person decided, in `gold` or `review`;
* ``abstain_unreviewed`` -- the engine declined and nobody has looked yet.

The class-conditional guarantee of §8.6 applies **only** to the ``auto``
subset, and the report says so explicitly rather than letting a reader assume
it covers everything.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from ..codebook import Codebook
from ..corpus import Corpus
from .calibration import CalibrationRecord


class Source(StrEnum):
    """Where one cell of the coded dataset came from (§7.4)."""

    AUTO = "auto"
    HUMAN = "human"
    ABSTAIN_UNREVIEWED = "abstain_unreviewed"
    NOT_AUTOMATABLE = "not_automatable"


@dataclass(frozen=True)
class Decision:
    """One (item, code) cell of the delivered dataset."""

    item_id: str
    code_id: str
    label: int | None
    probability: float | None
    source: Source


@dataclass
class AppliedCoding:
    """The delivered dataset, plus what still needs a person."""

    decisions: dict[str, dict[str, Decision]]
    codes: tuple[str, ...]

    def abstentions(self) -> list[Decision]:
        """Items the engine refused to code, with their probabilities.

        This file is the one nobody else produces (§0.5). It is not a failure
        list; it is the part of the corpus that genuinely needs judgement.
        """
        return [
            decision
            for per_code in self.decisions.values()
            for decision in per_code.values()
            if decision.source is Source.ABSTAIN_UNREVIEWED
        ]

    def counts_by_source(self) -> dict[str, int]:
        counts = dict.fromkeys((s.value for s in Source), 0)
        for per_code in self.decisions.values():
            for decision in per_code.values():
                counts[decision.source.value] += 1
        return counts

    def coverage_for(self, code_id: str) -> float:
        """Share of items this code actually decided, from any source."""
        cells = [per_code[code_id] for per_code in self.decisions.values() if code_id in per_code]
        if not cells:
            return 0.0
        return sum(1 for c in cells if c.label is not None) / len(cells)


def apply_calibration(
    *,
    corpus: Corpus,
    codebook: Codebook,
    record: CalibrationRecord,
    scores: Mapping[str, Mapping[str, float | None]],
    human_labels: Mapping[tuple[str, str], int] | None = None,
) -> AppliedCoding:
    """Decide every (item, code) pair using the fitted bands.

    A human label always wins: if a person has judged this item for this code --
    in the gold sample or in review -- their judgement is the answer, and the
    machine's probability is recorded beside it rather than over it.
    """
    human = dict(human_labels or {})
    decisions: dict[str, dict[str, Decision]] = {}

    for item in corpus:
        per_code: dict[str, Decision] = {}
        for code in codebook:
            probability = scores.get(item.id, {}).get(code.id)
            calibration = record.codes.get(code.id)

            decided_by_human = human.get((item.id, code.id))
            if decided_by_human is not None:
                per_code[code.id] = Decision(
                    item_id=item.id,
                    code_id=code.id,
                    label=int(decided_by_human),
                    probability=probability,
                    source=Source.HUMAN,
                )
                continue

            if calibration is None or not calibration.usable or calibration.pair is None:
                # Not certified: scruple will not put a number here at all. A
                # guess on an uncertified code is exactly the output this project
                # exists to avoid producing.
                per_code[code.id] = Decision(
                    item_id=item.id,
                    code_id=code.id,
                    label=None,
                    probability=probability,
                    source=Source.NOT_AUTOMATABLE,
                )
                continue

            if probability is None:
                per_code[code.id] = Decision(
                    item_id=item.id,
                    code_id=code.id,
                    label=None,
                    probability=None,
                    source=Source.ABSTAIN_UNREVIEWED,
                )
                continue

            label = calibration.pair.decide(probability)
            per_code[code.id] = Decision(
                item_id=item.id,
                code_id=code.id,
                label=label,
                probability=probability,
                source=Source.AUTO if label is not None else Source.ABSTAIN_UNREVIEWED,
            )
        decisions[item.id] = per_code

    return AppliedCoding(decisions=decisions, codes=codebook.ids)
