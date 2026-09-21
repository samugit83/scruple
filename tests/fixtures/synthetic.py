"""Generate synthetic gold datasets with known properties (plan §14.11).

§14.11 asks for a generator producing gold datasets with *specified*
prevalence, agreement and calibration quality, so tests can assert against
known truth rather than against whatever the code currently does.

The coder is parameterised by its two **class-conditional error rates**, not by
an absolute probability level, because those are the quantities §8.6 controls
and they are what "equally good" means across prevalences. Fixing the confident
cluster at, say, 0.93 instead gives a coder with 7% negative-class error at
prevalence 0.50 and 0.4% at 0.05 — the same coder looking very different to the
procedure for reasons that have nothing to do with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scruple.gold import UNIFORM, GoldRecord


@dataclass(frozen=True)
class SyntheticCoder:
    """A calibrated coder of specified quality.

    ``leak`` is the share of true positives it confidently and wrongly calls
    "no"; it is the floor on P(wrong | accepted, gold = positive), because a low
    score is an accepted "no" at every band.

    ``false_positive_rate`` is the corresponding rate for the negative class.
    """

    prevalence: float
    leak: float = 0.01
    false_positive_rate: float = 0.01
    concentration: float = 30.0

    def probabilities(self, gold: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Emit a probability per item, calibrated to the labels it was given.

        Each cluster's mean is the true positive rate *within that cluster*,
        which is what makes the emitted probability mean what it says.
        """
        observed = float(gold.mean()) if gold.size else self.prevalence
        true_positive = (1.0 - self.leak) * observed
        false_positive = self.false_positive_rate * (1.0 - observed)
        false_negative = self.leak * observed
        true_negative = (1.0 - self.false_positive_rate) * (1.0 - observed)

        high = true_positive / max(1e-9, true_positive + false_positive)
        low = false_negative / max(1e-9, false_negative + true_negative)
        high = min(0.995, max(0.005, high))
        low = min(0.995, max(0.005, low))

        wrong_side = np.where(
            gold == 1.0,
            rng.random(gold.size) < self.leak,
            rng.random(gold.size) < self.false_positive_rate,
        )
        in_high_cluster = np.where(gold == 1.0, ~wrong_side, wrong_side)
        means = np.where(in_high_cluster, high, low)
        return np.clip(
            rng.beta(means * self.concentration, (1.0 - means) * self.concentration),
            0.0005,
            0.9995,
        )


def synthetic_corpus(n: int, *, seed: int = 0) -> tuple[list[str], list[str]]:
    """Item ids and texts. The text is filler: nothing reads it but the hasher."""
    rng = np.random.default_rng(seed)
    ids = [f"r{i:05d}" for i in range(n)]
    texts = [
        f"synthetic answer {i} with {int(rng.integers(1, 99))} words of padding" for i in range(n)
    ]
    return ids, texts


def synthetic_gold(
    n: int,
    *,
    coder: SyntheticCoder,
    seed: int = 0,
    second_coder_agreement: float | None = None,
) -> dict[str, np.ndarray]:
    """Labels, probabilities, and optionally a second coder's labels.

    Returns arrays rather than records, for tests that work directly with the
    statistics layer.
    """
    rng = np.random.default_rng(seed)
    gold = (rng.random(n) < coder.prevalence).astype(np.float64)
    out = {"gold": gold, "probabilities": coder.probabilities(gold, rng)}
    if second_coder_agreement is not None:
        agrees = rng.random(n) < second_coder_agreement
        out["second_coder"] = np.where(agrees, gold, 1.0 - gold)
    return out


def gold_records(
    item_ids: list[str],
    labels: np.ndarray,
    *,
    code_id: str,
    split: str,
    coder: str = "coder_1",
    inclusion_probability: float = 1.0,
    seed: int = 42,
) -> list[GoldRecord]:
    """Build gold records for tests that exercise the store or `check`."""
    return [
        GoldRecord(
            item_id=item_id,
            code_id=code_id,
            label=int(label),
            coder=coder,
            timestamp="2026-09-20T10:00:00+00:00",
            split=split,
            stratum=UNIFORM,
            inclusion_probability=inclusion_probability,
            # §6: always false for gold. A coder who has seen the model's answer
            # is no longer an independent second rater.
            model_visible=False,
            sample_seed=seed,
        )
        for item_id, label in zip(item_ids, labels, strict=True)
    ]
