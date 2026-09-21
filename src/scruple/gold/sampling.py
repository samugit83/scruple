"""Gold-sample design (plan §8.5).

Uniform random sampling is what the §8.6 guarantee rests on, because it is what
makes the exchangeability assumption true. But 300 uniform items at 3%
prevalence yields nine positive instances, and no statistic computed on nine
items means anything. Most research codes are rare (§4), so that tension is the
normal case rather than an edge case.

It is resolved in two stages:

1. **Uniform stratum.** A uniform random sample from calibration and from test.
   The guarantee rests on this, and it is exact.
2. **Enriched stratum** (optional, per code). Extra draws from a pool of
   likely-positive items, ranked by model probability, so a rare code has enough
   positives to say anything about.

Every sampled item carries the inclusion probability it was drawn with, and all
estimators use inverse-probability weighting over the combined sample. Because
the probabilities are known by construction, validity is preserved.

The enriched supplement is a two-phase design: it is drawn from the pool items
the uniform phase did not already take, so its inclusion probability is
conditional on the realised uniform sample. That is standard practice for
two-phase sampling and it is exact given the first phase, but it makes the
binomial test in §8.6 approximate rather than exact -- which is why
`ThresholdSelection.exact` exists and why the report states which case applies.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..errors import ValidationError

UNIFORM = "uniform"
ENRICHED = "enriched"


@dataclass(frozen=True)
class GoldPlanItem:
    """One item selected for hand-coding, with the design that selected it."""

    item_id: str
    split: str
    stratum: str
    inclusion_probability: float

    @property
    def weight(self) -> float:
        """The IPW weight, 1 / pi."""
        return 1.0 / self.inclusion_probability


@dataclass(frozen=True)
class GoldPlan:
    """A reproducible sampling plan."""

    items: tuple[GoldPlanItem, ...]
    seed: int
    design: str

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(item.item_id for item in self.items)

    def __len__(self) -> int:
        return len(self.items)


def plan_uniform(ids: Sequence[str], n: int, *, split: str, seed: int) -> GoldPlan:
    """A uniform random sample of `n` items, with exact inclusion probabilities.

    `gold` MUST sample with a recorded seed and MUST refuse a hand-picked
    sample (§8.6): the guarantee assumes the gold sample is a random sample of
    the corpus, and a hand-picked one silently breaks it.
    """
    if n < 1:
        raise ValidationError(f"gold sample size must be at least 1, got {n}")
    population = list(dict.fromkeys(ids))
    if not population:
        raise ValidationError(f"the {split} split has no items to sample")
    if n > len(population):
        raise ValidationError(
            f"cannot draw {n} items from a {split} split of {len(population)}",
            hint="Lower gold.n, or load a larger corpus.",
        )

    rng = random.Random(f"{seed}:{split}:uniform")
    chosen = rng.sample(population, n)
    pi = n / len(population)
    return GoldPlan(
        items=tuple(
            GoldPlanItem(item_id=i, split=split, stratum=UNIFORM, inclusion_probability=pi)
            for i in sorted(chosen)
        ),
        seed=seed,
        design=f"uniform({n}/{len(population)})",
    )


def enrichment_pool(
    probabilities: Mapping[str, float | None],
    *,
    pool_fraction: float = 0.2,
    minimum: int = 1,
) -> tuple[str, ...]:
    """The likely-positive pool: the top `pool_fraction` of items by probability.

    Defined by rank rather than by an absolute cut, so the pool has a known size
    whatever the probability distribution looks like -- which is what lets the
    inclusion probability be computed exactly.
    """
    if not 0.0 < pool_fraction <= 1.0:
        raise ValidationError(f"pool_fraction must lie in (0, 1], got {pool_fraction}")
    scored = [(i, p) for i, p in probabilities.items() if p is not None]
    if not scored:
        return ()
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    size = max(minimum, round(pool_fraction * len(scored)))
    return tuple(i for i, _ in scored[:size])


def plan_enriched(
    *,
    pool: Sequence[str],
    already_sampled: Sequence[str],
    n: int,
    split: str,
    seed: int,
    base_inclusion: float,
    code_id: str,
) -> GoldPlan:
    """Supplement an existing uniform sample with likely-positive items.

    Drawn from the pool items the uniform phase did not already take. The
    resulting inclusion probability is conditional on that realised first phase:

        pi_i = pi_uniform + (1 - pi_uniform) * n / |pool not already sampled|
    """
    if n < 1:
        raise ValidationError(f"enrichment size must be at least 1, got {n}")
    if not 0.0 < base_inclusion <= 1.0:
        raise ValidationError("base inclusion probability must lie in (0, 1]")

    taken = set(already_sampled)
    remaining = [i for i in dict.fromkeys(pool) if i not in taken]
    if not remaining:
        raise ValidationError(
            f"no un-sampled items left in the likely-positive pool for {code_id}",
            hint=(
                "The uniform sample already covers the pool; widen it, or accept "
                "the evidence you have."
            ),
        )
    draw = min(n, len(remaining))

    rng = random.Random(f"{seed}:{split}:enriched:{code_id}")
    chosen = rng.sample(remaining, draw)
    conditional = draw / len(remaining)
    pi = base_inclusion + (1.0 - base_inclusion) * conditional
    return GoldPlan(
        items=tuple(
            GoldPlanItem(
                item_id=i,
                split=split,
                stratum=ENRICHED,
                inclusion_probability=min(1.0, pi),
            )
            for i in sorted(chosen)
        ),
        seed=seed,
        design=f"enriched({code_id}, {draw}/{len(remaining)}, base={base_inclusion:.4g})",
    )


def plan_overlap(plan: GoldPlan, n: int, *, seed: int) -> tuple[str, ...]:
    """The subset a second coder also codes, giving the human-human ceiling (§12).

    A code where two humans disagree is a codebook problem, not a model problem,
    and `check` can only say so if this pipeline exists.
    """
    if n < 0:
        raise ValidationError("overlap size must not be negative")
    ids = list(plan.item_ids)
    rng = random.Random(f"{seed}:overlap")
    return tuple(sorted(rng.sample(ids, min(n, len(ids)))))
