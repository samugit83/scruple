"""Generate the synthetic `vaccine_survey` example (plan §13).

Wholly synthetic. §14.11 forbids committing real personal data or a real
corpus, and an example shipping real survey answers would be exactly the thing
this tool exists to help people handle carefully.

It writes a corpus, a codebook, recorded probabilities from a simulated coder,
a frozen split and a pre-coded gold sample, so the full pipeline runs offline
with no API key. First-run experience decides adoption, and "get an API key
first" loses most of it.

Code quality is varied deliberately so the example demonstrates all four
verdicts, including the two failures -- a tool that only ever shows success is
not demonstrating the thing that makes it worth using.
"""

from __future__ import annotations

import csv
import json
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path

from scruple.corpus import assign_splits
from scruple.hashing import item_hash, normalise_item_text

OUT = Path(__file__).resolve().parent.parent / "examples" / "vaccine_survey"
N = 1200
SEED = 20260921
GOLD_N = 600
OVERLAP_N = 150


@dataclass(frozen=True)
class CodeSpec:
    """One code, and how well the simulated backend does on it.

    Quality is given as the two class-conditional error rates, because those are
    what §8.6 controls. Parameterising by "the mean probability the coder emits"
    instead produces a coder that is either perfect or useless with almost
    nothing in between, which demonstrates nothing.
    """

    definition: str
    yes: tuple[str, ...]
    no: tuple[str, ...]
    prevalence: float
    leak: float
    """Share of true positives the backend confidently and wrongly calls 'no'."""
    false_positive_rate: float
    """Share of true negatives it confidently and wrongly calls 'yes'."""
    human_agreement: float
    """How often a second human coder agrees with the first."""


CODES: dict[str, CodeSpec] = {
    "access_barrier": CodeSpec(
        definition=(
            "The respondent cites a practical obstacle to attending: time, transport, "
            "clinic opening hours, distance or cost."
        ),
        yes=(
            "the clinic closes at five and I work two jobs",
            "there is no bus out here and I do not drive",
            "could not get the time off work for it",
            "the nearest centre is forty miles away",
            "the booking line was engaged every time I rang",
            "I would have had to pay for parking twice over",
            "my shifts never line up with their opening hours",
            "childcare makes a weekday appointment impossible",
        ),
        no=("I just do not think it is necessary",),
        prevalence=0.22,
        leak=0.005,
        false_positive_rate=0.005,
        human_agreement=0.96,
    ),
    "distrust_pharma": CodeSpec(
        definition=(
            "The respondent expresses distrust of pharmaceutical companies or their motives."
        ),
        yes=(
            "the drug companies only care about profit",
            "I do not trust big pharma, full stop",
            "they rushed it out to make money",
            "there is too much money in it for me to believe them",
            "the manufacturers have lied before and got away with it",
            "follow the money and you have your answer",
        ),
        no=("my doctor explained it and I was happy",),
        prevalence=0.16,
        leak=0.005,
        false_positive_rate=0.005,
        human_agreement=0.95,
    ),
    "no_recommendation": CodeSpec(
        definition=("The respondent says no clinician or trusted person recommended it to them."),
        yes=(
            "nobody ever suggested I should have it",
            "my GP never brought it up",
            "no one told me I was even eligible",
            "not one person in the surgery mentioned it",
            "I heard nothing from anybody about it",
        ),
        no=("my nurse recommended it twice",),
        prevalence=0.13,
        leak=0.01,
        false_positive_rate=0.01,
        human_agreement=0.93,
    ),
    "procrastination": CodeSpec(
        definition="The respondent intended to attend but had not yet got round to it.",
        yes=(
            "I keep meaning to book it and forgetting",
            "just have not got round to it yet",
            "it is on my list, honestly it is",
            "I will do it next month, probably",
            "kept putting it off for no good reason",
        ),
        no=("I decided against it deliberately",),
        prevalence=0.18,
        leak=0.08,
        false_positive_rate=0.05,
        human_agreement=0.88,
    ),
    "religious_objection": CodeSpec(
        definition="The respondent cites a religious or faith-based reason for declining.",
        yes=(
            "it goes against my faith",
            "my church teaches against it",
            "I leave these things in God's hands",
        ),
        no=("no religious reason, I simply forgot",),
        # Deliberately rare: at this prevalence a 320-item gold sample holds
        # roughly ten positives, so `check` must refuse to certify it (§8.5).
        prevalence=0.018,
        leak=0.02,
        false_positive_rate=0.01,
        human_agreement=0.95,
    ),
    "dignity_violation": CodeSpec(
        definition=(
            "The respondent describes being treated without dignity or respect by health staff."
        ),
        yes=(
            "they talked over me like I was not there",
            "I felt patronised for the whole visit",
            "was made to feel stupid for asking a question",
            "the receptionist was dismissive and rude",
            "nobody would explain anything to me properly",
        ),
        no=("the staff were perfectly polite",),
        # Interpretive rather than concrete: the literature reports LLMs doing
        # well on concrete themes and poorly on interpretive ones, and humans
        # disagree here too. This code should come back NOT_AUTOMATABLE.
        prevalence=0.09,
        leak=0.30,
        false_positive_rate=0.22,
        human_agreement=0.72,
    ),
}

OPENERS = (
    "honestly",
    "to be fair",
    "look",
    "well",
    "if I am honest",
    "put it this way",
    "truthfully",
)
CLOSERS = (
    "that is about the size of it",
    "nothing else to add really",
    "make of that what you will",
    "it is complicated",
    "hard to explain any better than that",
    "anyway, that is my reason",
)


def _sample_text(rng: random.Random, applied: dict[str, bool]) -> str:
    """Compose one survey answer from the fragments its codes imply."""
    parts = [rng.choice(CODES[code].yes) for code, on in applied.items() if on]
    if not parts:
        parts = [rng.choice(CODES[rng.choice(list(CODES))].no)]
    rng.shuffle(parts)
    if rng.random() < 0.55:
        parts.insert(0, rng.choice(OPENERS))
    if rng.random() < 0.45:
        parts.append(rng.choice(CLOSERS))
    joined = ", ".join(parts)
    return normalise_item_text(joined[0].upper() + joined[1:] + rng.choice((".", ".", "!")))


def _noisy(rng: random.Random, truth: bool, accuracy: float) -> int:
    """A second rater who agrees with the reference `accuracy` of the time.

    Only the *second* coder is noisy. In the real workflow the primary coder's
    judgements are the reference the model is validated against, and the second
    coder exists to measure how much two trained people differ on the same code
    -- the human ceiling of §12. Making both noisy would mean no coder could
    ever be certified at alpha = 5%, which is true but demonstrates nothing.
    """
    return int(truth) if rng.random() < accuracy else int(not truth)


def main() -> None:
    rng = random.Random(SEED)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / ".scruple").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    truth: list[dict[str, bool]] = []
    seen: set[str] = set()
    attempts = 0
    while len(rows) < N and attempts < N * 50:
        attempts += 1
        applied = {code: rng.random() < spec.prevalence for code, spec in CODES.items()}
        text = _sample_text(rng, applied)
        if text in seen:
            continue  # keep the corpus varied, like real free text
        seen.add(text)
        rows.append(
            {
                "respondent_id": f"R{len(rows) + 1:04d}",
                "age_band": rng.choice(("18-39", "40-64", "65+")),
                "response": text,
            }
        )
        truth.append(applied)

    corpus_path = OUT / "data" / "corpus.csv"
    with corpus_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["respondent_id", "age_band", "response"])
        writer.writeheader()
        writer.writerows(rows)

    # Recorded probabilities from a calibrated coder of the specified quality.
    #
    # For each (item, code) the coder lands in a confident-yes or confident-no
    # cluster, landing in the wrong one at the specified per-class error rate.
    # Each cluster's mean is then set to the true positive rate *within that
    # cluster*, which is what makes the emitted probability mean what it says --
    # the property the whole project depends on.
    observed = {code: sum(1 for t in truth if t[code]) / len(truth) for code in CODES}
    table: dict[str, dict[str, float]] = {}
    for row, applied in zip(rows, truth, strict=True):
        entry = table.setdefault(item_hash(row["response"]), {})
        for code, is_on in applied.items():
            spec = CODES[code]
            prevalence = observed[code]
            true_positives = (1.0 - spec.leak) * prevalence
            false_positives = spec.false_positive_rate * (1.0 - prevalence)
            false_negatives = spec.leak * prevalence
            true_negatives = (1.0 - spec.false_positive_rate) * (1.0 - prevalence)

            in_yes_cluster = (
                rng.random() >= spec.leak if is_on else rng.random() < spec.false_positive_rate
            )
            if in_yes_cluster:
                mean = true_positives / max(1e-9, true_positives + false_positives)
            else:
                mean = false_negatives / max(1e-9, false_negatives + true_negatives)
            mean = min(0.995, max(0.005, mean))
            entry[code] = round(
                min(0.9995, max(0.0005, rng.betavariate(mean * 30, (1.0 - mean) * 30))), 4
            )
    (OUT / "probabilities.json").write_text(
        json.dumps(
            {"model_version": "example-recorded-1", "probabilities": table},
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # Codebook.
    lines = [
        "# The codebook for the synthetic worked example.",
        "#",
        "# Two of these codes are designed to fail, because a tool that only ever",
        "# shows success is not demonstrating what makes it worth using:",
        "#   religious_objection  too rare to reach 15 positives in the gold sample",
        "#   dignity_violation    interpretive; even the two coders disagree",
        "version: 1",
        "codes:",
    ]
    for code, spec in CODES.items():
        lines.append(f"  {code}:")
        lines.append("    type: noul")
        lines.append("    definition: >")
        lines += [f"      {chunk}" for chunk in textwrap.wrap(spec.definition, 72)]
        lines.append("    examples_yes:")
        lines += [f'      - "{ex}"' for ex in spec.yes[:2]]
        lines.append("    examples_no:")
        lines += [f'      - "{ex}"' for ex in spec.no[:1]]
    (OUT / "codebook.yml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Freeze the split exactly as `scruple load` would.
    from scruple.hashing import corpus_hash

    ids = [row["respondent_id"] for row in rows]
    hashes = [item_hash(row["response"]) for row in rows]
    splits = assign_splits(ids, corpus_hash=corpus_hash(hashes), seed=42)
    splits.save(OUT / ".scruple" / "splits.json")

    # A pre-coded gold sample: what a researcher would have after `scruple gold`.
    truth_by_id = {row["respondent_id"]: applied for row, applied in zip(rows, truth, strict=True)}
    gold_rng = random.Random(SEED + 1)
    gold_lines: list[str] = []
    sampled: list[tuple[str, str, float]] = []
    for split_name in ("calibration", "test"):
        pool = [i for i in ids if splits.assignment[i].value == split_name]
        take = gold_rng.sample(pool, GOLD_N // 2)
        pi = (GOLD_N // 2) / len(pool)
        sampled += [(i, split_name, pi) for i in take]

    overlap = set(gold_rng.sample([i for i, _, _ in sampled], OVERLAP_N))
    for item_id, split_name, pi in sampled:
        for code, spec in CODES.items():
            is_on = truth_by_id[item_id][code]
            for coder in ("coder_1", "coder_2"):
                if coder == "coder_2" and item_id not in overlap:
                    continue
                gold_lines.append(
                    json.dumps(
                        {
                            "code_id": code,
                            "coder": coder,
                            "inclusion_probability": round(pi, 6),
                            "item_id": item_id,
                            "label": (
                                int(is_on)
                                if coder == "coder_1"
                                else _noisy(gold_rng, is_on, spec.human_agreement)
                            ),
                            "model_visible": False,
                            "sample_seed": 42,
                            "split": split_name,
                            "stratum": "uniform",
                            "timestamp": "2026-09-20T10:00:00+00:00",
                        },
                        sort_keys=True,
                    )
                )
    (OUT / ".scruple" / "gold.jsonl").write_text("\n".join(gold_lines) + "\n", encoding="utf-8")

    # Project config, pointing at the recorded probabilities.
    (OUT / "scruple.yml").write_text(
        "# Configured for the offline example: the `recorded` backend replays\n"
        "# probabilities from a file, so this runs with no API key and nothing\n"
        "# leaves your machine.\n"
        "version: 1\n"
        "corpus:\n"
        "  path: data/corpus.csv\n"
        "  text_column: response\n"
        "  id_column: respondent_id\n"
        "splits:\n"
        "  seed: 42\n"
        "  dev: 0.10\n"
        "  calibration: 0.45\n"
        "  test: 0.45\n"
        "backend:\n"
        "  name: recorded\n"
        "  model: example-recorded-1\n"
        "  endpoint: probabilities.json\n"
        "engine:\n"
        "  concurrency: 8\n"
        "  budget_usd: 5.0\n"
        "gold:\n"
        "  n: 600\n"
        "  enrich_rare_codes: true\n"
        "  double_coded_overlap: 150\n"
        "thresholds:\n"
        "  alpha: 0.10\n"
        "  delta: 0.05\n"
        "chunking:\n"
        "  max_tokens: 28000\n"
        "  overlap_tokens: 200\n"
        "  aggregation: max\n",
        encoding="utf-8",
    )

    print(f"corpus: {len(rows)} rows, {len(table)} distinct texts")
    print(f"gold:   {len(gold_lines)} judgements, {OVERLAP_N} items double-coded")
    for code, spec in CODES.items():
        positives = sum(1 for t in truth if t[code])
        print(
            f"  {code:<22} prevalence {positives / len(rows):.3f}  "
            f"leak {spec.leak:.2f}  fpr {spec.false_positive_rate:.2f}"
        )


if __name__ == "__main__":
    main()
