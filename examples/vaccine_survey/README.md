# Worked example: vaccine survey

A complete scruple project that runs offline, with no API key.

```bash
cd examples/vaccine_survey
uv run scruple codebook check
uv run scruple check              # fit thresholds, report on held-out data
uv run scruple run --yes          # code the corpus
uv run scruple export             # write the three output files
```

Then read `out/validation_report.md`.

## What is in here

| file | what it is |
|---|---|
| `data/corpus.csv` | 1,200 synthetic survey responses |
| `codebook.yml` | Six binary codes |
| `probabilities.json` | Probabilities from the **real** Jev endpoint, recorded |
| `.scruple/gold.jsonl` | A pre-coded gold sample of 600 items, 150 double-coded |
| `.scruple/splits.json` | The frozen dev / calibration / test partition |

**The corpus is synthetic and the ground truth is generated.** §14.11 of the
design forbids committing real personal data or a real corpus, and an example
shipping real survey answers would be exactly the thing this tool exists to
help people handle carefully. The *probabilities*, though, are real: the corpus
was scored by the live model, so what you see is how it actually behaves rather
than how a simulation of it behaves.

The gold sample stands in for work a researcher would have done by hand. It
takes coder 1's judgements as the reference and gives coder 2 a noisier
overlapping subset, which is what produces the human–human ceiling.

## What it shows

Three of the six codes certify; three do not, in three different ways. That is
the point of the example — a tool that only ever shows success is not
demonstrating the thing that makes it worth using.

| code | outcome | why |
|---|---|---|
| `access_barrier` | ok, κ ≈ 0.98 | Concrete and well-defined |
| `no_recommendation` | ok, κ ≈ 0.99 | Concrete, some genuine ambiguity |
| `procrastination` | ok, κ ≈ 0.93 | Concrete |
| `distrust_pharma` | `NOT_AUTOMATABLE` | Real errors, above the target rate |
| `religious_objection` | `INSUFFICIENT_EVIDENCE` | Too rare — 7 positives in the gold sample |
| `dignity_violation` | `NOT_AUTOMATABLE` | Interpretive; the two human coders agree at only κ ≈ 0.28 |

`dignity_violation` is the instructive one. `check` reports the human–human
agreement alongside the failure, because a code two trained people cannot agree
on is a **codebook** problem rather than a model problem — and being told that
is worth more than a number.

## Regenerating it

```bash
uv run python bench/make_example.py           # simulated probabilities
uv run python bench/make_example.py --live    # real Jev, about $0.03
```

`--live` needs `JEV_API_KEY` in the environment or in a `.env` at the
repository root.
