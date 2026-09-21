# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The CLI's `--json` output is a public, versioned interface (plan §13.1).
Breaking it is a breaking change.

## [Unreleased]

### Added

- **Statistics layer** (`scruple.stats`): Cohen's κ, Krippendorff's α,
  calibration diagnostics, stratified bootstrap intervals, selective and labour
  curves, inverse-probability weighting, split sealing, and class-conditional
  (Mondrian) threshold selection. 100% line and branch coverage, mypy strict.
- **Project layer**: `codebook.yml` and `scruple.yml` parsing, CSV / JSONL /
  text-folder corpora, and a frozen three-way dev/calibration/test split.
- **Backends**: `jev` (default), `openai`, `anthropic`, an OpenAI-compatible
  `local` backend, and `recorded` for offline replay.
- **Engine**: per (item × code) SQLite cache storing hashes only, chunking with
  frozen aggregation, a cost gate, run manifests and drift detection.
- **Gold collection**: uniform and enriched sampling with known inclusion
  probabilities, an append-only store, and blind terminal coding.
- **CLI**: `init`, `load`, `splits`, `codebook check`, `try`, `gold`, `check`,
  `run`, `review`, `export`, `compare` and `purge`, each with `--json`.
- **Validation report**, the §8.8 reliability-against-labour chart, and a
  methods paragraph.
- **Worked example** (`examples/vaccine_survey`): a synthetic corpus scored by
  the real Jev endpoint, running the full pipeline offline with no API key.

### Changed from the design document

Five changes were made during implementation because measurement disagreed with
the plan. Each is recorded in `scruple_integration_plan.md` §19 with its
reasoning, and in the code at the point it matters.

- **Threshold selection scans the grid rather than walking it.** §8.6 specified
  a fixed-sequence walk from the most conservative band, stopping at the first
  failure. That band is the hardest to certify, not the easiest — it accepts too
  few gold positives for the test to have power, and narrowing keeps
  confidently-wrong positives while dropping moderately-confident correct ones.
  On a simulated coder the walk certified 0% of trials where a corrected full
  scan certified ~47% at full coverage.
- **The grid correction is weighted, not flat.** A flat `δ/|Λ|` nearly doubles
  the gold sample required, for grid resolution nobody asked for. Shares now
  fall geometrically from the highest-coverage band.
- **`alpha` defaults to 0.10 and `gold.n` to 600.** The plan's `alpha: 0.05`
  with 300 gold items cannot certify anything: a backend with 2% per-class error
  needs roughly 1,400 gold items at α = 0.05 against about 360 at α = 0.10.
  `docs/methodology.md` §4 carries the arithmetic.
- **The Jev wire format was established against the live endpoint.** `questions`
  is a record keyed by code id rather than an array, the type is lowercase
  `noul`, and a question needs `instructions` rather than a bare `question`
  string.
- **The Python floor is 3.11, not 3.10.** Current numpy and scipy both require
  ≥3.12, so a 3.10 floor would force pinning to old numpy.

### Known gaps

- The Phase 0 benchmark has not been run against published
  qualitative-coding corpora. The README's chart demonstrates the machinery on a
  synthetic corpus with a real model; it says nothing yet about whether the
  thesis holds on interpretive coding.
- `score` and `choice` code types are v1.5 and are refused with a clear error.
- The R package of Phase 2 is not built.
