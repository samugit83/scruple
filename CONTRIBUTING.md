# Contributing

## Setup

```bash
uv sync --all-extras
uv run pre-commit install    # mirrors the lint, format and typecheck CI jobs
```

## Running the tests

The suite is layered by how fast it is, because a suite that is slow to run is
a suite people skip.

```bash
# The development loop: ~700 tests, a few seconds.
uv run pytest tests/unit tests/property tests/contract -q

# Everything except the simulations and the performance guards: ~4 minutes.
uv run pytest -m "not statistical and not slow" -q

# The guarantee itself, by simulation. Slow, and the most important tests here.
uv run pytest tests/statistical -q

# Performance guards.
uv run pytest -m slow -q
```

| tier | what it checks |
|---|---|
| `unit` | Pure functions against hand-computed expectations |
| `property` | Invariants that must hold for all inputs (hypothesis) |
| `statistical` | The §8.6 guarantee, by simulation, with negative controls |
| `contract` | Backends against recorded HTTP fixtures |
| `integration` | Engine, cache and split integrity together |
| `e2e` | The worked example, from `check` to `export` |

**No test in any tier may touch the network.** CI runs with proxies pointed at
a black hole, so an accidental live call fails loudly rather than passing on
one machine and failing on another. To re-record a backend fixture against the
real API, run `bench/probe_jev.py` deliberately.

## The rules that are not negotiable

These come from the design, and CI blocks a merge on the first three.

1. **`src/scruple/stats/` stays at 100% line and branch coverage.** No
   `# pragma: no cover`. The statistics are the product, so the test suite is
   the evidence for the claim rather than overhead.
2. **`stats/` imports nothing else from the package.** It depends on numpy and
   scipy and nothing more, so it is testable in isolation against synthetic
   data with known properties. `tests/unit/test_architecture.py` enforces this.
3. **The negative control must never certify an all-negative coder.** If
   `tests/statistical/test_guarantee.py` ever passes one, the guarantee is
   broken and nothing else in the repository matters. Stop and fix that first.
4. **Never fit and report on the same data.** There are assertions for this
   (`tests/integration/test_split_integrity.py`); do not remove them.
5. **When the design and reality disagree, reality wins**, and you record what
   you measured in the same commit. The changelog carries several such
   corrections: specified procedures that did not survive being run.

## Writing tests

Golden snapshots live in `tests/e2e/snapshots/`. After an intended change:

```bash
uv run pytest tests/e2e/test_golden.py --snapshot-update
```

The diff must appear in your commit. That is the point of them.

`tests/fixtures/` generates synthetic gold data with specified prevalence,
per-class error rates and second-coder agreement, so a test can assert against
known truth. Use it rather than inventing another generator — the last three
drifted apart from each other.

**Never commit a real corpus, a real gold sample, or an API key.** Test data is
synthetic; `.env` is gitignored, and a key in git history is a live key forever.

## Commits

Small, reviewable, and the message explains *why* rather than what. If you
changed behaviour because a measurement said so, put the measurement in the
message — the next person needs to know it was measured rather than assumed.
