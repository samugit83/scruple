# Phase 0 — the benchmark

**This is a benchmark, not a prototype. It exists to try to falsify the
thesis.**

The thesis, stated so it can fail:

> A calibrated decision model's stated confidence separates items it codes
> reliably from items it does not, on real qualitative-research data, well
> enough that a researcher reaches publication-grade reliability while
> hand-coding only a small minority of the corpus.

The test is the chart: **final reliability of the whole corpus** against
**human labour required**, with a random-selection baseline.

- **Thesis holds** if calibrated selection reaches κ ≥ 0.70 over the full
  corpus at substantially less labour than random selection. The gap between
  the curves is the product.
- **Thesis fails** if the curves sit close together. Confidence carries no
  usable signal, and the product collapses.

If they converge: **stop building.** Publish the negative result — it is
genuinely useful to the field and costs nothing extra. Do not rationalise it,
and do not tune until it looks good.

## Status

**Not yet run against the literature's corpora.** What exists here is the
harness and a worked example; the datasets in `datasets.md` still need
licensing confirmed and download scripts written. Until that is done, the
headline chart in the README is demonstrated on a synthetic corpus with a real
model — which shows the machinery works, and shows nothing about whether the
thesis holds on interpretive coding.

That distinction matters enough to keep saying out loud.

## What is here

| file | what it does |
|---|---|
| `make_example.py` | Generates the synthetic worked example. `--live` scores it with the real Jev endpoint. |
| `probe_jev.py` | Records live API responses as contract-test fixtures. Run deliberately; never from the test suite. |
| `datasets.md` | The corpora Phase 0 needs, and what has to be confirmed about each. |
| `run_benchmark.py` | Scores a prepared corpus with several backends and writes the comparison. |

## Running it

```bash
# 1. Prepare a dataset (see datasets.md). Data is never committed.
uv run python bench/fetch/<dataset>.py --out bench/data/<dataset>

# 2. Score it with each backend and write the curves.
uv run python bench/run_benchmark.py \
    --dataset bench/data/<dataset> \
    --backends jev,openai,local \
    --out bench/results/<dataset>

# 3. Read bench/results/<dataset>/report.md and the chart beside it.
```

## Rules

These are not style preferences. They are what keeps the result worth having.

1. **Write the codebooks from each dataset's published category definitions
   only.** Tuning a definition against the labels is the benchmark equivalent
   of fitting on the test set.
2. **Report the human–human ceiling.** Every dataset here has multiple coders
   per item for exactly this reason. No automated coder should be expected to
   beat the agreement two trained people reach on the same code, and a result
   that appears to has probably measured something else.
3. **Implement the comparison backends fairly.** Do not sandbag them. A rigged
   comparison destroys the project's credibility, which is the only asset it
   has.
4. **Report what the chart shows, including if it shows nothing.**
5. **Interpretive codes are the real test.** The literature reports LLMs doing
   well on concrete, surface-level categorisation and poorly on interpretive
   coding. Benchmarking only on the easy kind would overstate the result, and a
   reviewer would rightly say it is not qualitative research data.

Budget to reach the go/no-go decision: roughly $50 and two weeks.
