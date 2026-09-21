# Phase 0 datasets

Requirements, in order of importance:

1. **Multiple human coders per item**, so human–human agreement gives a ceiling.
2. **Interpretive codes**, not just surface-level categorisation.
3. A **published codebook**, so the definitions are not ours to tune.
4. A licence permitting research use, and permitting us to publish derived
   results.

**No dataset is committed to this repository.** §14.11 forbids it. Each dataset
gets a fetch script under `bench/fetch/` that downloads and prepares it into
`bench/data/`, which is gitignored.

---

## Primary: real qualitative-research corpora

These are the ones the result rests on. Directly comparable to numbers already
in the literature, which is the point.

| dataset | why | what to confirm |
|---|---|---|
| Corpora from the QualiGPT paper | Published LLM-qualitative-coding comparisons; results are directly comparable | Licence for derived results; whether coder-level labels are available or only adjudicated ones |
| `uhh-lt/llm4qda` benchmark | Purpose-built for this task | Licence; codebook provenance; per-coder labels |

**Open question for whoever runs this:** several qualitative datasets publish
only the *adjudicated* code, not the individual coders' judgements. Without
per-coder labels there is no human–human ceiling, and rule 2 in the benchmark
README cannot be honoured. Confirm this before spending any money on scoring.

## Secondary: a sanity check only

| dataset | why | the caveat |
|---|---|---|
| GoEmotions (~58k Reddit comments, 27 categories, multiple raters) | Multi-rater structure and published agreement figures; easy to obtain | **Surface-level, concrete categorisation.** The literature reports LLMs doing well here and poorly on interpretive coding, so a good result here says little |

If the write-up leads with GoEmotions it will overstate the finding, and a
reviewer will say so. Report it as a sanity check on the machinery, explicitly
labelled as such.

---

## Preparing a dataset

A prepared dataset is a directory containing:

```
bench/data/<name>/
  corpus.csv        item_id, text
  codebook.yml      codes, from the published definitions only
  gold.jsonl        one line per (item, code, coder) judgement
  SOURCE.md         provenance, licence, how it was fetched, what was changed
```

`SOURCE.md` is not optional. A benchmark whose inputs cannot be traced is a
benchmark nobody can check, and this project's only asset is being checkable.

The `gold.jsonl` format matches what `scruple gold` writes, so a prepared
dataset drops straight into a scruple project:

```json
{"item_id": "…", "code_id": "…", "label": 0, "coder": "rater_3",
 "split": "calibration", "stratum": "uniform", "inclusion_probability": 1.0,
 "model_visible": false, "sample_seed": 42, "timestamp": "…"}
```

Set `inclusion_probability` to 1.0 when every item was coded, which is the
usual case for a published dataset — the whole corpus is the sample.
