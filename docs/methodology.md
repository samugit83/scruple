# Methodology

Written for a methodologist reading sceptically. It states what scruple
computes, what it assumes, and where it stops — including the places where the
honest answer is "not enough evidence".

If you only read one section, read [What the guarantee does not cover](#what-the-guarantee-does-not-cover).

---

## 1. The problem, stated precisely

You have a corpus of texts and a codebook of binary categories you wrote. You
want each text coded against each category, and you want to be able to defend
the result. The conventional defence is a second human coder and an agreement
statistic: two people code the same items independently, and Cohen's κ says how
much they agreed beyond chance. Journals broadly expect κ ≥ 0.70.

An automated coder can stand in for neither coder, because its overall
agreement with a human sits well below that bar. The published figure for
general-purpose LLMs on interpretive coding is around κ = 0.57.

scruple does not try to raise that number. It asks a different question:

> Can we identify, **in advance**, the subset of items the machine codes
> reliably — and hand the rest to a person?

If yes, the delivered dataset is a mixture: machine-coded where the machine was
demonstrably reliable, human-coded everywhere else. Its reliability is measured
as a whole, against held-out human coding, and it can clear the bar even though
the machine alone could not.

Everything below is the machinery for deciding "in advance" honestly.

---

## 2. The three-way split

The corpus is partitioned once, deterministically, at load time, and the
partition is written to `.scruple/splits.json` and never changed:

| split | share | what it is for |
|---|---|---|
| **dev** | 10% | The only split `scruple try` may draw from. You iterate on code definitions here while looking at model output. That is fitting, and it is quarantined. |
| **calibration** | 45% | Gold items drawn from here fit the decision thresholds. |
| **test** | 45% | Gold items drawn from here produce every number reported as a result. |

Assignment is by keyed hash: each item's position comes from
`sha256(seed, item_id)`, the corpus is ordered by that key, and the ordering is
sliced. This makes the partition reproducible across machines without sharing
an RNG implementation, and independent of the order rows happen to appear in
the file. Slice sizes are exact rather than approximate — 10/45/45 of 1,000
items is exactly 100/450/450.

Because sizes are exact, they depend on the corpus size, so adding rows moves a
boundary. `splits.json` therefore records the corpus hash and scruple refuses a
corpus it was not built from, rather than silently re-partitioning: items
moving between calibration and test would invalidate everything already
computed.

The fitting path receives the test split inside a wrapper that raises on every
read. Unsealing requires naming a purpose from a two-item allowlist that
excludes fitting. This is enforced in code rather than in prose because the
mistake is a one-character slip and the consequence is a number that looks
publishable and is not.

---

## 3. What is fitted, and how

For each code independently, on the **calibration split only**:

### 3.1 The decision rule

A threshold pair `(t_lo, t_hi)` defines an abstention band:

```
p ≥ t_hi   → decide 1
p ≤ t_lo   → decide 0
otherwise  → abstain; a human codes this item
```

### 3.2 The risk being controlled

The guarantee is **class-conditional**, computed separately within each true
class:

```
P(decision wrong | accepted, gold = positive) ≤ α
P(decision wrong | accepted, gold = negative) ≤ α
```

each holding with confidence 1 − δ.

This is not a stylistic preference. A *marginal* error guarantee is degenerate
on rare codes: at 3% prevalence, a coder that answers "no" to everything has a
3% marginal error rate at 100% coverage, passes any α = 5% test, and has κ = 0.
Splitting the risk by true class removes the degeneracy, because that coder is
wrong on 100% of the accepted positive class. A regression test asserts it is
rejected at every prevalence from 0.50 down to 0.02; if that test ever passes
such a coder, nothing else in the project matters.

### 3.3 The test

For each candidate band and each class separately, with `n` accepted items of
that class and `E` errors among them, the p-value for `H₀: risk > α` is the
binomial tail `P(Binom(n, α) ≤ E)`. The band is valid for that class when the
p-value clears the candidate's confidence level.

A class with **no** accepted items does not pass. Zero evidence is not evidence
of safety, and accepting nothing from the positive class would otherwise be a
free pass — a second route to the same degeneracy §3.2 closes.

### 3.4 Multiplicity, and why the grid is scanned rather than walked

Every candidate in the grid is tested, and the grid is corrected with a
**weighted Bonferroni** whose shares fall geometrically from the
highest-coverage band: ½, ¼, ⅛, … The band that the selection rule prefers is
therefore tested at δ/2.

Two decisions are embedded here, and both were made against measurement rather
than from first principles.

**Why not fixed-sequence.** The natural procedure — order the grid from most
conservative, walk it, stop at the first failure — is sound only if the most
conservative band is the easiest to certify. It is the hardest, for two reasons
that both bite hardest on rare codes:

- *No power.* A high `t_hi` accepts only a handful of gold positives, and a
  handful of clean items is not evidence of controlled risk. At α = δ = 0.05,
  zero errors out of six gives p = 0.74.
- *Adverse selection.* Narrowing the band drops moderately-confident **correct**
  decisions while keeping confidently-wrong positives, because a low score is an
  accepted "no". The error *share* of the positive class can therefore rise as
  coverage falls.

On a simulated coder that leaks 1–2% of its positives into the
confident-negative cluster, the walk certified **0%** of trials where a
corrected full scan certified about **47%** at full coverage. The two classes
also move in opposite directions as the band changes, so no ordering of the
grid is monotone for both at once and no fixed-sequence walk in either
direction is sound.

**Why weighted rather than flat.** A flat `δ/|Λ|` nearly doubles the evidence
required — 112 accepted items per class against 59 uncorrected, at α = δ = 0.05
— in exchange for grid resolution nobody asked for. Bonferroni permits any
pre-specified split of δ, and the ordering that assigns the shares here is band
width, a property of the grid rather than of the data.

The cost, stated plainly: bands far from full coverage receive a very small
share and are effectively untestable at realistic gold sizes. A code that can
only be certified inside a narrow accept region will report `NOT_AUTOMATABLE`.
That is a code where the backend is weak and coverage would be low anyway.

### 3.5 Across codes, no correction

Each code is a separate claim, reported separately, exactly as researchers
already report per-code κ. A family-wise correction over 60 codes would reduce
δ to 0.0008 and reject virtually everything against a realistic calibration
split, while adding no claim anyone makes. The report states that the
guarantees are per code and marginal over items.

---

## 4. How much gold coding you actually need

This is the question that decides whether the tool is usable on your study, and
it has an arithmetic answer.

Certification is limited by the **smaller class of each code**, not by the
sample as a whole. With zero observed errors, rejecting `risk > α` at level δ′
requires

```
n ≥ ln(δ′) / ln(1 − α)
```

accepted items in that class. With observed errors it requires more. Measured
requirements, at δ′ = δ/2 = 0.025:

| true per-class error | α = 0.05 | α = 0.10 | α = 0.15 |
|---|---|---|---|
| 0% | 72 | 36 | 23 |
| 1% | 72 | 36 | 23 |
| 2% | 142 | 36 | 23 |
| 3% | 364 | 54 | 23 |
| 5% | never | 114 | 35 |

Read the "never" literally: α = 0.05 cannot certify a coder whose true
per-class error is 5%, however much gold you collect, because the null being
tested is that the risk exceeds α.

**What this means in practice.** At 20% prevalence, a backend with 2% per-class
error needs roughly **1,400** gold items at α = 0.05, against about **360** at
α = 0.10. The defaults are `alpha: 0.10` and `gold.n: 600`, which work
together. Setting α = 0.05 is supported and is the right choice if your sample
can carry it — `scruple check` reports the shortfall in items when a code fails
for want of evidence rather than accuracy, because "collect 30 more positives"
is actionable and "NOT_AUTOMATABLE" is not.

The positive-class bound is, in effect, a **recall** requirement: at most α of
the true positives the model accepts may be called "no". True positives the
backend scored near zero are accepted at every band and are errors at every
band, so a backend that confidently misses more than α of the positives cannot
be certified at any threshold — correctly. Expect this to be the commonest
reason a real code comes back `NOT_AUTOMATABLE`.

---

## 5. Sampling, and the weights

The guarantee rests on the gold sample being a random sample of the corpus with
known inclusion probabilities. `scruple gold` samples with a recorded seed and
refuses a hand-picked sample, because a hand-picked one breaks the assumption
silently.

Two strata:

1. **Uniform.** A uniform random draw from calibration and from test. Inclusion
   probability is `n/N` exactly. The guarantee rests on this, and the binomial
   test is exact.
2. **Enriched** (optional, per code). 300 uniform items at 3% prevalence yields
   nine positives, on which no statistic means anything — and most research
   codes are rare, so this is the normal case rather than an edge case. The
   enriched stratum draws extra items from a pool of likely positives, ranked by
   model probability. The pool is defined by **rank** rather than by an absolute
   probability cut, so its size is known and the inclusion probability can be
   computed rather than estimated.

Every estimator accepts weights and uses inverse-probability weighting over the
combined sample.

The enriched supplement is a two-phase design: it is drawn from the pool items
the uniform phase did not already take, so its inclusion probability is
conditional on the realised first phase,

```
π_i = π_uniform + (1 − π_uniform) · n_supplement / |pool not already sampled|
```

That is exact given phase one, but it makes the binomial test of §3.3
approximate rather than exact, because the weighted error count is integerised
against Kish's effective sample size `(Σw)² / Σw²` rather than a raw count. The
report states which case applies; look for "the risk test is **exact**" or
"**approximate**". With a uniform-only sample it is always exact.

**Hard gate.** `check` refuses to certify any code with fewer than **15
positive instances** in the relevant split, reporting `INSUFFICIENT_EVIDENCE`
with the count. Reporting a confident-looking κ built on nine items is the most
likely way this tool could produce a false result.

---

## 6. What is reported, and why it is not selective κ

### 6.1 Selective κ is a diagnostic, not the headline

κ can **fall** as coverage drops, even when the coder is behaving perfectly.
Confident subsets skew toward one class, which raises chance agreement and
compresses κ. Worked example, both tables real:

```
Confident subset (500 items, 5 positive):
  a=4  b=1  c=2  d=493     accuracy 99.4%    κ = 0.724

Full coverage (1000 items, 200 positive):
  a=170 b=30 c=40 d=760    accuracy 93.0%    κ = 0.785
```

99.4% accuracy scores *worse* than 93%. Reporting selective κ against coverage
as the headline would make a working model look broken. scruple computes it,
reports it as a diagnostic with prevalence beside it, and a unit test asserts
that this inversion still occurs so a future refactor cannot quietly "fix" it.

### 6.2 The headline: reliability of the delivered dataset

Computed on the **test split**, simulating the real workflow:

- items inside the abstention band take the **human** label, because a person
  coded them;
- items outside it take the **model's** decision;
- κ is then computed over **every** test item.

That is the number describing the object you are actually delivered. It is
reported with a 95% bootstrap percentile interval, stratified by the gold label
— stratified because on a rare code an unstratified resample routinely draws
zero positives, which makes κ undefined and the interval meaningless.

Krippendorff's α is reported alongside κ. The two are implemented independently
from their own definitions rather than one derived from the other, so their
agreement is a genuine cross-check. They are not algebraically identical: α is
Scott's π with a small-sample correction, using pooled marginals where κ uses
each coder's own. Where they diverge by more than 0.10 the report says so,
because a gap that size usually means skewed prevalence.

### 6.3 The hero chart

Reliability of the finished dataset plotted against the share of the corpus a
person must code, with a random-selection baseline — the naive strategy of
coding everything by machine and hand-checking a random share. The vertical gap
between the two curves is what calibrated abstention buys, expressed in hours
at a fixed quality bar.

---

## 7. What the guarantee does not cover

Read this before quoting any number from the report.

- **It covers only the model-decided subset.** Items routed to a human are not
  covered by it, and do not need to be. The report gives the per-source
  breakdown so you can see how much of the dataset each claim applies to.
- **It is per code, and marginal over items.** It is not a simultaneous
  statement across codes, and not a statement about any individual item.
- **It assumes the gold sample is random**, with known inclusion probabilities.
  A hand-picked sample breaks it, and nothing in the output would look wrong.
- **It assumes the corpus is exchangeable** with what the thresholds were fitted
  on. Coding a 2019 corpus with thresholds fitted on 2026 data is outside it.
- **It says nothing about codes that were not certified.** Those are yours to
  code by hand, end to end.
- **Calibration diagnostics are diagnostics.** Brier and ECE describe the
  probabilities; they are not the guarantee, and a well-calibrated backend with
  poor discrimination still fails certification.
- **The automated coder is a coder, not ground truth.** Report it as a coder.

One reassurance, since it is easy to assume the opposite: the guarantee does
**not** depend on the backend being well calibrated. The procedure tests
observed error counts, not stated probabilities, so a monotone-overconfident
backend preserves item ranking and risk control survives. Calibration is what
makes a *useful* band reachable — it is what buys coverage, not what buys
validity.

---

## 8. Reliability of a mixed-provenance dataset

The delivered dataset has cells from three sources: model-decided (`auto`),
human-decided (`human`, from the gold sample or from review), and undecided
(`abstain_unreviewed`), plus cells on codes that were never certified
(`not_automatable`).

The report characterises it as follows, and you should quote it the same way:

- κ over the **test split as a whole**, under the simulation of §6.2. This is
  the number that describes the delivered object.
- The per-source breakdown — count and share of cells from each source —
  accompanies it, so a reader can see the mixture.
- The class-conditional guarantee applies to the `auto` subset **only**, and the
  report says so explicitly rather than letting a reader assume otherwise.

There is no single number that summarises "the dataset plus the guarantee". The
honest presentation is κ for the object, the guarantee for the machine-decided
part, and the provenance table connecting them.

---

## 9. Verifying these claims

The statistics layer depends only on numpy and scipy and imports nothing else
from the package, so it is testable in isolation. Its test suite is the evidence
for the claims above, not an afterthought:

- **100% line and branch coverage** on `src/scruple/stats/`, enforced in CI.
- κ and α checked against hand-computed confusion tables, including the §6.1
  paradox pair and every degenerate case.
- The guarantee itself validated by simulation: thresholds fitted on one
  simulated calibration sample, the realised class-conditional error measured on
  a **fresh** sample, repeated 1,000 times per prevalence, asserted to hold in at
  least 1 − δ of trials at prevalence 0.50, 0.20, 0.05 and 0.02.
- Negative controls that must never certify: an all-negative coder, an
  all-positive coder, a coder whose probabilities are independent of the truth,
  and a coder leaking more than α of its positives.

```
uv run pytest tests/statistical -q     # the guarantee, by simulation
uv run pytest -q                       # everything
```
