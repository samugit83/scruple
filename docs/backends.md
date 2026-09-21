# Backends

A backend turns (text, codes) into one probability per code. Nothing above that
layer knows which model produced a number, and nothing below it makes a
decision — all thresholding happens in `stats/`, so changing your mind about
thresholds never means re-running the model.

| backend | what it is | probability from |
|---|---|---|
| `jev` | TypeSafe's System One, trained with RLCD for calibrated decisions | The model directly |
| `local` | Any OpenAI-compatible endpoint you run | Token logprobs, else k-sample voting |
| `openai` | OpenAI chat completions | Token logprobs, else k-sample voting |
| `anthropic` | Claude, via the Messages API | k-sample voting |
| `recorded` | Replays probabilities from a file | A previous run |

---

## `jev` — the default

The project's thesis rests on calibration: without probabilities that mean what
they say, abstention carries no information and the threshold procedure has
nothing to work with. Jev is trained specifically to have that property.

```yaml
backend:
  name: jev
  model: typesafe-ai/jev
  endpoint: null      # or a gateway route
```

Credentials come from `JEV_API_KEY`, with `JEV_BASE_URL` and `JEV_MODEL`
overriding the endpoint and model. All three are read from the environment or
from a `.env` beside `scruple.yml`.

**Wire format.** Established against the live endpoint, not inferred:

```jsonc
// request
{ "model": "typesafe-ai/jev",
  "state": "<the item text>",
  "questions": {                          // a record keyed by code id, not a list
    "access_barrier": { "type": "noul",   // lowercase
                        "instructions": "<your code definition>" } } }

// 200
{ "model": "typesafe-ai/jev",
  "answers": { "access_barrier": { "type": "noul", "noul": 0.98 } },
  "usage": { "input_tokens": 323, "output_tokens": 43 } }
```

Your definition goes into `instructions` verbatim — scruple never paraphrases
it, because the definition is the intellectual contribution of your study. Any
`examples_yes` / `examples_no` are appended to the same field: the API ignores
keys it does not recognise, so sending them under an `examples` key would
silently discard them.

All codes go in a single call. Questions are evaluated in parallel and in
isolation, so asking about twelve codes costs barely more than asking about
one. The engine splits into several calls only when the token budget requires
it, and every code appears in exactly one call.

The gateway reports the real cost of each call, which the engine accumulates —
so the run manifest records what you actually spent, not only what was
projected.

To re-verify the format after an API change:

```bash
uv run python bench/probe_jev.py --explore
```

That records fresh fixtures into `tests/contract/fixtures/jev/` and prints what
the API says about variant request shapes. It is run deliberately and never
from the test suite: no test in any tier is allowed to touch the network.

---

## `local` — for personal data

Not an afterthought. Interview transcripts and survey free text are personal
data, often special-category, and sending them to a single-region hosted API
will not clear many European university ethics committees. A tool those
researchers cannot use does not help them.

```yaml
backend:
  name: local
  model: your-model-name
  endpoint: http://localhost:8000/v1/chat/completions
```

Anything speaking the OpenAI chat-completions schema works: vLLM, llama.cpp's
server, Ollama, LM Studio, or a gateway in front of your institution's own
deployment. No API key is required; set `SCRUPLE_LOCAL_API_KEY` if your server
wants one.

**The honest trade-off.** A local open-weight model is slower and will usually
be less accurate than the hosted calibrated one, which means lower coverage:
more items routed to you, and more of your hours. The §8.8 chart in your own
validation report shows exactly how much — run `scruple compare` against both
and read the gap. Nobody has to take a claim about this on trust.

---

## `openai` and `anthropic` — for comparison

These exist so the benchmark can compare against the alternative honestly. They
are implemented fairly: structured output, a neutral prompt, deterministic
temperature, and logprobs where the endpoint exposes them. Sandbagging them
would destroy the only asset this project has.

Two ways to get a probability from a chat model:

1. **Token logprobs.** One call, and the number at least derives from the
   model's own distribution. Used whenever the endpoint provides them.
2. **k-sample voting** (k = 5, temperature 1). Five calls per item for a
   five-point probability scale, and poorly calibrated — the literature's
   κ ≈ 0.57 figure is measured on coders like this. Used only as a fallback,
   and the backend stops paying for logprobs once an endpoint declines them.

A real limitation, stated rather than hidden: with one response covering
several codes, the logprob distribution is shared across them, so the
per-code confidence is coarser than it looks. This is the best an unmodified
chat endpoint offers, and it is precisely why a model trained for calibrated
decisions is the default.

Anthropic's Messages API does not expose token logprobs, so that backend always
uses voting.

---

## `recorded` — offline

Replays probabilities from a JSON file keyed by item hash.

```yaml
backend:
  name: recorded
  endpoint: probabilities.json
```

Two uses: the worked example ships one so the whole pipeline runs with no API
key, and it makes a run exactly reproducible for a reviewer who has your
probabilities but not your credentials.

Keying on the item hash rather than the id means a recording survives ids being
regenerated, and identical texts share an entry. In strict mode — the default —
an item or code the recording does not cover raises, rather than silently
returning `None` and pushing everything into the abstention band.

---

## Writing your own

The contract is small:

```python
class Backend(Protocol):
    name: str
    def model_version(self) -> str: ...
    def score(self, state: str, codes: Sequence[Code]) -> dict[str, float | None]: ...
```

Subclass `BaseBackend` and implement `model_version` and `_score_batch`; you
get token-budget splitting, usage accounting and response validation for free.

Three invariants are enforced for you rather than trusted of you:

1. The returned mapping contains **every** requested code id.
2. Values are floats in `[0, 1]`, or `None`.
3. A failure is `None`, **never** 0.0 — a missing probability is not a negative
   judgement.

An out-of-range value raises rather than being clamped. Clamping would hide a
real bug behind a plausible number, and every guarantee downstream assumes the
probability means something.

Register it in `scruple/backends/registry.py`, add a line to the privacy table
there saying what leaves the machine, and write contract tests against recorded
fixtures — never a live call.
