"""Project configuration: `scruple.yml` (§7.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from .errors import ValidationError

DEFAULT_BACKEND = "jev"
DEFAULT_MODEL = "typesafe-ai/jev"
"""§17.4 is a values decision, resolved to the hosted calibrated backend: it is
the model the thesis rests on and it gives the best first run. `scruple init`
prints the privacy trade-off, and `docs/privacy.md` states exactly what leaves
the machine, because for personal data the local backend is the right answer."""


class _Strict(BaseModel):
    """Reject unknown keys: a typo in scruple.yml must not be silently ignored."""

    model_config = ConfigDict(extra="forbid")


class CorpusConfig(_Strict):
    path: str = "data/corpus.csv"
    text_column: str = "text"
    id_column: str | None = None


class SplitsConfig(_Strict):
    seed: int = 42
    dev: float = Field(default=0.10, ge=0.0, le=1.0)
    calibration: float = Field(default=0.45, ge=0.0, le=1.0)
    test: float = Field(default=0.45, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _sums_to_one(self) -> SplitsConfig:
        total = self.dev + self.calibration + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"dev + calibration + test must be 1.0, got {total:g}")
        return self


class BackendConfig(_Strict):
    name: str = DEFAULT_BACKEND

    model: str = DEFAULT_MODEL
    """Model alias. The run manifest records whichever version the API actually
    returned, not this, because the report must say what ran (§7.3)."""

    endpoint: str | None = None
    """Override to route through a gateway (§9.1). Left unset, the backend uses
    its own default, or JEV_BASE_URL from the environment or `.env`."""


class EngineConfig(_Strict):
    concurrency: int = Field(default=16, ge=1, le=256)
    budget_usd: float = Field(default=5.0, ge=0.0)
    max_retries: int = Field(default=5, ge=0, le=20)

    # §10.1 requires a cost estimate before a run can surprise someone with a
    # bill. Prices are configuration rather than constants on purpose: a table
    # of per-model prices baked into the package would be wrong within months,
    # and §13.2 asks that this still install cleanly in three years. Left unset,
    # the gate reports tokens and calls and says the cost is unknown.
    price_per_million_input: float | None = Field(default=None, ge=0.0)
    price_per_million_output: float | None = Field(default=None, ge=0.0)
    confirm_above_calls: int = Field(default=500, ge=0)


class GoldConfig(_Strict):
    """How much hand-coding to do.

    §7.2 proposed 300. That is enough to *measure* reliability but often not
    enough to *certify* it: the guarantee is limited by how many items land in
    the smaller class of each code. At 20% prevalence a 300-item sample gives
    30 positives per split, against the 36 that certifying a near-perfect coder
    needs at alpha = 0.10. 600 clears it with room for the codes that are rarer
    than average, which in qualitative work is most of them (§4).
    """

    n: int = Field(default=600, ge=1)
    enrich_rare_codes: bool = True

    double_coded_overlap: int = Field(default=150, ge=0)
    """Items a second coder also judges, giving the human-human ceiling (§12).

    Real work -- roughly a quarter of the gold sample coded twice -- but without
    it `check` cannot tell a researcher whether a failing code is a model
    problem or a codebook problem, which is the more useful of the two answers.
    """

    @model_validator(mode="after")
    def _overlap_fits(self) -> GoldConfig:
        if self.double_coded_overlap > self.n:
            raise ValueError(
                f"double_coded_overlap ({self.double_coded_overlap}) "
                f"cannot exceed gold.n ({self.n})"
            )
        return self


class ThresholdsConfig(_Strict):
    """Target class-conditional error rate, and the confidence it holds with.

    §7.2 proposed alpha = 0.05. Measured against the procedure, that target and
    the original 300-item gold sample are mutually incompatible: certifying a
    backend whose true per-class error is 2% needs about 142 accepted items in
    the smaller class, which at 20% prevalence is a gold sample of roughly
    1,400. At alpha = 0.10 the same backend needs 36, or about 360 gold items --
    which is the scale §0.4 actually promises.

    So the default is 0.10. It is a target error rate *within each class among
    the items the model decided*, with everything else routed to a person, and
    the validation report states it plainly so a reader can judge it. Set it
    to 0.05 if your gold sample is large enough to support it; `scruple check`
    tells you how many items that would take.
    """

    alpha: float = Field(default=0.10, gt=0.0, lt=1.0)
    delta: float = Field(default=0.05, gt=0.0, lt=1.0)


class ChunkingConfig(_Strict):
    """§10.3. The aggregation rule is frozen before calibration and recorded in
    the manifest; changing it invalidates calibration exactly as editing a code
    definition does."""

    max_tokens: int = Field(default=28_000, ge=256)
    overlap_tokens: int = Field(default=200, ge=0)
    aggregation: Literal["max", "mean"] = "max"

    @model_validator(mode="after")
    def _overlap_fits(self) -> ChunkingConfig:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        return self


class ProjectConfig(_Strict):
    version: int = 1
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    gold: GoldConfig = Field(default_factory=GoldConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)

    @field_validator("version")
    @classmethod
    def _known_version(cls, value: int) -> int:
        if value != 1:
            raise ValueError(f"unsupported config version {value}; this build understands 1")
        return value


def parse_config(text: str) -> ProjectConfig:
    """Parse and validate scruple.yml content."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValidationError(f"scruple.yml is not valid YAML: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValidationError("scruple.yml must be a mapping")
    try:
        return ProjectConfig.model_validate(raw)
    except PydanticValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in error['loc']) or 'config'}: {error['msg']}"
            for error in exc.errors()
        )
        raise ValidationError(f"scruple.yml is invalid -- {problems}") from exc


def load_config(path: Path) -> ProjectConfig:
    if not path.exists():
        raise ValidationError(
            f"no scruple.yml at {path}",
            hint="Run `scruple init` in the project directory first.",
        )
    return parse_config(path.read_text(encoding="utf-8"))


def config_template(*, backend: str = DEFAULT_BACKEND) -> str:
    """The scruple.yml `init` writes."""
    return f"""\
# scruple project configuration.
version: 1

corpus:
  path: data/corpus.csv
  text_column: text
  # id_column: respondent_id   # optional; row_NNN ids are generated if absent

splits:
  # Frozen at `scruple load`. Changing the seed invalidates all calibration.
  seed: 42
  dev: 0.10          # `try` may only draw from here
  calibration: 0.45  # fits the thresholds
  test: 0.45         # produces every number reported as a result

backend:
  name: {backend}
  model: typesafe-ai/jev
  # Leave null to use the backend default, or JEV_BASE_URL from .env.
  endpoint: null

engine:
  concurrency: 16
  budget_usd: 5.0    # a run projected above this asks before spending
  # Your backend's prices, so the cost gate can quote money rather than tokens.
  # Left unset, scruple reports estimated tokens and calls and says so.
  # price_per_million_input: 2.50
  # price_per_million_output: 10.00
  confirm_above_calls: 500

gold:
  # Certification is limited by the smaller class of each code, not by the
  # sample as a whole. `scruple check` reports the shortfall when a code fails
  # for want of evidence rather than accuracy.
  n: 600
  enrich_rare_codes: true   # extra draws for rare codes, IPW-corrected
  double_coded_overlap: 150 # items coded twice, giving the human-human ceiling

thresholds:
  # Target class-conditional error rate, per class, among items the model
  # decided. Lowering it to 0.05 roughly quadruples the gold sample you need:
  # `scruple check` reports the shortfall when a code fails for want of
  # evidence rather than accuracy.
  alpha: 0.10
  delta: 0.05   # confidence level for the guarantee

chunking:
  # Frozen before calibration. Changing any of this invalidates calibration.
  max_tokens: 28000
  overlap_tokens: 200
  aggregation: max
"""


def dump_config(config: ProjectConfig) -> str:
    """Serialise a config back to YAML, for `init` and for tests."""
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
