"""Project configuration: `scruple.yml` (plan §7.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from .errors import ValidationError

DEFAULT_BACKEND = "jev"
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
    model: str = "jev-latest"
    endpoint: str | None = None


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
    n: int = Field(default=300, ge=1)
    enrich_rare_codes: bool = True
    double_coded_overlap: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def _overlap_fits(self) -> GoldConfig:
        if self.double_coded_overlap > self.n:
            raise ValueError(
                f"double_coded_overlap ({self.double_coded_overlap}) "
                f"cannot exceed gold.n ({self.n})"
            )
        return self


class ThresholdsConfig(_Strict):
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
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
  model: jev-latest
  endpoint: null     # override to route through a gateway

engine:
  concurrency: 16
  budget_usd: 5.0    # a run projected above this asks before spending
  # Your backend's prices, so the cost gate can quote money rather than tokens.
  # Left unset, scruple reports estimated tokens and calls and says so.
  # price_per_million_input: 2.50
  # price_per_million_output: 10.00
  confirm_above_calls: 500

gold:
  n: 300
  enrich_rare_codes: true   # extra draws for rare codes, IPW-corrected
  double_coded_overlap: 100 # items coded twice, giving the human-human ceiling

thresholds:
  alpha: 0.05   # target class-conditional error rate, per class
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
