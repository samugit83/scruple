"""The calibration record: what `check` decided, and what `run` must honour.

Cheap codebook revision is a selling point of this tool -- but revision
invalidates prior calibration, and silently reusing stale thresholds would
produce an invalid guarantee. So `check` writes down exactly what it fitted
against, and `run` refuses to proceed if any of it has changed (§10.4).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..errors import ProjectError
from ..stats import ReasonCode, ThresholdPair, ThresholdSelection, Verdict


@dataclass
class CodeCalibration:
    """The fitted outcome for one code."""

    code_id: str
    verdict: str
    reason: str | None
    t_lo: float | None
    t_hi: float | None
    coverage: float | None
    held_out_kappa: float | None
    kappa_ci: list[float] | None
    n_gold: int
    n_gold_positive: int
    realised_risk_positive: float | None = None
    realised_risk_negative: float | None = None
    message: str = ""

    @property
    def usable(self) -> bool:
        return Verdict(self.verdict).usable

    @property
    def pair(self) -> ThresholdPair | None:
        if self.t_lo is None or self.t_hi is None:
            return None
        return ThresholdPair(t_lo=self.t_lo, t_hi=self.t_hi)

    @classmethod
    def from_selection(
        cls,
        code_id: str,
        selection: ThresholdSelection,
        *,
        realised: dict[int, float | None] | None = None,
    ) -> CodeCalibration:
        kappa = selection.held_out_kappa
        return cls(
            code_id=code_id,
            verdict=Verdict(selection.verdict).value,
            reason=ReasonCode(selection.reason).value if selection.reason else None,
            t_lo=selection.selected.t_lo if selection.selected else None,
            t_hi=selection.selected.t_hi if selection.selected else None,
            coverage=selection.coverage,
            held_out_kappa=kappa.value if kappa else None,
            kappa_ci=list(kappa.ci) if kappa and kappa.ci else None,
            n_gold=selection.n_gold,
            n_gold_positive=selection.n_gold_positive,
            realised_risk_positive=(realised or {}).get(1),
            realised_risk_negative=(realised or {}).get(0),
            message=selection.message,
        )


@dataclass
class CalibrationRecord:
    """Everything `run` needs, and everything drift detection compares against."""

    created_at: str
    backend: str
    model_version: str
    codebook_hash: str
    code_hashes: dict[str, str]
    corpus_hash: str
    splits_seed: int
    chunking: dict[str, Any]
    alpha: float
    delta: float
    delta_per_candidate: float
    gold_n: int
    gold_n_double_coded: int
    exact_guarantee: bool
    codes: dict[str, CodeCalibration] = field(default_factory=dict)
    human_ceiling: dict[str, float | None] = field(default_factory=dict)
    version: int = 1

    @property
    def usable_codes(self) -> tuple[str, ...]:
        return tuple(cid for cid, c in self.codes.items() if c.usable)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["codes"] = {cid: asdict(c) for cid, c in self.codes.items()}
        return data

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationRecord:
        if data.get("version") != 1:
            raise ProjectError(f"unsupported calibration version {data.get('version')!r}")
        payload = dict(data)
        raw_codes = payload.pop("codes", {}) or {}
        known = set(CodeCalibration.__dataclass_fields__)
        codes = {
            cid: CodeCalibration(**{k: v for k, v in body.items() if k in known})
            for cid, body in raw_codes.items()
        }
        fields = set(cls.__dataclass_fields__)
        return cls(codes=codes, **{k: v for k, v in payload.items() if k in fields})

    @classmethod
    def load(cls, path: Path) -> CalibrationRecord:
        if not path.exists():
            raise ProjectError(
                "this project has not been calibrated yet",
                hint="Run `scruple gold` to collect a gold sample, then `scruple check`.",
            )
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise ProjectError(f"{path} is not valid JSON: {exc}") from exc


def new_record(
    *,
    backend: str,
    model_version: str,
    codebook_hash: str,
    code_hashes: dict[str, str],
    corpus_hash: str,
    splits_seed: int,
    chunking: dict[str, Any],
    alpha: float,
    delta: float,
    delta_per_candidate: float,
    gold_n: int,
    gold_n_double_coded: int,
    exact_guarantee: bool,
) -> CalibrationRecord:
    return CalibrationRecord(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        backend=backend,
        model_version=model_version,
        codebook_hash=codebook_hash,
        code_hashes=dict(code_hashes),
        corpus_hash=corpus_hash,
        splits_seed=splits_seed,
        chunking=dict(chunking),
        alpha=alpha,
        delta=delta,
        delta_per_candidate=delta_per_candidate,
        gold_n=gold_n,
        gold_n_double_coded=gold_n_double_coded,
        exact_guarantee=exact_guarantee,
    )
