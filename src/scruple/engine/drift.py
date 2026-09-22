"""Drift detection (§10.4).

`run` refuses to proceed if any code hash, the chunking settings, or the
aggregation rule differ from those recorded at calibration time, and the error
names what changed.

Why refuse rather than warn: the §8.6 guarantee is a statement about the
thresholds that were fitted to *those* definitions on *those* aggregated scores.
Reuse them against an edited codebook and the guarantee still gets printed in
the report, but it is no longer true of anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..codebook import Codebook
from ..config import ProjectConfig
from ..errors import DriftError
from .calibration import CalibrationRecord


@dataclass
class Drift:
    """What has changed since calibration."""

    changed_codes: list[str] = field(default_factory=list)
    added_codes: list[str] = field(default_factory=list)
    removed_codes: list[str] = field(default_factory=list)
    settings: list[str] = field(default_factory=list)

    @property
    def blocks_run(self) -> bool:
        """Added codes alone do not block: they are simply not calibrated yet.

        A new code has no thresholds, so it cannot be auto-coded and cannot make
        a false claim. Silently ignoring an *edited* code is the dangerous case.
        """
        return bool(self.changed_codes or self.settings)

    @property
    def any(self) -> bool:
        return bool(self.changed_codes or self.added_codes or self.removed_codes or self.settings)

    def describe(self) -> str:
        parts = []
        if self.changed_codes:
            parts.append(f"edited code definitions: {', '.join(sorted(self.changed_codes))}")
        if self.settings:
            parts.append(f"changed settings: {'; '.join(self.settings)}")
        if self.removed_codes:
            parts.append(f"codes removed: {', '.join(sorted(self.removed_codes))}")
        if self.added_codes:
            parts.append(f"codes added since calibration: {', '.join(sorted(self.added_codes))}")
        return " | ".join(parts)


def chunking_fingerprint(config: ProjectConfig) -> dict[str, object]:
    """The chunking settings that calibration depends on (§10.3)."""
    return {
        "max_tokens": config.chunking.max_tokens,
        "overlap_tokens": config.chunking.overlap_tokens,
        "aggregation": config.chunking.aggregation,
    }


def detect_drift(record: CalibrationRecord, codebook: Codebook, config: ProjectConfig) -> Drift:
    """Compare the live project against what calibration was fitted to."""
    drift = Drift()

    live = codebook.code_hashes
    for code_id, code_hash in live.items():
        recorded = record.code_hashes.get(code_id)
        if recorded is None:
            drift.added_codes.append(code_id)
        elif recorded != code_hash:
            drift.changed_codes.append(code_id)
    for code_id in record.code_hashes:
        if code_id not in live:
            drift.removed_codes.append(code_id)

    live_chunking = chunking_fingerprint(config)
    for key, value in live_chunking.items():
        recorded_value = record.chunking.get(key)
        if recorded_value != value:
            drift.settings.append(f"chunking.{key}: {recorded_value!r} -> {value!r}")

    if record.splits_seed != config.splits.seed:
        drift.settings.append(f"splits.seed: {record.splits_seed} -> {config.splits.seed}")
    if record.backend != config.backend.name:
        drift.settings.append(f"backend.name: {record.backend!r} -> {config.backend.name!r}")

    return drift


def require_no_drift(record: CalibrationRecord, codebook: Codebook, config: ProjectConfig) -> Drift:
    """Raise if calibration can no longer be trusted (§10.4)."""
    drift = detect_drift(record, codebook, config)
    if drift.blocks_run:
        raise DriftError(
            f"calibration is stale -- {drift.describe()}",
            hint=(
                "Thresholds were fitted to the previous definitions and settings, so reusing "
                "them would print a guarantee that is no longer true. Re-run `scruple check`."
            ),
        )
    return drift
