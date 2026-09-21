"""The codebook: the researcher's categories (plan §7.1).

The codebook is the intellectual contribution of the study. scruple never
invents, edits or extends it -- §5 puts inductive coding explicitly out of
scope. Everything here is validation and hashing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from ..hashing import code_hash, normalise_definition, sha256_hex

CODE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,40}$")
"""§7.1. Lowercase snake_case, 3 to 41 characters, so ids are safe as CSV column
names, SPSS variable names and shell arguments alike."""

MAX_EXAMPLES = 5
"""§7.1. More than a handful of examples crowds out the definition in the prompt
and makes the code_hash churn on edits that change nothing."""


class CodeType(StrEnum):
    """Code types. v1 implements `noul` only; the rest are v1.5 (§5)."""

    NOUL = "noul"
    SCORE = "score"
    CHOICE = "choice"

    @property
    def implemented(self) -> bool:
        return self is CodeType.NOUL


@dataclass(frozen=True)
class Code:
    """One named category, applied or not applied to each item."""

    id: str
    definition: str
    type: CodeType = CodeType.NOUL
    examples_yes: tuple[str, ...] = ()
    examples_no: tuple[str, ...] = ()

    @property
    def code_hash(self) -> str:
        """Hash of this code's meaning; part of the cache key (§9.4)."""
        return code_hash(self.definition, self.examples_yes, self.examples_no)

    @property
    def normalised_definition(self) -> str:
        return normalise_definition(self.definition)


@dataclass(frozen=True)
class Codebook:
    """A validated set of codes."""

    codes: tuple[Code, ...]
    version: int = 1
    path: str | None = None
    _by_id: dict[str, Code] = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_id", {code.id: code for code in self.codes})

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(code.id for code in self.codes)

    @property
    def hash(self) -> str:
        """Hash over every code, for the run manifest (§7.3)."""
        return sha256_hex("codebook", str(self.version), *(c.code_hash for c in self.codes))

    @property
    def code_hashes(self) -> dict[str, str]:
        """Per-code hashes, which the manifest records so drift can be named (§10.4)."""
        return {code.id: code.code_hash for code in self.codes}

    def __len__(self) -> int:
        return len(self.codes)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.codes)

    def __contains__(self, code_id: object) -> bool:
        return code_id in self._by_id

    def __getitem__(self, code_id: str) -> Code:
        try:
            return self._by_id[code_id]
        except KeyError:
            raise KeyError(f"no code {code_id!r}; codebook has {', '.join(self.ids)}") from None

    def subset(self, code_ids: list[str] | tuple[str, ...]) -> Codebook:
        """A codebook restricted to `code_ids`, preserving declaration order."""
        missing = [c for c in code_ids if c not in self._by_id]
        if missing:
            raise KeyError(f"unknown codes: {', '.join(missing)}")
        wanted = set(code_ids)
        return Codebook(
            codes=tuple(c for c in self.codes if c.id in wanted),
            version=self.version,
            path=self.path,
        )
