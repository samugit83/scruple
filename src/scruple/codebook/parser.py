"""Parsing and validating `codebook.yml` (§7.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..errors import ValidationError
from .model import CODE_ID_PATTERN, MAX_EXAMPLES, Code, Codebook, CodeType

TEMPLATE = """\
# The codes you are looking for in the corpus. This file is yours: scruple
# validates it and never edits or extends it.
#
# Write each definition so that someone else could apply it to an ambiguous
# answer and reach the same verdict you would. That is the whole job.
version: 1
codes:
  access_barrier:
    type: noul
    definition: >
      The respondent cites a practical obstacle to attending:
      time, transport, clinic opening hours, distance or cost.
    examples_yes:
      - "the clinic closes at 5 and I work two jobs"
    examples_no:
      - "I just don't think it's necessary"
"""


class _NoDuplicateKeyLoader(yaml.SafeLoader):
    """A loader that refuses duplicate mapping keys.

    PyYAML silently keeps the last of two identical keys. §7.1 requires
    duplicate code ids to be an error, and silently dropping one of two
    definitions of the same code is exactly the kind of quiet data loss that
    invalidates a study without anyone noticing.
    """

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ValidationError(
                    f"duplicate key {key!r} at line {key_node.start_mark.line + 1}",
                    hint="Each code id may appear only once.",
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _require_mapping(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{what} must be a mapping, got {type(value).__name__}")
    return value


def _validate_examples(raw: Any, code_id: str, field: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValidationError(f"{code_id}.{field} must be a list of strings")
    if len(raw) > MAX_EXAMPLES:
        raise ValidationError(
            f"{code_id}.{field} has {len(raw)} examples; the maximum is {MAX_EXAMPLES}",
            hint="More examples crowd out the definition without improving it.",
        )
    out = []
    for example in raw:
        if not isinstance(example, str) or not example.strip():
            raise ValidationError(f"{code_id}.{field} contains an empty or non-string example")
        out.append(example.strip())
    return tuple(out)


def parse_code(code_id: str, raw: Any) -> Code:
    """Validate one code entry."""
    if not CODE_ID_PATTERN.match(code_id):
        raise ValidationError(
            f"invalid code id {code_id!r}",
            hint=(
                "Ids must be lowercase letters, digits and underscores, start with a "
                "letter, and be 3 to 41 characters long."
            ),
        )
    body = _require_mapping(raw, f"code {code_id!r}")

    unknown = set(body) - {"type", "definition", "examples_yes", "examples_no"}
    if unknown:
        raise ValidationError(f"{code_id} has unknown keys: {', '.join(sorted(unknown))}")

    definition = body.get("definition")
    if not isinstance(definition, str) or not definition.strip():
        raise ValidationError(
            f"{code_id} needs a non-empty definition",
            hint="One sentence describing what makes an item positive.",
        )

    raw_type = body.get("type", "noul")
    try:
        code_type = CodeType(raw_type)
    except ValueError:
        raise ValidationError(
            f"{code_id} has unknown type {raw_type!r}",
            hint=f"Valid types are: {', '.join(t.value for t in CodeType)}.",
        ) from None
    if not code_type.implemented:
        raise ValidationError(
            f"{code_id} uses type {code_type.value!r}, which is not implemented in v1",
            hint="v1 supports binary `noul` codes only; `score` and `choice` are v1.5 (§5).",
        )

    return Code(
        id=code_id,
        definition=definition.strip(),
        type=code_type,
        examples_yes=_validate_examples(body.get("examples_yes"), code_id, "examples_yes"),
        examples_no=_validate_examples(body.get("examples_no"), code_id, "examples_no"),
    )


def parse_codebook(text: str, *, path: str | None = None) -> Codebook:
    """Parse and validate codebook YAML."""
    try:
        raw = yaml.load(text, Loader=_NoDuplicateKeyLoader)
    except yaml.YAMLError as exc:
        raise ValidationError(f"codebook is not valid YAML: {exc}") from exc

    if raw is None:
        raise ValidationError("codebook is empty")
    document = _require_mapping(raw, "codebook")

    version = document.get("version", 1)
    if version != 1:
        raise ValidationError(
            f"unsupported codebook version {version!r}",
            hint="This build understands version 1.",
        )

    codes_raw = document.get("codes")
    if codes_raw is None:
        raise ValidationError("codebook has no `codes:` section")
    codes_map = _require_mapping(codes_raw, "`codes`")
    if not codes_map:
        raise ValidationError("codebook defines no codes")

    codes = tuple(parse_code(str(cid), body) for cid, body in codes_map.items())
    return Codebook(codes=codes, version=1, path=path)


def load_codebook(path: Path) -> Codebook:
    """Read and validate a codebook from disk."""
    if not path.exists():
        raise ValidationError(
            f"no codebook at {path}",
            hint="Run `scruple init` to write a template.",
        )
    return parse_codebook(path.read_text(encoding="utf-8"), path=str(path))


def single_sentence_warnings(codebook: Codebook) -> list[str]:
    """Advisory checks (§7.1 SHOULDs). These never block a run.

    A definition spanning several sentences usually means several codes, and a
    code two humans read differently is a codebook problem rather than a model
    problem (§12) -- worth saying early, not worth refusing over.
    """
    warnings = []
    for code in codebook:
        text = code.normalised_definition
        sentences = [s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        if len(sentences) > 2:
            warnings.append(
                f"{code.id}: definition runs to {len(sentences)} sentences. A code that "
                "needs a paragraph is often two codes."
            )
        if len(text) > 400:
            warnings.append(f"{code.id}: definition is {len(text)} characters; consider trimming.")
    return warnings
