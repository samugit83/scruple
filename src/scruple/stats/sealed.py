"""Sealing test data against the fitting code (plan §8.4).

The three-way split only protects the result if nothing in the fitting path can
read the test split. Relying on discipline is not enough: the mistake is a
one-character slip and the consequence is a number that looks publishable and
is not. So test data is handed to the fitter inside a wrapper that raises on
every ordinary means of reading it, and unsealing requires naming a purpose
from a short allowlist that excludes fitting.
"""

from __future__ import annotations

from typing import Any, Generic, NoReturn, TypeVar

T = TypeVar("T")

ALLOWED_PURPOSES = frozenset({"reporting", "final_evaluation"})
"""Purposes for which test data may legitimately be read: reporting the final
numbers, and the §8.8 simulation that produces them. Never fitting."""


class SealedDataError(RuntimeError):
    """Raised when sealed data is read outside an allowed purpose."""


class Sealed(Generic[T]):
    """A payload that fitting code must not read.

    Every read path raises. ``unseal(purpose=...)`` is the only way in, it
    validates the purpose, and it records the access so the run manifest can
    show when the test split was touched.
    """

    __slots__ = ("_label", "_log", "_payload")

    def __init__(self, payload: T, *, label: str) -> None:
        object.__setattr__(self, "_payload", payload)
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_log", [])

    @property
    def label(self) -> str:
        """Which split this is. Metadata only: it reveals nothing about the data."""
        return str(object.__getattribute__(self, "_label"))

    @property
    def unseal_log(self) -> tuple[str, ...]:
        """The purposes for which this payload has been unsealed, in order."""
        return tuple(object.__getattribute__(self, "_log"))

    def unseal(self, *, purpose: str) -> T:
        """Read the payload for an explicitly named, allowed purpose."""
        if purpose not in ALLOWED_PURPOSES:
            raise SealedDataError(
                f"cannot unseal the {self.label!r} split for purpose {purpose!r}; "
                f"allowed purposes are {sorted(ALLOWED_PURPOSES)}. "
                "Fitting on this split would invalidate every number computed from it (plan §8.4)."
            )
        log: list[str] = object.__getattribute__(self, "_log")
        log.append(purpose)
        payload: T = object.__getattribute__(self, "_payload")
        return payload

    def _refuse(self, how: str) -> NoReturn:
        raise SealedDataError(
            f"the {self.label!r} split is sealed and cannot be read by {how}; "
            "call unseal(purpose=...) if this really is a reporting step (plan §8.4)."
        )

    def __getattr__(self, name: str) -> NoReturn:
        # Let genuine dunder probes fail normally so unrelated protocol checks
        # (hasattr on __iter__, say) do not surface as a sealing error.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        self._refuse(f"attribute access ({name!r})")

    def __getitem__(self, key: object) -> NoReturn:
        self._refuse("indexing")

    def __len__(self) -> NoReturn:
        self._refuse("len()")

    def __iter__(self) -> NoReturn:
        self._refuse("iteration")

    def __contains__(self, item: object) -> NoReturn:
        self._refuse("containment testing")

    def __bool__(self) -> NoReturn:
        self._refuse("truth testing")

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._refuse("calling")

    def __reduce__(self) -> NoReturn:
        self._refuse("pickling")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        self._refuse("deep copying")

    def __repr__(self) -> str:
        return f"<Sealed {self.label!r} split: contents withheld (plan §8.4)>"
