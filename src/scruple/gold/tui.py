"""Terminal coding for `gold` and `review` (plan §12).

One item at a time: `y` / `n` / `?` (skip) / `b` (back), with progress and
elapsed time.

The difference between the two modes is the whole point of the audit trail:

* **gold** -- model output MUST be hidden, and the tool records that it was
  hidden. A coder who has seen the machine's answer is no longer an independent
  second rater, and the entire reliability claim rests on that independence.
* **review** -- showing the probability is fine and useful. The researcher is
  adjudicating a case the machine already declined, not rating blind.

Terminal only in v1. A web UI is a Phase 2 nicety, not a prerequisite (§12).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..codebook import Code

SKIP = "?"
BACK = "b"
QUIT = "q"
YES = "y"
NO = "n"

PromptFn = Callable[[str], str]


@dataclass(frozen=True)
class Task:
    """One judgement to collect."""

    item_id: str
    text: str
    code: Code
    probability: float | None = None


@dataclass
class Judgement:
    """What the coder decided."""

    task: Task
    label: int


@dataclass
class Session:
    """Outcome of a coding session."""

    judgements: list[Judgement] = field(default_factory=list)
    skipped: list[Task] = field(default_factory=list)
    stopped_early: bool = False
    elapsed_seconds: float = 0.0

    @property
    def completed(self) -> int:
        return len(self.judgements)

    @property
    def seconds_per_judgement(self) -> float:
        return self.elapsed_seconds / self.completed if self.completed else 0.0


def _truncate(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... {len(text) - limit:,} more characters ...]"


def code_tasks(
    tasks: Sequence[Task],
    *,
    show_probability: bool,
    console: Console | None = None,
    prompt: PromptFn | None = None,
    coder: str = "coder_1",
) -> Session:
    """Walk the coder through `tasks`, collecting one judgement each."""
    out = Console() if console is None else console
    ask = prompt if prompt is not None else (lambda message: out.input(message))
    session = Session()
    started = time.monotonic()

    decisions: dict[int, int] = {}
    index = 0
    while 0 <= index < len(tasks):
        task = tasks[index]
        header = f"[{index + 1}/{len(tasks)}]  code: {task.code.id}"
        if show_probability and task.probability is not None:
            header += f"   model probability: {task.probability:.2f}"
        elif not show_probability:
            # Stated explicitly so the coder knows the blinding is deliberate,
            # and so a screen recording of the session evidences it.
            header += "   (model output hidden)"

        out.print()
        out.print(Panel(Text(_truncate(task.text)), title=header, title_align="left"))
        out.print(Text(task.code.normalised_definition, style="dim"))
        if task.code.examples_yes:
            out.print(Text("  applies e.g.: " + " | ".join(task.code.examples_yes), style="dim"))
        if task.code.examples_no:
            out.print(Text("  not e.g.: " + " | ".join(task.code.examples_no), style="dim"))

        elapsed = time.monotonic() - started
        rate = f"{elapsed / max(1, len(decisions)):.1f}s/item" if decisions else "--"
        answer = ask(
            f"  [{YES}]es / [{NO}]o / [{SKIP}] skip / [{BACK}]ack / [{QUIT}]uit  ({rate})  "
        )
        answer = (answer or "").strip().lower()

        if answer == QUIT:
            session.stopped_early = True
            break
        if answer == BACK:
            if index > 0:
                index -= 1
                decisions.pop(index, None)
            continue
        if answer == SKIP:
            session.skipped.append(task)
            index += 1
            continue
        if answer in (YES, NO):
            decisions[index] = 1 if answer == YES else 0
            index += 1
            continue
        out.print(Text(f"  did not understand {answer!r}", style="yellow"))

    session.elapsed_seconds = time.monotonic() - started
    session.judgements = [
        Judgement(task=tasks[i], label=label) for i, label in sorted(decisions.items())
    ]
    del coder
    return session


def summarise(session: Session, console: Console | None = None) -> None:
    """Print what the session achieved, including the rate for planning."""
    out = Console() if console is None else console
    out.print()
    out.print(
        f"coded {session.completed} judgement(s) in {session.elapsed_seconds:.0f}s "
        f"({session.seconds_per_judgement:.1f}s each)"
    )
    if session.skipped:
        out.print(f"skipped {len(session.skipped)} -- re-run to pick them up again")
    if session.stopped_early:
        out.print("stopped early; progress so far has been saved")
