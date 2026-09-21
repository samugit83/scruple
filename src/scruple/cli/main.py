"""The scruple command line (plan §12).

All commands operate on the current project directory. Human-readable text on
stdout, `--json` available for every command -- and that JSON is a public,
versioned interface (§13.1), because the Phase 2 R package wraps this CLI as a
subprocess contract rather than reimplementing the statistics.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from .. import __version__
from ..backends import PRIVACY, available, build_backend
from ..codebook import Codebook, single_sentence_warnings
from ..config import DEFAULT_BACKEND
from ..corpus import Split, assign_splits
from ..engine import (
    Cache,
    CalibrationRecord,
    Engine,
    latest_run,
    require_no_drift,
    save_run,
)
from ..engine.apply import apply_calibration
from ..engine.check import run_check
from ..env import load_env
from ..errors import ExitCode, ProjectError, ScrupleError, ValidationError
from ..export import abstention_rows, coded_rows, default_filename, write_table
from ..gold import (
    GoldStore,
    ReviewStore,
    Task,
    code_tasks,
    enrichment_pool,
    plan_enriched,
    plan_overlap,
    plan_uniform,
    summarise,
)
from ..project import Project
from ..report import (
    ReportInputs,
    build_report,
    draw_labour_chart,
    most_informative,
)
from .display import check_json, check_table, envelope, error_envelope, show_error


@dataclass
class ComparisonRow:
    """One backend's showing in `scruple compare`. Part of the §13.1 contract."""

    backend: str
    model_version: str
    usable: list[str]
    codes: int
    share_needing_review: float
    cost_usd: float
    kappa: dict[str, float | None]


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Qualitative coding that codes what it is sure of, refuses the rest, and proves both.",
)
console = Console()
JsonFlag = Annotated[bool, typer.Option("--json", help="Machine-readable output.")]


def _fail(command: str, error: ScrupleError, as_json: bool) -> None:
    if as_json:
        typer.echo(error_envelope(command, error))
    else:
        show_error(console, error)
    raise typer.Exit(int(error.exit_code))


def _emit(command: str, data: dict[str, Any], as_json: bool) -> None:
    if as_json:
        typer.echo(envelope(command, data))


def _subset(codebook: Codebook, spec: str | None) -> Codebook:
    """Restrict a codebook to `--codes`, reporting an unknown id clearly."""
    if not spec:
        return codebook
    wanted = [c.strip() for c in spec.split(",") if c.strip()]
    try:
        return codebook.subset(wanted)
    except KeyError as exc:
        raise ValidationError(
            f"unknown code in --codes: {exc.args[0]}",
            hint=f"The codebook defines: {', '.join(codebook.ids)}.",
        ) from exc


def _project() -> Project:
    # A `.env` beside scruple.yml is read before the backend is built, so a
    # researcher never has to export shell variables to run the tool.
    load_env(Path.cwd())
    return Project.load(Path.cwd())


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"scruple {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True)
    ] = False,
) -> None:
    """scruple -- a reliability instrument for deductive qualitative coding."""
    del version


# ---------------------------------------------------------------------------
# init / load / splits / codebook
# ---------------------------------------------------------------------------


@app.command()
def init(
    backend: Annotated[str, typer.Option(help="Probability provider.")] = DEFAULT_BACKEND,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing project.")] = False,
    as_json: JsonFlag = False,
) -> None:
    """Create scruple.yml, .scruple/ and a template codebook.yml."""
    try:
        if backend not in available():
            raise ValidationError(
                f"unknown backend {backend!r}", hint=f"Available: {', '.join(available())}."
            )
        project = Project.create(Path.cwd(), backend=backend, force=force)
    except ScrupleError as error:
        _fail("init", error, as_json)
        return

    if as_json:
        _emit(
            "init",
            {"root": str(project.root), "backend": backend, "privacy": PRIVACY[backend]},
            True,
        )
        return

    console.print(f"created [bold]{project.config_path.name}[/bold] and a template codebook")
    console.print(f"backend: [bold]{backend}[/bold] — {PRIVACY[backend]}")
    if backend != "local":
        console.print(
            "[dim]Interview and survey text is usually personal data. If yours is, "
            "set backend.name: local so nothing leaves this machine — see docs/privacy.md.[/dim]"
        )
    console.print("\nnext: write your codes in codebook.yml, then `scruple load <file>`")


@app.command()
def load(
    file: Annotated[Path, typer.Argument(help="Corpus file or folder.")],
    text_col: Annotated[str, typer.Option("--text-col", help="Column holding the text.")] = "text",
    id_col: Annotated[str | None, typer.Option("--id-col")] = None,
    as_json: JsonFlag = False,
) -> None:
    """Register a corpus, validate it, and freeze the dev/calibration/test split."""
    try:
        project = _project()
        from ..corpus import load_corpus

        corpus = load_corpus(file, text_column=text_col, id_column=id_col)

        if project.has_splits():
            raise ProjectError(
                "this project already has a frozen split",
                hint="Run `scruple splits reset` to re-partition, which invalidates calibration.",
            )

        splits = assign_splits(
            corpus.ids,
            corpus_hash=corpus.hash,
            seed=project.config.splits.seed,
            dev=project.config.splits.dev,
            calibration=project.config.splits.calibration,
            test=project.config.splits.test,
        )
        project.ensure_state()
        splits.save(project.splits_path)

        # Record where the corpus actually is, so later commands find it.
        text = project.config_path.read_text(encoding="utf-8")
        text = text.replace("path: data/corpus.csv", f"path: {file}")
        text = text.replace("text_column: text", f"text_column: {text_col}")
        if id_col:
            text = text.replace(
                "  # id_column: respondent_id   # optional; row_NNN ids are generated if absent",
                f"  id_column: {id_col}",
            )
        project.config_path.write_text(text, encoding="utf-8")
    except ScrupleError as error:
        _fail("load", error, as_json)
        return

    stats = corpus.length_stats()
    counts = {s.value: n for s, n in splits.counts().items()}
    if as_json:
        _emit("load", {"items": len(corpus), "splits": counts, "length": stats}, True)
        return

    console.print(f"loaded [bold]{len(corpus):,}[/bold] items from {file}")
    console.print(
        f"length (characters): median {stats['median']:.0f}, "
        f"p95 {stats['p95']:.0f}, max {stats['max']:.0f}"
    )
    if stats["max"] > project.config.chunking.max_tokens * 3:
        console.print(
            "[dim]Some items are long enough to be chunked; the aggregation rule in "
            "scruple.yml is frozen before calibration — see docs/methodology.md.[/dim]"
        )
    console.print(
        f"split frozen (seed {splits.seed}): "
        + ", ".join(f"{name} {n:,}" for name, n in counts.items())
    )


splits_app = typer.Typer(help="Inspect or reset the frozen split.", no_args_is_help=True)
app.add_typer(splits_app, name="splits")


@splits_app.command("show")
def splits_show(as_json: JsonFlag = False) -> None:
    """Print the frozen partition."""
    try:
        project = _project()
        splits = project.splits()
    except ScrupleError as error:
        _fail("splits show", error, as_json)
        return

    counts = {s.value: n for s, n in splits.counts().items()}
    if as_json:
        _emit(
            "splits show",
            {"seed": splits.seed, "counts": counts, "corpus_hash": splits.corpus_hash},
            True,
        )
        return
    console.print(f"seed: {splits.seed}   frozen: {splits.created_at}")
    for name, count in counts.items():
        console.print(f"  {name:<12} {count:,}")


@splits_app.command("reset")
def splits_reset(
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation.")] = False,
    as_json: JsonFlag = False,
) -> None:
    """Re-partition the corpus. Invalidates all calibration."""
    try:
        project = _project()
        if not project.has_splits():
            raise ProjectError("this project has no frozen split yet")
        if not yes and not as_json:
            # §8.4: changing the seed invalidates calibration, and says so loudly.
            console.print(
                "[bold red]This invalidates every threshold and every reported number.[/bold red]"
            )
            console.print(
                "Items will move between calibration and test, so anything fitted on the "
                "old partition no longer describes this one."
            )
            if not typer.confirm("Re-partition anyway?"):
                raise typer.Abort()
        project.splits_path.unlink()
        if project.calibration_path.exists():
            project.calibration_path.unlink()
    except ScrupleError as error:
        _fail("splits reset", error, as_json)
        return

    _emit("splits reset", {"reset": True}, as_json)
    if not as_json:
        console.print("split and calibration cleared; run `scruple load` again")


codebook_app = typer.Typer(help="Work with codebook.yml.", no_args_is_help=True)
app.add_typer(codebook_app, name="codebook")


@codebook_app.command("check")
def codebook_check(as_json: JsonFlag = False) -> None:
    """Validate codebook.yml and print each code with its hash."""
    try:
        project = _project()
        codebook = project.codebook()
    except ScrupleError as error:
        _fail("codebook check", error, as_json)
        return

    warnings = single_sentence_warnings(codebook)
    if as_json:
        _emit(
            "codebook check",
            {
                "codes": [
                    {"id": c.id, "type": c.type.value, "hash": c.code_hash} for c in codebook
                ],
                "codebook_hash": codebook.hash,
                "warnings": warnings,
            },
            True,
        )
        return

    console.print(f"[bold]{len(codebook)}[/bold] codes, codebook hash {codebook.hash[:16]}")
    for code in codebook:
        console.print(f"  {code.id:<28} {code.code_hash[:12]}  {code.type.value}")
    for warning in warnings:
        console.print(f"[yellow]  {warning}[/yellow]")


# ---------------------------------------------------------------------------
# try
# ---------------------------------------------------------------------------


@app.command("try")
def try_command(
    n: Annotated[int, typer.Option("--n", help="Sample size, drawn from dev only.")] = 50,
    codes: Annotated[str | None, typer.Option("--codes", help="Comma-separated code ids.")] = None,
    as_json: JsonFlag = False,
) -> None:
    """Run on a random sample from the dev split. The iteration loop."""
    try:
        project = _project()
        codebook = _subset(project.codebook(), codes)
        corpus = project.corpus()
        splits = project.splits()
        splits.require_corpus(corpus.hash)

        dev_ids = splits.ids(Split.DEV)
        if not dev_ids:
            raise ProjectError("the dev split is empty")
        plan = plan_uniform(
            dev_ids,
            min(n, len(dev_ids)),
            split=Split.DEV.value,
            seed=project.config.splits.seed,
        )
        # §8.4: `try` MUST refuse to sample outside dev. Iterating on definitions
        # while looking at calibration or test data is fitting.
        splits.require(plan.item_ids, Split.DEV, what="`try`")

        sample = corpus.subset(plan.item_ids)
        with Cache(project.cache_path) as cache:
            engine = Engine(
                backend=build_backend(project.config.backend), config=project.config, cache=cache
            )
            result = engine.run(sample, codebook, corpus_path=project.corpus_path, purpose="try")
    except ScrupleError as error:
        _fail("try", error, as_json)
        return

    if as_json:
        _emit(
            "try",
            {
                "scores": result.scores,
                "cache_hits": result.manifest.cache_hits,
                "failures": result.manifest.failures,
            },
            True,
        )
        return

    by_id = sample.by_id()
    for item_id, per_code in list(result.scores.items())[:n]:
        text = by_id[item_id].text
        console.print(f"\n[dim]{item_id}[/dim]  {text[:160]}")
        for code_id, probability in per_code.items():
            shown = "failed" if probability is None else f"{probability:.2f}"
            console.print(f"    {code_id:<28} {shown}")
    console.print(
        f"\n[dim]{result.manifest.cache_hits} cache hits, "
        f"{result.manifest.calls} backend calls, {result.manifest.failures} failures[/dim]"
    )


# ---------------------------------------------------------------------------
# gold
# ---------------------------------------------------------------------------


@app.command()
def gold(
    n: Annotated[int | None, typer.Option("--n", help="How many items to sample.")] = None,
    coder: Annotated[
        str, typer.Option("--coder", help="Coder id, recorded per judgement.")
    ] = "coder_1",
    overlap: Annotated[
        int | None, typer.Option("--overlap", help="Second-coder subset size.")
    ] = None,
    enrich: Annotated[str | None, typer.Option("--enrich", help="Code id to enrich for.")] = None,
    as_json: JsonFlag = False,
) -> None:
    """Hand-code a random sample, blind. The model's output is never shown."""
    try:
        project = _project()
        codebook = project.codebook()
        corpus = project.corpus()
        splits = project.splits()
        splits.require_corpus(corpus.hash)
        store = GoldStore(project.gold_path)
        seed = project.config.splits.seed
        requested = n if n is not None else project.config.gold.n

        if overlap is not None:
            # §12: a second coder on an overlapping subset, giving the ceiling.
            existing = sorted({r.item_id for r in store.all_records()})
            if not existing:
                raise ProjectError(
                    "no first-coder sample to overlap with",
                    hint=(
                        "Run `scruple gold` first, then `scruple gold --overlap N --coder coder_2`."
                    ),
                )
            from ..gold.sampling import GoldPlan, GoldPlanItem

            lookup = {r.item_id: r for r in store.all_records()}
            base = GoldPlan(
                items=tuple(
                    GoldPlanItem(
                        item_id=i,
                        split=lookup[i].split,
                        stratum=lookup[i].stratum,
                        inclusion_probability=lookup[i].inclusion_probability,
                    )
                    for i in existing
                ),
                seed=seed,
                design="overlap",
            )
            chosen_ids = plan_overlap(base, overlap, seed=seed)
            plan_items = [i for i in base.items if i.item_id in set(chosen_ids)]
        elif enrich:
            if enrich not in codebook:
                raise ValidationError(f"no code {enrich!r} in the codebook")
            with Cache(project.cache_path) as cache:
                engine = Engine(
                    backend=build_backend(project.config.backend),
                    config=project.config,
                    cache=cache,
                )
                scored = engine.run(corpus, codebook.subset([enrich]), approved=True)
            probabilities = {
                item_id: per_code.get(enrich) for item_id, per_code in scored.scores.items()
            }
            already = sorted({r.item_id for r in store.all_records()})
            plan_items = []
            for split in (Split.CALIBRATION, Split.TEST):
                in_split = {i: p for i, p in probabilities.items() if splits.of(i) is split}
                pool = enrichment_pool(in_split, pool_fraction=0.2)
                base_pi = min(0.99, max(1e-6, len(already) / max(1, len(in_split))))
                supplement = plan_enriched(
                    pool=pool,
                    already_sampled=already,
                    n=max(1, requested // 2),
                    split=split.value,
                    seed=seed,
                    base_inclusion=base_pi,
                    code_id=enrich,
                )
                plan_items.extend(supplement.items)
        else:
            plan_items = []
            for split, share in (
                (Split.CALIBRATION, project.config.splits.calibration),
                (Split.TEST, project.config.splits.test),
            ):
                total = project.config.splits.calibration + project.config.splits.test
                want = max(1, round(requested * share / total))
                ids = splits.ids(split)
                plan_items.extend(
                    plan_uniform(ids, min(want, len(ids)), split=split.value, seed=seed).items
                )

        by_id = corpus.by_id()
        already_coded = store.coded_pairs(coder=coder)
        target_codes = list(codebook) if not enrich else [codebook[enrich]]
        tasks = [
            Task(item_id=item.item_id, text=by_id[item.item_id].text, code=code)
            for item in plan_items
            if item.item_id in by_id
            for code in target_codes
            if (item.item_id, code.id) not in already_coded
        ]
        if not tasks:
            raise ProjectError(
                "nothing left to code for this coder",
                hint="Every sampled item has already been judged. Use --coder for a second rater.",
            )
    except ScrupleError as error:
        _fail("gold", error, as_json)
        return

    if as_json:
        _emit("gold", {"tasks": len(tasks), "coder": coder}, True)
        return

    console.print(
        f"[bold]{len(tasks)}[/bold] judgements to make as [bold]{coder}[/bold]. "
        "Model output stays hidden, and that is recorded with every judgement."
    )
    session = code_tasks(tasks, show_probability=False, console=console, coder=coder)

    design = {i.item_id: i for i in plan_items}
    records = [
        store.record(
            item_id=j.task.item_id,
            code_id=j.task.code.id,
            label=j.label,
            coder=coder,
            split=design[j.task.item_id].split,
            stratum=design[j.task.item_id].stratum,
            inclusion_probability=design[j.task.item_id].inclusion_probability,
            sample_seed=project.config.splits.seed,
        )
        for j in session.judgements
    ]
    store.append(records)
    summarise(session, console)
    console.print(f"saved {len(records)} judgements to {project.gold_path.name}")


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


@app.command()
def check(
    alpha: Annotated[float | None, typer.Option("--alpha")] = None,
    delta: Annotated[float | None, typer.Option("--delta")] = None,
    resamples: Annotated[int, typer.Option("--resamples", help="Bootstrap resamples.")] = 2000,
    as_json: JsonFlag = False,
) -> None:
    """Fit thresholds on calibration, report on test. The decision point."""
    try:
        project = _project()
        codebook = project.codebook()
        corpus = project.corpus()
        splits = project.splits()
        splits.require_corpus(corpus.hash)
        store = GoldStore(project.gold_path)

        gold_ids = sorted({r.item_id for r in store.all_records()})
        if not gold_ids:
            raise ProjectError(
                "no gold sample yet",
                hint="Run `scruple gold` first — thresholds need hand-coded ground truth.",
            )

        backend = build_backend(project.config.backend)
        with Cache(project.cache_path) as cache:
            engine = Engine(backend=backend, config=project.config, cache=cache)
            scored = engine.run(
                corpus.subset(gold_ids),
                codebook,
                corpus_path=project.corpus_path,
                splits_seed=splits.seed,
                purpose="check",
            )

        report = run_check(
            codebook=codebook,
            config=project.config,
            splits=splits,
            gold=store,
            scores=scored.scores,
            backend_name=backend.name,
            model_version=backend.model_version(),
            corpus_hash=corpus.hash,
            alpha=alpha,
            delta=delta,
            bootstrap_resamples=resamples,
            # §14.4: handed in sealed, and never unsealed by the fitting path.
            sealed_test=splits.sealed(Split.TEST),
        )
        if report.record is not None:
            report.record.save(project.calibration_path)
        save_run(scored, project.runs_dir)
    except ScrupleError as error:
        _fail("check", error, as_json)
        return

    if as_json:
        _emit("check", check_json(report), True)
    else:
        check_table(report, console)

    if not report.usable:
        raise typer.Exit(int(ExitCode.NOT_CERTIFIED))


# ---------------------------------------------------------------------------
# run / review
# ---------------------------------------------------------------------------


@app.command()
def run(
    codes: Annotated[str | None, typer.Option("--codes")] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Approve an over-budget run.")] = False,
    retry_failed: Annotated[bool, typer.Option("--retry-failed")] = False,
    as_json: JsonFlag = False,
) -> None:
    """Code the full corpus. Refuses if the codebook or chunking has changed."""
    try:
        project = _project()
        codebook = project.codebook()
        corpus = project.corpus()
        splits = project.splits()
        splits.require_corpus(corpus.hash)
        record = CalibrationRecord.load(project.calibration_path)

        # §10.4: refuse on drift, naming what changed.
        require_no_drift(record, codebook, project.config)

        codebook = _subset(codebook, codes)

        target = corpus
        if retry_failed:
            previous = sorted(project.runs_dir.glob("*/failures.json"))
            if not previous:
                raise ProjectError("no recorded failures to retry")
            failed = {
                entry["item_id"] for entry in json.loads(previous[-1].read_text(encoding="utf-8"))
            }
            target = corpus.subset(sorted(failed))

        backend = build_backend(project.config.backend)
        with Cache(project.cache_path) as cache:
            engine = Engine(backend=backend, config=project.config, cache=cache)
            estimate = engine.estimate(target, codebook)
            if not as_json and not yes:
                console.print(estimate.describe())
            result = engine.run(
                target,
                codebook,
                corpus_path=project.corpus_path,
                splits_seed=splits.seed,
                approved=yes,
            )
        save_run(result, project.runs_dir)
    except ScrupleError as error:
        _fail("run", error, as_json)
        return

    if as_json:
        _emit(
            "run",
            {
                "run_id": result.run_id,
                "items": len(result.scores),
                "failures": result.manifest.failures,
                "cache_hits": result.manifest.cache_hits,
                "calls": result.manifest.calls,
            },
            True,
        )
        return

    console.print(
        f"coded {len(result.scores):,} items · {result.manifest.calls:,} calls · "
        f"{result.manifest.cache_hits:,} cache hits · {result.manifest.failures} failures"
    )
    if result.manifest.failures:
        console.print("[yellow]re-run with --retry-failed to pick up the failures[/yellow]")
    console.print("next: `scruple review` for the abstained items, then `scruple export`")


@app.command()
def review(
    coder: Annotated[str, typer.Option("--coder")] = "coder_1",
    limit: Annotated[int | None, typer.Option("--limit", help="Stop after this many.")] = None,
    as_json: JsonFlag = False,
) -> None:
    """Hand-code only the items the engine abstained on."""
    try:
        project = _project()
        codebook = project.codebook()
        corpus = project.corpus()
        record = CalibrationRecord.load(project.calibration_path)
        applied, _ = _current_coding(project, corpus, codebook, record)
        review_store = ReviewStore(project.review_path)
        done = set(review_store.latest())

        by_id = corpus.by_id()
        pending = [d for d in applied.abstentions() if (d.item_id, d.code_id) not in done]
        pending.sort(key=lambda d: abs((d.probability or 0.5) - 0.5), reverse=True)
        if limit is not None:
            pending = pending[:limit]
        if not pending:
            raise ProjectError("nothing to review — no unreviewed abstentions")

        tasks = [
            Task(
                item_id=d.item_id,
                text=by_id[d.item_id].text,
                code=codebook[d.code_id],
                probability=d.probability,
            )
            for d in pending
        ]
    except ScrupleError as error:
        _fail("review", error, as_json)
        return

    if as_json:
        _emit("review", {"pending": len(tasks)}, True)
        return

    console.print(f"[bold]{len(tasks)}[/bold] abstained judgements to make")
    session = code_tasks(tasks, show_probability=True, console=console, coder=coder)
    records = [
        review_store.record(
            item_id=j.task.item_id,
            code_id=j.task.code.id,
            label=j.label,
            coder=coder,
            probability=j.task.probability,
        )
        for j in session.judgements
    ]
    review_store.append(records)
    summarise(session, console)
    console.print(f"saved {len(records)} decisions to {project.review_path.name}")


def _current_coding(project: Project, corpus: Any, codebook: Any, record: CalibrationRecord):  # type: ignore[no-untyped-def]
    """Apply the fitted bands to the most recent **full** run on disk.

    A `check` also writes probabilities, but only for the gold sample. Picking
    those up here would quietly produce a coded.csv in which most of the corpus
    is unreviewed, which looks like a working export and is not one.
    """
    directory = latest_run(project.runs_dir, purpose="run")
    if directory is None:
        raise ProjectError(
            "no run to apply", hint="Run `scruple run` before reviewing or exporting."
        )
    scores = json.loads((directory / "probabilities.json").read_text(encoding="utf-8"))

    human: dict[tuple[str, str], int] = {}
    for record_ in GoldStore(project.gold_path).latest().values():
        human[(record_.item_id, record_.code_id)] = record_.label
    for key, decision in ReviewStore(project.review_path).latest().items():
        human[key] = decision.label

    applied = apply_calibration(
        corpus=corpus, codebook=codebook, record=record, scores=scores, human_labels=human
    )
    return applied, scores


# ---------------------------------------------------------------------------
# export / purge
# ---------------------------------------------------------------------------


@app.command()
def export(
    fmt: Annotated[str, typer.Option("--format", help="csv, parquet, stata or spss.")] = "csv",
    report: Annotated[bool, typer.Option("--report/--no-report")] = True,
    chart: Annotated[bool, typer.Option("--chart/--no-chart")] = True,
    as_json: JsonFlag = False,
) -> None:
    """Write coded.csv, abstentions.csv and validation_report.md."""
    try:
        project = _project()
        codebook = project.codebook()
        corpus = project.corpus()
        splits = project.splits()
        record = CalibrationRecord.load(project.calibration_path)
        applied, _ = _current_coding(project, corpus, codebook, record)

        project.out_dir.mkdir(parents=True, exist_ok=True)
        written = [
            write_table(
                coded_rows(corpus, applied),
                project.out_dir / default_filename("coded", fmt),
                fmt,
                integer_columns=codebook.ids,
            ),
            write_table(
                abstention_rows(corpus, applied),
                project.out_dir / default_filename("abstentions", fmt),
                fmt,
            ),
        ]

        if report:
            from ..engine.manifest import Manifest

            directory = latest_run(project.runs_dir, purpose="run")
            if directory is None:
                raise ProjectError("no run manifest to report from")
            manifest = Manifest.load(directory / "manifest.json")

            check_report = run_check(
                codebook=codebook,
                config=project.config,
                splits=splits,
                gold=GoldStore(project.gold_path),
                scores=json.loads((directory / "probabilities.json").read_text(encoding="utf-8")),
                backend_name=record.backend,
                model_version=record.model_version,
                corpus_hash=corpus.hash,
                bootstrap_resamples=2000,
            )

            chart_name: str | None = None
            if chart:
                best = most_informative(check_report.usable)
                if best is not None:
                    try:
                        draw_labour_chart(
                            calibrated=best.labour,
                            baseline=best.baseline,
                            path=project.out_dir / "reliability_vs_labour.png",
                            title=f"Reliability against human labour — {best.code_id}",
                        )
                        chart_name = "reliability_vs_labour.png"
                    except ValidationError:
                        # matplotlib is optional; a missing chart must not cost
                        # the researcher the report itself.
                        chart_name = None

            reviewed = len(ReviewStore(project.review_path).latest())
            text = build_report(
                ReportInputs(
                    check=check_report,
                    record=record,
                    manifest=manifest,
                    applied=applied,
                    corpus_size=len(corpus),
                    reviewed=reviewed,
                    scruple_version=__version__,
                    chart_path=chart_name,
                )
            )
            report_path = project.out_dir / "validation_report.md"
            report_path.write_text(text, encoding="utf-8")
            written.append(report_path)
    except ScrupleError as error:
        _fail("export", error, as_json)
        return

    if as_json:
        _emit("export", {"files": [str(p) for p in written]}, True)
        return
    for path in written:
        console.print(f"wrote {path}")


@app.command()
def compare(
    backends: Annotated[str, typer.Option("--backends", help="Comma-separated backend names.")],
    resamples: Annotated[int, typer.Option("--resamples")] = 500,
    yes: Annotated[bool, typer.Option("--yes", help="Approve over-budget runs.")] = False,
    as_json: JsonFlag = False,
) -> None:
    """Compare backends on your own gold sample (§8.8).

    Scores the gold items with each backend in turn and reports what each would
    certify, and at what coverage, against the same held-out human coding. This
    is how you find out what a local model costs you in hours, rather than
    taking a claim about it on trust.
    """
    try:
        project = _project()
        codebook = project.codebook()
        corpus = project.corpus()
        splits = project.splits()
        splits.require_corpus(corpus.hash)
        store = GoldStore(project.gold_path)

        gold_ids = sorted({r.item_id for r in store.all_records()})
        if not gold_ids:
            raise ProjectError(
                "no gold sample to compare against",
                hint="Run `scruple gold` first -- a comparison needs shared ground truth.",
            )
        names = [b.strip() for b in backends.split(",") if b.strip()]
        if len(names) < 2:
            raise ValidationError(
                "a comparison needs at least two backends",
                hint=f"Available: {', '.join(available())}.",
            )

        sample = corpus.subset(gold_ids)
        rows: list[ComparisonRow] = []
        for name in names:
            backend = build_backend(project.config.backend.model_copy(update={"name": name}))
            # One cache file per backend: serving one model's probability for
            # another would be a one-character mistake with no visible symptom.
            with Cache(project.state_dir / "compare" / f"{name}.sqlite") as cache:
                engine = Engine(backend=backend, config=project.config, cache=cache)
                scored = engine.run(sample, codebook, approved=yes)
            report = run_check(
                codebook=codebook,
                config=project.config,
                splits=splits,
                gold=store,
                scores=scored.scores,
                backend_name=name,
                model_version=backend.model_version(),
                corpus_hash=corpus.hash,
                bootstrap_resamples=resamples,
                sealed_test=splits.sealed(Split.TEST),
            )
            usage = getattr(backend, "usage", None)
            rows.append(
                ComparisonRow(
                    backend=name,
                    model_version=backend.model_version(),
                    usable=[c.code_id for c in report.usable],
                    codes=len(report.codes),
                    share_needing_review=report.corpus_share_needing_review,
                    cost_usd=usage.cost_usd if usage else 0.0,
                    kappa={c.code_id: (c.kappa.value if c.kappa else None) for c in report.codes},
                )
            )
    except ScrupleError as error:
        _fail("compare", error, as_json)
        return

    if as_json:
        _emit("compare", {"backends": [asdict(row) for row in rows]}, True)
        return

    console.print()
    for row in rows:
        console.print(
            f"[bold]{row.backend}[/bold] ({row.model_version}): "
            f"{len(row.usable)} of {row.codes} codes usable · "
            f"{row.share_needing_review * 100:.1f}% of items need you"
            + (f" · ${row.cost_usd:.2f}" if row.cost_usd else "")
        )
    console.print(
        "\n[dim]A local model is usually slower and less accurate than a hosted one. "
        "The difference shows up as coverage -- more items routed to you -- and this "
        "is where you see how much.[/dim]"
    )


@app.command()
def purge(
    gold_data: Annotated[bool, typer.Option("--gold", help="Delete the gold sample.")] = False,
    cache_data: Annotated[
        bool, typer.Option("--cache", help="Delete cached probabilities.")
    ] = False,
    everything: Annotated[bool, typer.Option("--all", help="Delete all derived state.")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    as_json: JsonFlag = False,
) -> None:
    """Delete personal data at rest. See docs/privacy.md."""
    try:
        project = _project()
        if not (gold_data or cache_data or everything):
            raise ValidationError(
                "nothing selected to purge", hint="Pass --gold, --cache or --all."
            )
        targets: list[str] = []
        if gold_data or everything:
            targets += ["gold.jsonl", "review.jsonl"]
        if cache_data or everything:
            targets.append("cache.sqlite")
        if everything:
            targets.append("runs/")

        if not yes and not as_json:
            console.print(f"about to delete: {', '.join(targets)}")
            console.print(
                "[dim]gold.jsonl and review.jsonl hold raw text and cannot be recovered.[/dim]"
            )
            if not typer.confirm("Delete?"):
                raise typer.Abort()

        removed = []
        if gold_data or everything:
            if GoldStore(project.gold_path).purge():
                removed.append("gold.jsonl")
            if ReviewStore(project.review_path).purge():
                removed.append("review.jsonl")
        if (cache_data or everything) and project.cache_path.exists():
            project.cache_path.unlink()
            removed.append("cache.sqlite")
        if everything and project.runs_dir.exists():
            import shutil

            shutil.rmtree(project.runs_dir)
            removed.append("runs/")
    except ScrupleError as error:
        _fail("purge", error, as_json)
        return

    _emit("purge", {"removed": removed}, as_json)
    if not as_json:
        console.print(f"deleted: {', '.join(removed) if removed else 'nothing was present'}")


def run_cli() -> None:  # pragma: no cover - console-script entry point
    try:
        app()
    except ScrupleError as error:
        show_error(console, error)
        sys.exit(int(error.exit_code))
