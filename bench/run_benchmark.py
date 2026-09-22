"""Phase 0: score a prepared corpus with several backends and compare (§11).

This produces the chart the kill criterion rests on — final reliability of the
whole corpus against human labour, per backend, with a random-selection
baseline — plus the calibration diagnostics and the human-human ceiling.

    uv run python bench/run_benchmark.py \\
        --dataset bench/data/<name> \\
        --backends jev,openai,local \\
        --out bench/results/<name>

A prepared dataset is a directory of corpus.csv, codebook.yml, gold.jsonl and
SOURCE.md; `bench/datasets.md` says how to build one. Datasets are never
committed (§14.11).

Report what the chart shows, **including if it shows nothing**. If the
calibrated curve sits close to the random baseline, the thesis has failed and
§3 says to stop building and publish that. Do not tune until it looks good.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from scruple.backends import build_backend
from scruple.codebook import load_codebook
from scruple.config import BackendConfig, parse_config
from scruple.corpus import Split, assign_splits, load_csv
from scruple.engine import Cache, Engine
from scruple.engine.check import run_check
from scruple.env import load_env
from scruple.gold import GoldStore
from scruple.report.chart import draw_labour_chart
from scruple.stats import Verdict, cohens_kappa

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class BackendResult:
    name: str
    model_version: str
    report: object
    seconds: float
    cost_usd: float
    calls: int


def score(dataset: Path, backend_name: str, out: Path):  # type: ignore[no-untyped-def]
    """Score one dataset with one backend and check it."""
    load_env(ROOT)
    codebook = load_codebook(dataset / "codebook.yml")
    corpus = load_csv(dataset / "corpus.csv", text_column="text", id_column="item_id")
    gold = GoldStore(dataset / "gold.jsonl")
    config = parse_config("engine:\n  concurrency: 16\n")
    splits = assign_splits(corpus.ids, corpus_hash=corpus.hash, seed=config.splits.seed)

    backend = build_backend(BackendConfig(name=backend_name))
    started = time.monotonic()
    # One cache per backend: a probability from one model must never be served
    # for another, and a shared file would make that a one-character mistake.
    with Cache(out / "cache" / f"{backend_name}.sqlite") as cache:
        engine = Engine(backend=backend, config=config, cache=cache)
        run = engine.run(corpus, codebook, approved=True)

    report = run_check(
        codebook=codebook,
        config=config,
        splits=splits,
        gold=gold,
        scores=run.scores,
        backend_name=backend.name,
        model_version=backend.model_version(),
        corpus_hash=corpus.hash,
        bootstrap_resamples=2000,
        sealed_test=splits.sealed(Split.TEST),
    )
    usage = getattr(backend, "usage", None)
    return BackendResult(
        name=backend_name,
        model_version=backend.model_version(),
        report=report,
        seconds=time.monotonic() - started,
        cost_usd=usage.cost_usd if usage else 0.0,
        calls=usage.calls if usage else 0,
    )


def human_ceiling(gold: GoldStore, code_id: str) -> float | None:
    first, second, shared = gold.double_coded(code_id)
    if not shared:
        return None
    return cohens_kappa(first, second).value


def write_report(results: list[BackendResult], dataset: Path, out: Path) -> Path:
    gold = GoldStore(dataset / "gold.jsonl")
    lines = [
        f"# Benchmark: {dataset.name}",
        "",
        "Final reliability of the whole corpus against human labour, per backend.",
        "The number that matters is the gap between each curve and the random",
        "baseline: that gap is what calibrated abstention buys, in hours at a",
        "fixed quality bar. If it is small, the thesis has failed (§3).",
        "",
        "## Backends",
        "",
        "| backend | model version | codes usable | corpus needing review | calls | cost | wall clock |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in results:
        report = result.report
        usable = len(report.usable)  # type: ignore[attr-defined]
        total = len(report.codes)  # type: ignore[attr-defined]
        share = report.corpus_share_needing_review  # type: ignore[attr-defined]
        lines.append(
            f"| `{result.name}` | `{result.model_version}` | {usable}/{total} | "
            f"{share * 100:.1f}% | {result.calls:,} | ${result.cost_usd:.2f} | "
            f"{result.seconds:.0f}s |"
        )

    lines += ["", "## Per code", ""]
    codes = [c.code_id for c in results[0].report.codes]  # type: ignore[attr-defined]
    header = "| code | human-human kappa |" + "".join(f" {r.name} |" for r in results)
    lines += [header, "|---" * (2 + len(results)) + "|"]
    for code_id in codes:
        ceiling = human_ceiling(gold, code_id)
        row = f"| `{code_id}` | {'--' if ceiling is None else f'{ceiling:.2f}'} |"
        for result in results:
            check = next(
                c
                for c in result.report.codes
                if c.code_id == code_id  # type: ignore[attr-defined]
            )
            if check.kappa is not None and check.kappa.value is not None:
                row += f" {check.kappa.value:.2f} ({check.selection.coverage * 100:.0f}% auto) |"
            else:
                row += f" {Verdict(check.selection.verdict).value} |"
        lines.append(row)

    lines += [
        "",
        "The human-human column is the ceiling. No automated coder should be",
        "expected to beat the agreement two trained people reach on the same",
        "code, and a result that appears to has probably measured something else.",
        "",
        "## Charts",
        "",
    ]
    for result in results:
        usable_checks = [c for c in result.report.usable if c.labour]  # type: ignore[attr-defined]
        if not usable_checks:
            lines.append(f"- `{result.name}`: no code was certified, so there is no curve.")
            continue
        best = max(
            usable_checks,
            key=lambda c: sum(
                max(0.0, a.kappa.value - b.kappa.value)
                for a, b in zip(c.labour, c.baseline, strict=False)
                if a.kappa.value is not None and b.kappa.value is not None
            ),
        )
        name = f"{result.name}_{best.code_id}.png"
        draw_labour_chart(
            calibrated=best.labour,
            baseline=best.baseline,
            path=out / name,
            title=f"{result.name} — {best.code_id}",
        )
        lines.append(f"![{result.name}]({name})")

    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="A prepared dataset directory.")
    parser.add_argument("--backends", default="jev", help="Comma-separated backend names.")
    parser.add_argument("--out", type=Path, required=True, help="Where to write results.")
    args = parser.parse_args()

    if not (args.dataset / "corpus.csv").exists():
        raise SystemExit(f"no prepared dataset at {args.dataset} -- see bench/datasets.md")
    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    for name in (b.strip() for b in args.backends.split(",") if b.strip()):
        print(f"scoring with {name}...")
        results.append(score(args.dataset, name, args.out))

    path = write_report(results, args.dataset, args.out)
    (args.out / "summary.json").write_text(
        json.dumps(
            [
                {
                    "backend": r.name,
                    "model_version": r.model_version,
                    "seconds": round(r.seconds, 1),
                    "cost_usd": r.cost_usd,
                    "calls": r.calls,
                }
                for r in results
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
