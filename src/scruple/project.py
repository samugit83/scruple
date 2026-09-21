"""The on-disk project (plan §6).

    myproject/
      scruple.yml              project config
      codebook.yml             the researcher's codes
      data/corpus.csv          input, untouched
      .scruple/
        splits.json            frozen dev/calibration/test assignment + seed
        cache.sqlite           (item, code, backend) -> probability. Hashes only.
        gold.jsonl             human-coded gold sample, append-only
        review.jsonl           human decisions on abstained items
        calibration.json       fitted thresholds + the hashes they were fitted at
        runs/<run_id>/         raw probabilities + run manifest
      out/
        coded.csv, abstentions.csv, validation_report.md

`gold.jsonl` and `review.jsonl` hold raw text and are therefore personal data at
rest; `cache.sqlite` stores hashes only, never text (§6, §9.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .codebook import Codebook, load_codebook
from .config import ProjectConfig, config_template, load_config
from .corpus import Corpus, Splits, load_corpus
from .errors import ProjectError

STATE_DIR = ".scruple"
CONFIG_NAME = "scruple.yml"
CODEBOOK_NAME = "codebook.yml"


@dataclass
class Project:
    """Paths and lazily loaded state for one project directory."""

    root: Path
    config: ProjectConfig
    _codebook: Codebook | None = field(default=None, repr=False)
    _corpus: Corpus | None = field(default=None, repr=False)
    _splits: Splits | None = field(default=None, repr=False)

    # -- paths ------------------------------------------------------------
    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_NAME

    @property
    def codebook_path(self) -> Path:
        return self.root / CODEBOOK_NAME

    @property
    def state_dir(self) -> Path:
        return self.root / STATE_DIR

    @property
    def splits_path(self) -> Path:
        return self.state_dir / "splits.json"

    @property
    def cache_path(self) -> Path:
        return self.state_dir / "cache.sqlite"

    @property
    def gold_path(self) -> Path:
        return self.state_dir / "gold.jsonl"

    @property
    def review_path(self) -> Path:
        return self.state_dir / "review.jsonl"

    @property
    def calibration_path(self) -> Path:
        return self.state_dir / "calibration.json"

    @property
    def runs_dir(self) -> Path:
        return self.state_dir / "runs"

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    @property
    def corpus_path(self) -> Path:
        return self.root / self.config.corpus.path

    # -- lifecycle --------------------------------------------------------
    @classmethod
    def create(cls, root: Path, *, backend: str, force: bool = False) -> Project:
        """Write a fresh project skeleton (`scruple init`)."""
        root.mkdir(parents=True, exist_ok=True)
        config_path = root / CONFIG_NAME
        if config_path.exists() and not force:
            raise ProjectError(
                f"{config_path} already exists",
                hint="Pass --force to overwrite it.",
            )

        from .codebook import TEMPLATE as CODEBOOK_TEMPLATE

        config_path.write_text(config_template(backend=backend), encoding="utf-8")
        codebook_path = root / CODEBOOK_NAME
        if not codebook_path.exists() or force:
            codebook_path.write_text(CODEBOOK_TEMPLATE, encoding="utf-8")

        (root / STATE_DIR).mkdir(exist_ok=True)
        (root / STATE_DIR / "runs").mkdir(exist_ok=True)
        (root / "data").mkdir(exist_ok=True)
        (root / "out").mkdir(exist_ok=True)
        # Derived state is never the study's record; keep it out of git for the
        # user, since gold.jsonl is personal data (§6).
        (root / STATE_DIR / ".gitignore").write_text("*\n", encoding="utf-8")

        return cls(root=root, config=load_config(config_path))

    @classmethod
    def load(cls, root: Path | None = None) -> Project:
        """Load the project rooted at `root` (default: the current directory)."""
        root = Path.cwd() if root is None else root
        return cls(root=root, config=load_config(root / CONFIG_NAME))

    # -- lazily loaded state ---------------------------------------------
    def codebook(self) -> Codebook:
        if self._codebook is None:
            self._codebook = load_codebook(self.codebook_path)
        return self._codebook

    def corpus(self) -> Corpus:
        if self._corpus is None:
            self._corpus = load_corpus(
                self.corpus_path,
                text_column=self.config.corpus.text_column,
                id_column=self.config.corpus.id_column,
            )
        return self._corpus

    def splits(self) -> Splits:
        """The frozen split. Immutable once written (§8.4)."""
        if self._splits is None:
            self._splits = Splits.load(self.splits_path)
        return self._splits

    def has_splits(self) -> bool:
        return self.splits_path.exists()

    def ensure_state(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
