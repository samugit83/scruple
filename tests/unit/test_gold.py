"""Unit tests for gold sampling, storage and the coding interface (§8.5, §12)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from scruple.codebook import Code
from scruple.errors import ProjectError, ValidationError
from scruple.gold import (
    ENRICHED,
    UNIFORM,
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

IDS = tuple(f"r{i:04d}" for i in range(1000))
CODE = Code(id="access_barrier", definition="A practical obstacle.")


class TestUniformSampling:
    def test_draws_the_requested_number(self) -> None:
        assert len(plan_uniform(IDS, 150, split="calibration", seed=42)) == 150

    def test_inclusion_probability_is_exact(self) -> None:
        plan = plan_uniform(IDS, 150, split="calibration", seed=42)
        assert all(item.inclusion_probability == pytest.approx(0.15) for item in plan.items)
        assert all(item.weight == pytest.approx(1 / 0.15) for item in plan.items)

    def test_is_reproducible_from_the_seed(self) -> None:
        """§8.6: `gold` MUST sample with a recorded seed and MUST refuse a
        hand-picked sample -- the guarantee assumes a random sample."""
        a = plan_uniform(IDS, 100, split="calibration", seed=42)
        b = plan_uniform(IDS, 100, split="calibration", seed=42)
        assert a.item_ids == b.item_ids

    def test_a_different_seed_draws_a_different_sample(self) -> None:
        a = plan_uniform(IDS, 100, split="calibration", seed=42)
        b = plan_uniform(IDS, 100, split="calibration", seed=7)
        assert a.item_ids != b.item_ids

    def test_splits_are_sampled_independently(self) -> None:
        # Calibration and test draws must not mirror each other.
        cal = plan_uniform(IDS, 100, split="calibration", seed=42)
        test = plan_uniform(IDS, 100, split="test", seed=42)
        assert cal.item_ids != test.item_ids

    def test_the_design_is_recorded_for_the_report(self) -> None:
        assert plan_uniform(IDS, 150, split="calibration", seed=42).design == "uniform(150/1000)"

    def test_duplicates_in_the_population_are_ignored(self) -> None:
        plan = plan_uniform(("a", "a", "b", "c"), 3, split="dev", seed=1)
        assert len(set(plan.item_ids)) == 3

    def test_asking_for_more_than_exists_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="cannot draw 50 items") as caught:
            plan_uniform(IDS[:10], 50, split="calibration", seed=1)
        assert "gold.n" in (caught.value.hint or "")

    def test_an_empty_split_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="no items to sample"):
            plan_uniform((), 10, split="calibration", seed=1)

    def test_a_nonsense_size_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="at least 1"):
            plan_uniform(IDS, 0, split="calibration", seed=1)


class TestEnrichment:
    def probabilities(self) -> dict[str, float | None]:
        return {i: (0.9 if int(i[1:]) % 20 == 0 else 0.02) for i in IDS}

    def test_the_pool_is_the_top_ranked_items(self) -> None:
        pool = enrichment_pool(self.probabilities(), pool_fraction=0.05)
        assert len(pool) == 50
        assert all(int(i[1:]) % 20 == 0 for i in pool)

    def test_the_pool_has_a_known_size_whatever_the_distribution(self) -> None:
        # Defined by rank, not by an absolute cut, so the inclusion probability
        # can be computed exactly.
        flat = dict.fromkeys(IDS, 0.5)
        assert len(enrichment_pool(flat, pool_fraction=0.1)) == 100

    def test_items_without_a_probability_are_excluded(self) -> None:
        mixed: dict[str, float | None] = {"a": 0.9, "b": None, "c": 0.8}
        assert "b" not in enrichment_pool(mixed, pool_fraction=1.0)

    def test_no_probabilities_gives_an_empty_pool(self) -> None:
        assert enrichment_pool({"a": None}) == ()

    def test_an_invalid_pool_fraction_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="pool_fraction"):
            enrichment_pool(self.probabilities(), pool_fraction=0.0)

    def test_enrichment_raises_the_inclusion_probability(self) -> None:
        """§8.5: extra draws with elevated probability of being positive,
        recording the known inclusion probability of every sampled item."""
        uniform = plan_uniform(IDS, 150, split="calibration", seed=42)
        pool = enrichment_pool(self.probabilities(), pool_fraction=0.1)
        enriched = plan_enriched(
            pool=pool,
            already_sampled=uniform.item_ids,
            n=40,
            split="calibration",
            seed=42,
            base_inclusion=0.15,
            code_id="rare_code",
        )
        assert len(enriched) == 40
        assert all(i.stratum == ENRICHED for i in enriched.items)
        # Higher than uniform, so the IPW weight is correspondingly lower.
        assert enriched.items[0].inclusion_probability > 0.15
        assert enriched.items[0].weight < uniform.items[0].weight

    def test_enrichment_never_redraws_an_already_sampled_item(self) -> None:
        uniform = plan_uniform(IDS, 300, split="calibration", seed=42)
        pool = enrichment_pool(self.probabilities(), pool_fraction=0.5)
        enriched = plan_enriched(
            pool=pool,
            already_sampled=uniform.item_ids,
            n=40,
            split="calibration",
            seed=42,
            base_inclusion=0.3,
            code_id="rare_code",
        )
        assert not set(enriched.item_ids) & set(uniform.item_ids)

    def test_the_inclusion_probability_is_capped_at_one(self) -> None:
        pool = list(IDS[:10])
        enriched = plan_enriched(
            pool=pool,
            already_sampled=(),
            n=10,
            split="calibration",
            seed=1,
            base_inclusion=0.9,
            code_id="rare_code",
        )
        assert all(i.inclusion_probability <= 1.0 for i in enriched.items)

    def test_an_exhausted_pool_is_reported_clearly(self) -> None:
        with pytest.raises(ValidationError, match="no un-sampled items left"):
            plan_enriched(
                pool=list(IDS[:5]),
                already_sampled=IDS[:5],
                n=10,
                split="calibration",
                seed=1,
                base_inclusion=0.1,
                code_id="rare_code",
            )

    def test_the_design_string_records_what_was_done(self) -> None:
        enriched = plan_enriched(
            pool=list(IDS[:100]),
            already_sampled=(),
            n=20,
            split="calibration",
            seed=1,
            base_inclusion=0.15,
            code_id="rare_code",
        )
        assert "rare_code" in enriched.design
        assert "20/100" in enriched.design


class TestOverlap:
    def test_draws_a_subset_for_the_second_coder(self) -> None:
        plan = plan_uniform(IDS, 300, split="calibration", seed=42)
        overlap = plan_overlap(plan, 100, seed=42)
        assert len(overlap) == 100
        assert set(overlap) <= set(plan.item_ids)

    def test_is_reproducible(self) -> None:
        plan = plan_uniform(IDS, 300, split="calibration", seed=42)
        assert plan_overlap(plan, 100, seed=42) == plan_overlap(plan, 100, seed=42)

    def test_asking_for_more_than_the_sample_gives_the_whole_sample(self) -> None:
        plan = plan_uniform(IDS, 50, split="calibration", seed=42)
        assert len(plan_overlap(plan, 500, seed=42)) == 50


class TestGoldStore:
    def store(self, tmp_path: Path) -> GoldStore:
        return GoldStore(tmp_path / ".scruple" / "gold.jsonl")

    def add(
        self,
        store: GoldStore,
        item: str,
        label: int,
        *,
        coder: str = "coder_1",
        split: str = "calibration",
        pi: float = 0.15,
    ) -> None:
        store.append(
            [
                store.record(
                    item_id=item,
                    code_id="access_barrier",
                    label=label,
                    coder=coder,
                    split=split,
                    stratum=UNIFORM,
                    inclusion_probability=pi,
                    sample_seed=42,
                )
            ]
        )

    def test_records_are_blind_by_construction(self, tmp_path: Path) -> None:
        """§6: gold MUST record that model output was hidden -- always false.

        This is the audit trail that makes the study defensible. It is not a
        parameter, so no caller can set it to true by mistake.
        """
        store = self.store(tmp_path)
        self.add(store, "r0001", 1)
        record = store.all_records()[0]
        assert record.model_visible is False
        assert record.timestamp
        assert record.coder == "coder_1"
        assert record.sample_seed == 42

    def test_is_append_only(self, tmp_path: Path) -> None:
        store = self.store(tmp_path)
        self.add(store, "r0001", 1)
        self.add(store, "r0002", 0)
        assert len(store.all_records()) == 2

    def test_a_correction_is_a_new_line_and_the_history_survives(self, tmp_path: Path) -> None:
        store = self.store(tmp_path)
        self.add(store, "r0001", 1)
        self.add(store, "r0001", 0)
        assert len(store.all_records()) == 2, "the earlier judgement must stay on disk"
        assert store.latest()[("r0001", "access_barrier", "coder_1")].label == 0

    def test_labels_carry_ipw_weights(self, tmp_path: Path) -> None:
        store = self.store(tmp_path)
        self.add(store, "r0001", 1, pi=0.15)
        self.add(store, "r0002", 0, pi=0.60)
        ids, labels, weights = store.labels_for("access_barrier")
        assert ids == ("r0001", "r0002")
        assert np.allclose(labels, [1.0, 0.0])
        assert np.allclose(weights, [1 / 0.15, 1 / 0.60])

    def test_labels_filter_by_split(self, tmp_path: Path) -> None:
        # §8.4: calibration fits, test reports. They must never be mixed.
        store = self.store(tmp_path)
        self.add(store, "r0001", 1, split="calibration")
        self.add(store, "r0002", 0, split="test")
        assert store.labels_for("access_barrier", split="calibration")[0] == ("r0001",)
        assert store.labels_for("access_barrier", split="test")[0] == ("r0002",)

    def test_labels_filter_by_coder(self, tmp_path: Path) -> None:
        store = self.store(tmp_path)
        self.add(store, "r0001", 1, coder="coder_1")
        self.add(store, "r0002", 0, coder="coder_2")
        assert store.labels_for("access_barrier", coder="coder_2")[0] == ("r0002",)

    def test_coded_pairs_let_a_session_resume(self, tmp_path: Path) -> None:
        store = self.store(tmp_path)
        self.add(store, "r0001", 1)
        assert store.coded_pairs() == {("r0001", "access_barrier")}

    def test_coders_are_listed(self, tmp_path: Path) -> None:
        store = self.store(tmp_path)
        self.add(store, "r0001", 1, coder="alice")
        self.add(store, "r0002", 0, coder="bob")
        assert store.coders() == ("alice", "bob")

    def test_the_primary_coder_is_the_one_who_did_most_of_the_work(self, tmp_path: Path) -> None:
        store = self.store(tmp_path)
        self.add(store, "r0001", 1, coder="alice")
        self.add(store, "r0002", 1, coder="alice")
        self.add(store, "r0003", 0, coder="bob")
        assert store.primary_coder() == "alice"

    def test_no_records_means_no_primary_coder(self, tmp_path: Path) -> None:
        assert self.store(tmp_path).primary_coder() is None

    def test_double_coded_items_give_the_human_ceiling(self, tmp_path: Path) -> None:
        """§12: a code where humans disagree is a codebook problem, not a model
        problem -- and `check` can only say so if this pipeline exists."""
        store = self.store(tmp_path)
        for item, first, second in [("r1", 1, 1), ("r2", 1, 0), ("r3", 0, 0)]:
            self.add(store, item, first, coder="coder_1")
            self.add(store, item, second, coder="coder_2")
        a, b, items = store.double_coded("access_barrier")
        assert items == ("r1", "r2", "r3")
        assert np.allclose(a, [1, 1, 0])
        assert np.allclose(b, [1, 0, 0])

    def test_single_coded_items_are_excluded_from_the_ceiling(self, tmp_path: Path) -> None:
        store = self.store(tmp_path)
        self.add(store, "r1", 1, coder="coder_1")
        self.add(store, "r2", 1, coder="coder_1")
        self.add(store, "r2", 0, coder="coder_2")
        _, _, items = store.double_coded("access_barrier")
        assert items == ("r2",)

    def test_no_double_coding_gives_no_ceiling(self, tmp_path: Path) -> None:
        store = self.store(tmp_path)
        self.add(store, "r1", 1)
        assert self.store(tmp_path).double_coded("access_barrier")[2] == ()

    def test_an_invalid_label_is_refused(self, tmp_path: Path) -> None:
        store = self.store(tmp_path)
        with pytest.raises(ValueError, match="must be 0 or 1"):
            store.record(
                item_id="r1",
                code_id="c",
                label=7,
                coder="c1",
                split="calibration",
                stratum=UNIFORM,
                inclusion_probability=0.1,
                sample_seed=1,
            )

    def test_an_invalid_inclusion_probability_is_refused(self, tmp_path: Path) -> None:
        store = self.store(tmp_path)
        with pytest.raises(ValueError, match="inclusion probability"):
            store.record(
                item_id="r1",
                code_id="c",
                label=1,
                coder="c1",
                split="calibration",
                stratum=UNIFORM,
                inclusion_probability=0.0,
                sample_seed=1,
            )

    def test_a_corrupt_line_names_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "gold.jsonl"
        path.write_text('{"item_id": "r1"}\n', encoding="utf-8")
        with pytest.raises(ProjectError, match="unreadable gold record"):
            GoldStore(path).all_records()

    def test_malformed_json_names_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "gold.jsonl"
        path.write_text("{not json\n", encoding="utf-8")
        with pytest.raises(ProjectError, match=":1 is not valid JSON"):
            GoldStore(path).all_records()

    def test_reading_an_absent_file_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert self.store(tmp_path).all_records() == []

    def test_purge_removes_the_personal_data(self, tmp_path: Path) -> None:
        store = self.store(tmp_path)
        self.add(store, "r0001", 1)
        assert store.purge() is True
        assert store.all_records() == []
        assert store.purge() is False


class TestReviewStore:
    def test_review_records_show_the_probability(self, tmp_path: Path) -> None:
        # §12: showing the probability is fine and useful for review.
        store = ReviewStore(tmp_path / "review.jsonl")
        store.append(
            [store.record(item_id="r1", code_id="c1", label=1, coder="coder_1", probability=0.62)]
        )
        record = store.latest()[("r1", "c1")]
        assert record.model_visible is True
        assert record.probability == 0.62

    def test_a_missing_probability_is_allowed(self, tmp_path: Path) -> None:
        store = ReviewStore(tmp_path / "review.jsonl")
        store.append(
            [store.record(item_id="r1", code_id="c1", label=0, coder="c", probability=None)]
        )
        assert store.latest()[("r1", "c1")].probability is None

    def test_later_decisions_win(self, tmp_path: Path) -> None:
        store = ReviewStore(tmp_path / "review.jsonl")
        store.append(
            [store.record(item_id="r1", code_id="c1", label=1, coder="c", probability=0.5)]
        )
        store.append(
            [store.record(item_id="r1", code_id="c1", label=0, coder="c", probability=0.5)]
        )
        assert store.latest()[("r1", "c1")].label == 0

    def test_purge(self, tmp_path: Path) -> None:
        store = ReviewStore(tmp_path / "review.jsonl")
        store.append(
            [store.record(item_id="r1", code_id="c1", label=1, coder="c", probability=0.5)]
        )
        assert store.purge() is True
        assert store.latest() == {}

    def test_a_corrupt_record_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "review.jsonl"
        path.write_text('{"item_id": "r1"}\n', encoding="utf-8")
        with pytest.raises(ProjectError, match="unreadable review record"):
            ReviewStore(path).latest()


class TestCodingInterface:
    def tasks(self, n: int = 3) -> list[Task]:
        return [
            Task(item_id=f"r{i}", text=f"answer number {i}", code=CODE, probability=0.87)
            for i in range(n)
        ]

    def run(self, answers: list[str], *, show_probability: bool):  # type: ignore[no-untyped-def]
        console = Console(file=io.StringIO(), width=70)
        stream = iter(answers)
        session = code_tasks(
            self.tasks(),
            show_probability=show_probability,
            console=console,
            prompt=lambda _: next(stream, QUIT_SENTINEL),
        )
        return session, console.file.getvalue()

    def test_yes_and_no_are_recorded(self) -> None:
        session, _ = self.run(["y", "n", "y"], show_probability=False)
        assert [(j.task.item_id, j.label) for j in session.judgements] == [
            ("r0", 1),
            ("r1", 0),
            ("r2", 1),
        ]

    def test_gold_mode_never_shows_the_probability(self) -> None:
        """§12: for gold, model output MUST be hidden."""
        _, output = self.run(["y", "y", "y"], show_probability=False)
        assert "0.87" not in output
        assert "model output hidden" in output

    def test_review_mode_shows_the_probability(self) -> None:
        _, output = self.run(["y", "y", "y"], show_probability=True)
        assert "0.87" in output

    def test_skip_leaves_the_item_uncoded(self) -> None:
        session, _ = self.run(["?", "y", "y"], show_probability=False)
        assert len(session.skipped) == 1
        assert session.skipped[0].item_id == "r0"
        assert len(session.judgements) == 2

    def test_back_lets_a_coder_correct_themselves(self) -> None:
        session, _ = self.run(["y", "b", "n", "y", "y"], show_probability=False)
        labels = {j.task.item_id: j.label for j in session.judgements}
        assert labels["r0"] == 0, "going back must replace the earlier judgement"

    def test_back_on_the_first_item_is_harmless(self) -> None:
        session, _ = self.run(["b", "y", "y", "y"], show_probability=False)
        assert len(session.judgements) == 3

    def test_quit_stops_early_and_keeps_progress(self) -> None:
        session, _ = self.run(["y", "q"], show_probability=False)
        assert session.stopped_early is True
        assert len(session.judgements) == 1

    def test_unrecognised_input_is_survivable(self) -> None:
        session, output = self.run(["z", "y", "y", "y"], show_probability=False)
        assert "did not understand" in output
        assert len(session.judgements) == 3

    def test_long_text_is_truncated_for_the_terminal(self) -> None:
        console = Console(file=io.StringIO(), width=70)
        long_task = Task(item_id="r0", text="word " * 2000, code=CODE)
        code_tasks([long_task], show_probability=False, console=console, prompt=lambda _: "y")
        assert "more characters" in console.file.getvalue()

    def test_progress_and_elapsed_time_are_reported(self) -> None:
        # §12: show progress and elapsed time -- coders need to plan the session.
        session, output = self.run(["y", "y", "y"], show_probability=False)
        assert "[1/3]" in output
        assert session.elapsed_seconds >= 0.0
        console = Console(file=io.StringIO(), width=70)
        summarise(session, console)
        assert "coded 3 judgement" in console.file.getvalue()

    def test_summary_mentions_skips_and_early_exit(self) -> None:
        session, _ = self.run(["?", "q"], show_probability=False)
        console = Console(file=io.StringIO(), width=70)
        summarise(session, console)
        text = console.file.getvalue()
        assert "skipped 1" in text
        assert "stopped early" in text

    def test_examples_are_shown_to_the_coder(self) -> None:
        console = Console(file=io.StringIO(), width=120)
        code = Code(
            id="access_barrier",
            definition="d",
            examples_yes=("clinic closes at 5",),
            examples_no=("not necessary",),
        )
        code_tasks(
            [Task(item_id="r0", text="t", code=code)],
            show_probability=False,
            console=console,
            prompt=lambda _: "y",
        )
        output = console.file.getvalue()
        assert "clinic closes at 5" in output
        assert "not necessary" in output


QUIT_SENTINEL = "q"
