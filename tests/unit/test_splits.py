"""Split-integrity tests (plan §8.4, §14.4).

§14.13 blocks merges when these fail. They are the regression guard for the
partition that makes every reported number honest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scruple.corpus.splits import Split, Splits, assign_splits
from scruple.errors import ProjectError, SplitViolation
from scruple.stats.sealed import SealedDataError

IDS = tuple(f"r{i}" for i in range(1000))


def fresh(seed: int = 42, **kwargs: float) -> Splits:
    return assign_splits(IDS, corpus_hash="corpus-hash", seed=seed, **kwargs)  # type: ignore[arg-type]


class TestPartitionProperties:
    def test_partitions_are_disjoint(self) -> None:
        splits = fresh()
        sets = [set(splits.ids(s)) for s in Split]
        assert sets[0] & sets[1] == set()
        assert sets[1] & sets[2] == set()
        assert sets[0] & sets[2] == set()

    def test_partitions_are_exhaustive(self) -> None:
        splits = fresh()
        assert set().union(*(set(splits.ids(s)) for s in Split)) == set(IDS)

    def test_sizes_are_exact_not_approximate(self) -> None:
        # Slicing a hash-ordered list gives exactly 10/45/45, where independent
        # coin flips per item would give whatever the flips produced.
        assert fresh().counts() == {Split.DEV: 100, Split.CALIBRATION: 450, Split.TEST: 450}

    def test_counts_always_sum_to_the_corpus_size(self) -> None:
        odd = assign_splits(
            tuple(f"r{i}" for i in range(997)),
            corpus_hash="h",
            dev=0.1,
            calibration=0.45,
            test=0.45,
        )
        assert sum(odd.counts().values()) == 997

    def test_is_deterministic_given_the_seed(self) -> None:
        assert fresh().assignment == fresh().assignment

    def test_a_different_seed_produces_a_different_partition(self) -> None:
        assert fresh(seed=42).assignment != fresh(seed=7).assignment

    def test_a_changed_corpus_is_refused_rather_than_repartitioned(self) -> None:
        """Slice sizes are exact, so they depend on the corpus size.

        Appending rows therefore *does* move a boundary and can reassign items.
        Rather than pretend otherwise, the frozen split records the corpus hash
        and refuses a corpus it was not built from -- re-partitioning silently
        would invalidate every number already computed.
        """
        splits = fresh()
        assert splits.matches_corpus("corpus-hash") is True
        splits.require_corpus("corpus-hash")

        with pytest.raises(ProjectError, match="corpus has changed") as caught:
            splits.require_corpus("a-different-corpus")
        assert "splits reset" in (caught.value.hint or "")

    def test_growth_can_move_a_slice_boundary(self) -> None:
        # Documenting the consequence of exact sizes, so nobody assumes otherwise.
        before = fresh()
        grown = assign_splits((*IDS, *(f"new_{i}" for i in range(100))), corpus_hash="h2", seed=42)
        assert grown.counts()[Split.DEV] == 110
        assert any(grown.assignment[i] is not before.assignment[i] for i in IDS)

    def test_assignment_is_independent_of_input_order(self) -> None:
        forward = assign_splits(IDS, corpus_hash="h", seed=42)
        backward = assign_splits(tuple(reversed(IDS)), corpus_hash="h", seed=42)
        assert forward.assignment == backward.assignment

    def test_the_seed_is_recorded(self) -> None:
        # §14.4: the seed must be recorded so the partition can be reproduced.
        assert fresh(seed=99).seed == 99


class TestPartitionRejections:
    def test_fractions_must_sum_to_one(self) -> None:
        with pytest.raises(ProjectError, match=r"must sum to 1\.0"):
            fresh(dev=0.2, calibration=0.5, test=0.5)

    def test_fractions_must_not_be_negative(self) -> None:
        with pytest.raises(ProjectError, match="not be negative"):
            fresh(dev=-0.1, calibration=0.6, test=0.5)

    def test_an_empty_corpus_cannot_be_partitioned(self) -> None:
        with pytest.raises(ProjectError, match="empty corpus"):
            assign_splits((), corpus_hash="h")

    def test_duplicate_ids_cannot_be_partitioned(self) -> None:
        with pytest.raises(ProjectError, match="duplicate item ids"):
            assign_splits(("a", "a"), corpus_hash="h")


class TestSplitEnforcement:
    def test_try_may_draw_from_dev(self) -> None:
        splits = fresh()
        splits.require(splits.ids(Split.DEV)[:10], Split.DEV, what="`try`")

    def test_try_refuses_to_sample_outside_dev(self) -> None:
        """§8.4 and §12: `try` shows model output while the researcher edits
        definitions. That is fitting, so it must never touch calibration or test."""
        splits = fresh()
        stray = splits.ids(Split.TEST)[:3]
        with pytest.raises(SplitViolation, match="may only use the dev split") as caught:
            splits.require(stray, Split.DEV, what="`try`")
        assert "invalidates" in (caught.value.hint or "")

    def test_the_violation_names_the_offending_items(self) -> None:
        splits = fresh()
        stray = splits.ids(Split.CALIBRATION)[0]
        with pytest.raises(SplitViolation, match=stray):
            splits.require([stray], Split.DEV, what="`try`")

    def test_unknown_items_are_also_refused(self) -> None:
        with pytest.raises(SplitViolation):
            fresh().require(["not_in_corpus"], Split.DEV, what="`try`")

    def test_of_returns_the_assignment(self) -> None:
        splits = fresh()
        assert splits.of(splits.ids(Split.DEV)[0]) is Split.DEV

    def test_of_rejects_an_unknown_item(self) -> None:
        with pytest.raises(KeyError, match="not in the frozen split"):
            fresh().of("ghost")


class TestSealing:
    def test_the_test_split_comes_back_sealed(self) -> None:
        # §14.4: the threshold fitter receives a sealed object; reading it raises.
        sealed = fresh().sealed(Split.TEST)
        with pytest.raises(SealedDataError):
            len(sealed)
        assert sealed.label == "test"

    def test_unsealing_for_reporting_yields_the_ids(self) -> None:
        splits = fresh()
        assert set(splits.sealed(Split.TEST).unseal(purpose="reporting")) == set(
            splits.ids(Split.TEST)
        )

    def test_unsealing_for_fitting_is_refused(self) -> None:
        with pytest.raises(SealedDataError, match="purpose"):
            fresh().sealed(Split.TEST).unseal(purpose="fitting")


class TestPersistence:
    def test_round_trips_through_disk(self, tmp_path: Path) -> None:
        original = fresh()
        path = tmp_path / ".scruple" / "splits.json"
        original.save(path)
        reloaded = Splits.load(path)
        assert reloaded.assignment == original.assignment
        assert reloaded.seed == original.seed
        assert reloaded.corpus_hash == original.corpus_hash
        assert reloaded.created_at == original.created_at

    def test_reloading_reuses_the_frozen_split_rather_than_repartitioning(
        self, tmp_path: Path
    ) -> None:
        """§14.4. Even if the fractions in scruple.yml change later, what is on
        disk is what governs -- splits are immutable once written (§8.4)."""
        path = tmp_path / "splits.json"
        fresh(dev=0.10, calibration=0.45, test=0.45).save(path)
        reloaded = Splits.load(path)
        assert reloaded.counts()[Split.DEV] == 100
        # A differently configured partition of the same corpus disagrees, which
        # is exactly why the file rather than the config is authoritative.
        assert reloaded.assignment != fresh(dev=0.5, calibration=0.25, test=0.25).assignment

    def test_the_file_records_the_explicit_assignment_for_audit(self, tmp_path: Path) -> None:
        path = tmp_path / "splits.json"
        fresh().save(path)
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data["items"]) == {"dev", "calibration", "test"}
        assert len(data["items"]["dev"]) == 100
        assert data["counts"]["dev"] == 100
        assert data["fractions"]["dev"] == 0.10

    def test_missing_file_points_at_load(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectError, match="no frozen split") as caught:
            Splits.load(tmp_path / "absent.json")
        assert "scruple load" in (caught.value.hint or "")

    def test_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "splits.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ProjectError, match="not valid JSON"):
            Splits.load(path)

    def test_unsupported_version(self, tmp_path: Path) -> None:
        path = tmp_path / "splits.json"
        path.write_text('{"version": 99}', encoding="utf-8")
        with pytest.raises(ProjectError, match=r"unsupported splits\.json version"):
            Splits.load(path)

    def test_malformed_items_section(self, tmp_path: Path) -> None:
        path = tmp_path / "splits.json"
        path.write_text('{"version": 1, "items": "nope"}', encoding="utf-8")
        with pytest.raises(ProjectError, match="malformed `items`"):
            Splits.load(path)

    def test_missing_seed(self, tmp_path: Path) -> None:
        path = tmp_path / "splits.json"
        path.write_text('{"version": 1, "items": {"dev": []}}', encoding="utf-8")
        with pytest.raises(ProjectError, match="integer `seed`"):
            Splits.load(path)

    def test_malformed_per_split_list(self, tmp_path: Path) -> None:
        path = tmp_path / "splits.json"
        path.write_text('{"version": 1, "seed": 42, "items": {"dev": "nope"}}', encoding="utf-8")
        with pytest.raises(ProjectError, match=r"malformed `items\.dev`"):
            Splits.load(path)
