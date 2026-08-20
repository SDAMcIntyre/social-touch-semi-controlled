"""Unit tests for ``postprocessing.forearm_dedup_metadata``.

The sidecar is the only place the *effective* dedup epsilon and the resulting
vertex count survive, and it is what turns "this ``vertex_id`` indexes some
forearm" into "this ``vertex_id`` indexes *that* forearm".  So the reader is
tested for what it refuses, not only for what it accepts: a sidecar that is
missing, stale, or internally inconsistent must fail loudly at the stage that
would otherwise stamp an unverifiable index onto a data file.

No Open3D here — the module is a JSON format and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from postprocessing.forearm_dedup_metadata import (
    EPSILON_SOURCE_DAG_CONFIG,
    EPSILON_SOURCE_INTERACTIVE_MONITOR,
    FOREARM_DEDUP_METADATA_SCHEMA_VERSION,
    ForearmDedupMetadata,
    forearm_dedup_metadata_path,
    read_forearm_dedup_metadata,
    write_forearm_dedup_metadata,
)

STATS = {"n_original": 3718, "n_deduped": 1807, "n_removed": 1911}


def write(tmp_path: Path, **overrides) -> Path:
    """Write a valid sidecar beside a notional PLY and return the PLY path."""
    ply = tmp_path / "session_forearm.ply"
    kwargs = {
        "source_ply": tmp_path / "source_forearm.ply",
        "epsilon": 0.5,
        "epsilon_source": EPSILON_SOURCE_DAG_CONFIG,
        "stats": STATS,
    }
    kwargs.update(overrides)
    write_forearm_dedup_metadata(ply, **kwargs)
    return ply


def corrupt(ply: Path, **changes) -> None:
    """Rewrite the sidecar with *changes* applied; a value of ``None`` deletes."""
    path = forearm_dedup_metadata_path(ply)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class TestPathRule:
    def test_the_sidecar_sits_beside_the_ply_with_its_stem(self, tmp_path):
        ply = tmp_path / "sub" / "arm.ply"
        assert forearm_dedup_metadata_path(ply) == tmp_path / "sub" / "arm_dedup_metadata.json"

    def test_the_rule_is_stable(self, tmp_path):
        ply = tmp_path / "arm.ply"
        assert forearm_dedup_metadata_path(ply) == forearm_dedup_metadata_path(ply)


class TestRoundTrip:
    def test_a_written_sidecar_reads_back_field_for_field(self, tmp_path):
        ply = write(tmp_path)
        meta = read_forearm_dedup_metadata(ply)

        assert isinstance(meta, ForearmDedupMetadata)
        assert meta.path == forearm_dedup_metadata_path(ply)
        assert meta.schema_version == FOREARM_DEDUP_METADATA_SCHEMA_VERSION
        assert meta.source_ply == "source_forearm.ply"
        assert meta.deduplicated_ply == "session_forearm.ply"
        assert meta.dedup_epsilon == 0.5
        assert meta.epsilon_source == EPSILON_SOURCE_DAG_CONFIG
        assert meta.n_vertices_original == 3718
        assert meta.n_vertices_deduped == 1807
        assert meta.n_vertices_removed == 1911

    def test_an_interactively_chosen_epsilon_is_recorded_as_such(self, tmp_path):
        ply = write(
            tmp_path, epsilon=0.37, epsilon_source=EPSILON_SOURCE_INTERACTIVE_MONITOR
        )
        meta = read_forearm_dedup_metadata(ply)
        assert meta.dedup_epsilon == pytest.approx(0.37)
        assert meta.epsilon_source == EPSILON_SOURCE_INTERACTIVE_MONITOR

    def test_the_epsilon_survives_as_a_float_not_a_string(self, tmp_path):
        ply = write(tmp_path, epsilon=0.123456789)
        assert read_forearm_dedup_metadata(ply).dedup_epsilon == 0.123456789


class TestReaderRefusals:
    def test_a_missing_sidecar_raises_naming_the_file(self, tmp_path):
        ply = tmp_path / "never_deduped.ply"
        with pytest.raises(FileNotFoundError) as excinfo:
            read_forearm_dedup_metadata(ply)
        assert "never_deduped_dedup_metadata.json" in str(excinfo.value)
        assert "deduplicate_xy" in str(excinfo.value)

    def test_malformed_json_raises(self, tmp_path):
        ply = write(tmp_path)
        forearm_dedup_metadata_path(ply).write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            read_forearm_dedup_metadata(ply)

    def test_a_json_array_is_not_a_sidecar(self, tmp_path):
        ply = write(tmp_path)
        forearm_dedup_metadata_path(ply).write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ValueError, match="not a JSON object"):
            read_forearm_dedup_metadata(ply)

    def test_an_unknown_schema_version_raises(self, tmp_path):
        ply = write(tmp_path)
        corrupt(ply, schema_version="99")
        with pytest.raises(ValueError, match="schema_version"):
            read_forearm_dedup_metadata(ply)

    @pytest.mark.parametrize(
        "key",
        [
            "source_ply",
            "deduplicated_ply",
            "dedup_epsilon",
            "epsilon_source",
            "n_vertices_original",
            "n_vertices_deduped",
            "n_vertices_removed",
        ],
    )
    def test_a_missing_key_raises_naming_it(self, tmp_path, key):
        ply = write(tmp_path)
        corrupt(ply, **{key: None})
        with pytest.raises(ValueError, match=key):
            read_forearm_dedup_metadata(ply)

    @pytest.mark.parametrize("bad", ["0.5", None, 0.0, -1.0, True])
    def test_a_non_positive_or_non_numeric_epsilon_raises(self, tmp_path, bad):
        ply = write(tmp_path)
        path = forearm_dedup_metadata_path(ply)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["dedup_epsilon"] = bad
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="dedup_epsilon"):
            read_forearm_dedup_metadata(ply)

    def test_an_unknown_epsilon_source_raises(self, tmp_path):
        ply = write(tmp_path)
        corrupt(ply, epsilon_source="whatever")
        with pytest.raises(ValueError, match="epsilon_source"):
            read_forearm_dedup_metadata(ply)

    def test_a_float_vertex_count_raises(self, tmp_path):
        ply = write(tmp_path)
        corrupt(ply, n_vertices_deduped=1807.0)
        with pytest.raises(ValueError, match="n_vertices_deduped"):
            read_forearm_dedup_metadata(ply)

    def test_counts_that_do_not_add_up_raise(self, tmp_path):
        ply = write(tmp_path)
        corrupt(ply, n_vertices_removed=1910)
        with pytest.raises(ValueError, match="inconsistent vertex counts"):
            read_forearm_dedup_metadata(ply)


class TestWriterRefusals:
    """The writer's guards, kept alongside the reader's so the pair stays symmetric."""

    def test_a_non_positive_epsilon_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="positive finite"):
            write(tmp_path, epsilon=0.0)

    def test_an_unknown_epsilon_source_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="epsilon_source"):
            write(tmp_path, epsilon_source="guessed")

    def test_missing_stats_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="n_removed"):
            write(tmp_path, stats={"n_original": 10, "n_deduped": 4})

    def test_counts_that_do_not_add_up_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="Inconsistent dedup vertex counts"):
            write(tmp_path, stats={"n_original": 10, "n_deduped": 4, "n_removed": 5})

    def test_the_write_is_deterministic(self, tmp_path):
        ply = write(tmp_path)
        first = forearm_dedup_metadata_path(ply).read_bytes()
        write(tmp_path)
        assert forearm_dedup_metadata_path(ply).read_bytes() == first
