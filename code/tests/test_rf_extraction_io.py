"""Tests for rf_extraction_io: round-trips, error handling, sentinel logic."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.receptive_field_mapping.data.rf_extraction_io import (
    load_cluster_session_data,
    load_extraction_summary,
    load_forearm_vertices_artifact,
    load_neuron_cluster_touches,
    load_neuron_contacts,
    load_neuron_touches,
    load_visualization_summary,
    save_cluster_session_data,
    save_extraction_summary,
    save_forearm_vertices,
    save_neuron_cluster_touches,
    save_neuron_contacts,
    save_neuron_touches,
    save_visualization_summary,
    visualization_is_up_to_date,
)


def _xyz(n: int = 10) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((n, 3)).astype(np.float64)


def _spike_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        'x': np.arange(n, dtype=float),
        'y': np.zeros(n),
        'z': np.zeros(n),
        'spike_count': np.arange(1, n + 1),
        'unique_touch_spike_count': np.ones(n, dtype=int),
    })


# ---------------------------------------------------------------------------
# Neuron contacts
# ---------------------------------------------------------------------------

def test_neuron_contacts_roundtrip(tmp_path):
    xyz = _xyz(20)
    save_neuron_contacts(tmp_path, 'S01', xyz)
    loaded = load_neuron_contacts(tmp_path, 'S01')
    np.testing.assert_array_almost_equal(xyz, loaded)


def test_neuron_contacts_missing_raises(tmp_path):
    with pytest.raises(ValueError, match="artifact missing"):
        load_neuron_contacts(tmp_path, 'S99')


def test_neuron_contacts_corrupt_raises(tmp_path):
    d = tmp_path / 'sessions' / 'S01'
    d.mkdir(parents=True)
    (d / 'neuron_contacts_xyz.npy').write_bytes(b'not a numpy file')
    with pytest.raises(ValueError, match="corrupt"):
        load_neuron_contacts(tmp_path, 'S01')


def test_neuron_contacts_wrong_shape_raises(tmp_path):
    d = tmp_path / 'sessions' / 'S01'
    d.mkdir(parents=True)
    np.save(d / 'neuron_contacts_xyz.npy', np.ones((10, 2)))
    with pytest.raises(ValueError, match="expected.*3"):
        load_neuron_contacts(tmp_path, 'S01')


# ---------------------------------------------------------------------------
# Forearm vertices (artifact path)
# ---------------------------------------------------------------------------

def test_forearm_vertices_artifact_roundtrip(tmp_path):
    verts = _xyz(100)
    save_forearm_vertices(tmp_path, 'S01', verts)
    loaded = load_forearm_vertices_artifact(tmp_path, 'S01')
    np.testing.assert_array_almost_equal(verts, loaded)


def test_forearm_vertices_artifact_missing_raises(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        load_forearm_vertices_artifact(tmp_path, 'S99')


def test_forearm_vertices_artifact_wrong_shape_raises(tmp_path):
    d = tmp_path / 'sessions' / 'S01'
    d.mkdir(parents=True)
    np.save(d / 'forearm_vertices.npy', np.ones((5, 4)))
    with pytest.raises(ValueError, match="expected.*3"):
        load_forearm_vertices_artifact(tmp_path, 'S01')


# ---------------------------------------------------------------------------
# Cluster session data
# ---------------------------------------------------------------------------

def test_cluster_session_data_roundtrip(tmp_path):
    cluster_dir = tmp_path / 'cluster_00'
    spike_df = _spike_df()
    xyz = _xyz(15)
    save_cluster_session_data(cluster_dir, 'S01', spike_df, xyz)
    loaded_df, loaded_xyz = load_cluster_session_data(cluster_dir, 'S01')
    pd.testing.assert_frame_equal(spike_df.reset_index(drop=True), loaded_df.reset_index(drop=True))
    np.testing.assert_array_almost_equal(xyz, loaded_xyz)


def test_cluster_session_data_missing_spike_csv_raises(tmp_path):
    cluster_dir = tmp_path / 'cluster_00'
    with pytest.raises(ValueError, match="missing spike CSV"):
        load_cluster_session_data(cluster_dir, 'S01')


def test_cluster_session_data_missing_contacts_raises(tmp_path):
    cluster_dir = tmp_path / 'cluster_00'
    d = cluster_dir / 'sessions' / 'S01'
    d.mkdir(parents=True)
    _spike_df().to_csv(d / 'session_spike_counts.csv', index=False)
    with pytest.raises(ValueError, match="missing contacts"):
        load_cluster_session_data(cluster_dir, 'S01')


def test_cluster_session_data_corrupt_contacts_raises(tmp_path):
    cluster_dir = tmp_path / 'cluster_00'
    d = cluster_dir / 'sessions' / 'S01'
    d.mkdir(parents=True)
    _spike_df().to_csv(d / 'session_spike_counts.csv', index=False)
    (d / 'cluster_contacts_xyz.npy').write_bytes(b'garbage')
    with pytest.raises(ValueError, match="corrupt"):
        load_cluster_session_data(cluster_dir, 'S01')


def test_cluster_session_data_wrong_shape_raises(tmp_path):
    cluster_dir = tmp_path / 'cluster_00'
    d = cluster_dir / 'sessions' / 'S01'
    d.mkdir(parents=True)
    _spike_df().to_csv(d / 'session_spike_counts.csv', index=False)
    np.save(d / 'cluster_contacts_xyz.npy', np.ones((5, 2)))
    with pytest.raises(ValueError, match="expected.*3"):
        load_cluster_session_data(cluster_dir, 'S01')


# ---------------------------------------------------------------------------
# Neuron touches
# ---------------------------------------------------------------------------

def test_neuron_touches_roundtrip(tmp_path):
    touches = {'S01': 10, 'S02': 20}
    save_neuron_touches(tmp_path, touches)
    assert load_neuron_touches(tmp_path) == touches


def test_neuron_touches_missing_raises(tmp_path):
    with pytest.raises(ValueError, match="artifact missing"):
        load_neuron_touches(tmp_path)


# ---------------------------------------------------------------------------
# Neuron cluster touches
# ---------------------------------------------------------------------------

def test_neuron_cluster_touches_roundtrip(tmp_path):
    cluster_dir = tmp_path / 'cluster_00'
    touches = {'S01': 5, 'S02': 3}
    save_neuron_cluster_touches(cluster_dir, touches)
    assert load_neuron_cluster_touches(cluster_dir) == touches


def test_neuron_cluster_touches_missing_raises(tmp_path):
    cluster_dir = tmp_path / 'cluster_00'
    with pytest.raises(ValueError, match="artifact missing"):
        load_neuron_cluster_touches(cluster_dir)


# ---------------------------------------------------------------------------
# Extraction summary
# ---------------------------------------------------------------------------

def test_extraction_summary_roundtrip(tmp_path):
    data = {'cluster_0': {'n_sessions': 2, 'n_touches': 100}}
    save_extraction_summary(tmp_path, 'combo', 'clusterer', data)
    loaded = load_extraction_summary(tmp_path)
    assert loaded['feature_combination'] == 'combo'
    assert loaded['clusterer'] == 'clusterer'
    assert loaded['clusters'] == data


def test_extraction_summary_missing_raises(tmp_path):
    with pytest.raises(ValueError, match="sentinel missing"):
        load_extraction_summary(tmp_path)


# ---------------------------------------------------------------------------
# Visualization summary and staleness check
# ---------------------------------------------------------------------------

def test_visualization_summary_roundtrip(tmp_path):
    save_visualization_summary(tmp_path, 'cylindrical_unwrap', 8.0, 1234567.0)
    loaded = load_visualization_summary(tmp_path)
    assert loaded['projection_method'] == 'cylindrical_unwrap'
    assert loaded['disjoint_mask_distance_mm'] == 8.0
    assert loaded['extraction_summary_mtime'] == 1234567.0


def test_visualization_summary_missing_raises(tmp_path):
    with pytest.raises(ValueError, match="sentinel missing"):
        load_visualization_summary(tmp_path)


def test_visualization_is_up_to_date_same_params(tmp_path):
    save_extraction_summary(tmp_path, 'combo', 'clusterer', {})
    mtime = (tmp_path / 'extraction_summary.json').stat().st_mtime
    save_visualization_summary(tmp_path, 'cylindrical_unwrap', 8.0, mtime)
    assert visualization_is_up_to_date(tmp_path, 'cylindrical_unwrap', 8.0)


def test_visualization_is_up_to_date_changed_method(tmp_path):
    save_extraction_summary(tmp_path, 'combo', 'clusterer', {})
    mtime = (tmp_path / 'extraction_summary.json').stat().st_mtime
    save_visualization_summary(tmp_path, 'tangent_plane', 8.0, mtime)
    assert not visualization_is_up_to_date(tmp_path, 'cylindrical_unwrap', 8.0)


def test_visualization_is_up_to_date_changed_distance(tmp_path):
    save_extraction_summary(tmp_path, 'combo', 'clusterer', {})
    mtime = (tmp_path / 'extraction_summary.json').stat().st_mtime
    save_visualization_summary(tmp_path, 'cylindrical_unwrap', 8.0, mtime)
    assert not visualization_is_up_to_date(tmp_path, 'cylindrical_unwrap', 12.0)


def test_visualization_is_up_to_date_force(tmp_path):
    save_extraction_summary(tmp_path, 'combo', 'clusterer', {})
    mtime = (tmp_path / 'extraction_summary.json').stat().st_mtime
    save_visualization_summary(tmp_path, 'cylindrical_unwrap', 8.0, mtime)
    assert not visualization_is_up_to_date(tmp_path, 'cylindrical_unwrap', 8.0, force=True)


def test_visualization_is_up_to_date_missing_extraction_sentinel(tmp_path):
    assert not visualization_is_up_to_date(tmp_path, 'cylindrical_unwrap', 8.0)


def test_visualization_is_up_to_date_missing_vis_sentinel(tmp_path):
    save_extraction_summary(tmp_path, 'combo', 'clusterer', {})
    assert not visualization_is_up_to_date(tmp_path, 'cylindrical_unwrap', 8.0)
