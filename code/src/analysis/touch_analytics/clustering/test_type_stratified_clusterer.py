# clustering/test_type_stratified_clusterer.py
import re

import numpy as np
import pandas as pd
import pytest

from .base import ClusteringContext
from .type_stratified_clusterer import TypeStratifiedClusterer


def _feature_df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'pressure': rng.random(n), 'velocity': rng.random(n)})


def _config(base_method='binning', n_bins=3):
    return {
        'base_method': base_method,
        'n_bins': n_bins,
    }


def _context(type_labels, direction_labels=None) -> ClusteringContext:
    return ClusteringContext(
        type_labels=np.asarray(type_labels, dtype=object),
        direction_labels=np.asarray(direction_labels, dtype=object) if direction_labels is not None else None,
    )


class TestSingleTypeTapOnly:
    def test_labels_are_tap_prefixed(self):
        n = 10
        df = _feature_df(n)
        labels, _ = TypeStratifiedClusterer().fit_predict(df, _config(), _context(['tap'] * n))
        assert all(str(lbl).startswith('tap_') for lbl in labels)

    def test_no_stroke_labels(self):
        n = 10
        df = _feature_df(n)
        labels, _ = TypeStratifiedClusterer().fit_predict(df, _config(), _context(['tap'] * n))
        assert not any('stroke' in str(lbl) for lbl in labels)


class TestMultipleTypes:
    def test_three_way_split(self):
        types = ['tap'] * 5 + ['stroke'] * 5 + ['stroke'] * 5
        directions = [None] * 5 + ['proximal'] * 5 + ['distal'] * 5
        df = _feature_df(15)
        labels, meta = TypeStratifiedClusterer().fit_predict(df, _config(), _context(types, directions))
        prefixes = {str(lbl).rsplit('_', 1)[0] for lbl in labels}
        assert 'tap' in prefixes
        assert 'stroke_proximal' in prefixes
        assert 'stroke_distal' in prefixes

    def test_per_type_metadata_keys(self):
        types = ['tap'] * 5 + ['stroke'] * 5 + ['stroke'] * 5
        directions = [None] * 5 + ['proximal'] * 5 + ['distal'] * 5
        df = _feature_df(15)
        _, meta = TypeStratifiedClusterer().fit_predict(df, _config(), _context(types, directions))
        assert 'tap' in meta['per_type']
        assert 'stroke_proximal' in meta['per_type']
        assert 'stroke_distal' in meta['per_type']


class TestDirectionFallback:
    def test_null_direction_becomes_stroke_distal(self):
        types = ['stroke'] * 6
        directions = [None, float('nan'), 'distal', None, None, None]
        df = _feature_df(6)
        labels, _ = TypeStratifiedClusterer().fit_predict(df, _config(), _context(types, directions))
        assert all(str(lbl).startswith('stroke_distal') for lbl in labels)

    def test_missing_direction_labels_key_defaults_to_distal(self):
        types = ['stroke'] * 4
        df = _feature_df(4)
        labels, _ = TypeStratifiedClusterer().fit_predict(df, _config(), _context(types, direction_labels=None))
        assert all(str(lbl).startswith('stroke_distal') for lbl in labels)


class TestLabelFormat:
    def test_all_labels_match_regex(self):
        pattern = re.compile(r'^(tap|stroke_proximal|stroke_distal)_\d{2}$')
        types = ['tap'] * 4 + ['stroke'] * 4 + ['stroke'] * 4
        directions = [None] * 4 + ['proximal'] * 4 + ['distal'] * 4
        df = _feature_df(12)
        labels, _ = TypeStratifiedClusterer().fit_predict(df, _config(), _context(types, directions))
        for lbl in labels:
            assert pattern.match(str(lbl)), f"Label '{lbl}' does not match expected format"


class TestMissingTypeLabelsRaises:
    def test_raises_value_error(self):
        df = _feature_df(5)
        cfg = {'base_method': 'binning', 'n_bins': 3}
        with pytest.raises(ValueError, match='type_labels'):
            TypeStratifiedClusterer().fit_predict(df, cfg, ClusteringContext())


class TestEmptyGroupSkipped:
    def test_no_tap_labels_when_no_taps(self):
        types = ['stroke'] * 8
        directions = ['proximal'] * 4 + ['distal'] * 4
        df = _feature_df(8)
        labels, _ = TypeStratifiedClusterer().fit_predict(df, _config(), _context(types, directions))
        assert not any('tap' in str(lbl) for lbl in labels)

    def test_warning_logged_for_empty_group(self, caplog):
        import logging
        types = ['stroke'] * 6
        directions = ['distal'] * 6
        df = _feature_df(6)
        with caplog.at_level(logging.WARNING):
            TypeStratifiedClusterer().fit_predict(df, _config(), _context(types, directions))
        messages = ' '.join(caplog.messages)
        assert 'tap' in messages


class TestStratifiedBinRangeParsing:
    """Verify split-on-last-_ parsing used by _build_cluster_description."""

    def test_tap_label_splits_correctly(self):
        label = 'tap_03'
        sep = label.rfind('_')
        assert label[:sep] == 'tap'
        assert int(label[sep + 1:]) == 3

    def test_stroke_proximal_label_splits_correctly(self):
        label = 'stroke_proximal_01'
        sep = label.rfind('_')
        assert label[:sep] == 'stroke_proximal'
        assert int(label[sep + 1:]) == 1

    def test_stroke_distal_label_splits_correctly(self):
        label = 'stroke_distal_05'
        sep = label.rfind('_')
        assert label[:sep] == 'stroke_distal'
        assert int(label[sep + 1:]) == 5

    def test_bin_range_resolved_from_nested_metadata(self):
        metadata = {
            'algorithm': 'type_stratified',
            'per_type': {
                'tap': {
                    'primary_feature': 'pressure',
                    'bin_edges': {'pressure': [0.0, 0.25, 0.50, 0.75, 1.0]},
                    'n_clusters': 4,
                },
            },
        }
        label = 'tap_03'
        sep = label.rfind('_')
        type_key = label[:sep]
        idx = int(label[sep + 1:])
        per_type_meta = metadata['per_type'][type_key]
        pf = per_type_meta['primary_feature']
        edges = per_type_meta['bin_edges'][pf]
        assert 0 <= idx < len(edges) - 1
        bin_range = {'feature': pf, 'low': round(edges[idx], 2), 'high': round(edges[idx + 1], 2)}
        assert bin_range == {'feature': 'pressure', 'low': 0.75, 'high': 1.0}


class TestRowOrderPreserved:
    def test_label_aligns_with_row(self):
        types = ['tap', 'stroke', 'stroke', 'tap', 'stroke']
        directions = [None, 'proximal', 'distal', None, 'proximal']
        df = _feature_df(5)
        labels, _ = TypeStratifiedClusterer().fit_predict(df, _config(), _context(types, directions))
        assert str(labels[0]).startswith('tap_')
        assert str(labels[1]).startswith('stroke_proximal_')
        assert str(labels[2]).startswith('stroke_distal_')
        assert str(labels[3]).startswith('tap_')
        assert str(labels[4]).startswith('stroke_proximal_')

    def test_output_length_matches_input(self):
        n = 20
        types = ['tap'] * 10 + ['stroke'] * 10
        directions = [None] * 10 + ['distal'] * 10
        df = _feature_df(n)
        labels, _ = TypeStratifiedClusterer().fit_predict(df, _config(), _context(types, directions))
        assert len(labels) == n
