# preparation/test_gesture_type.py
import pandas as pd
import pytest

from .gesture_type import assign_gesture_type, classify_gesture_type


def _tap_group(**kwargs):
    data = {
        'type_metadata': ['tap', 'tap'],
        'contact_location_x': [0.0, 0.0],
        'contact_location_y': [0.0, 0.0],
        'contact_location_z': [0.0, 0.0],
    }
    data.update(kwargs)
    return pd.DataFrame(data)


def _stroke_group(start_x: float, end_x: float, **kwargs):
    data = {
        'type_metadata': ['stroke', 'stroke'],
        'contact_location_x': [start_x, end_x],
        'contact_location_y': [0.0, 0.0],
        'contact_location_z': [0.0, 0.0],
    }
    data.update(kwargs)
    return pd.DataFrame(data)


def _stroke_group_with_nans(x_values: list, **kwargs):
    n = len(x_values)
    data = {
        'block_order_id': [1] * n,
        'trial_id': [1] * n,
        'single_touch_id': [1] * n,
        'type_metadata': ['stroke'] * n,
        'contact_location_x': x_values,
        'contact_location_y': [0.0] * n,
        'contact_location_z': [0.0] * n,
    }
    data.update(kwargs)
    return pd.DataFrame(data)


class TestClassifyGestureType:
    def test_tap_returns_tap(self):
        assert classify_gesture_type(_tap_group()) == 'tap'

    def test_stroke_proximal_when_end_x_greater(self):
        assert classify_gesture_type(_stroke_group(1.0, 2.0)) == 'stroke_proximal'

    def test_stroke_distal_when_end_x_less(self):
        assert classify_gesture_type(_stroke_group(2.0, 1.0)) == 'stroke_distal'

    def test_stroke_distal_when_x_equal(self):
        assert classify_gesture_type(_stroke_group(1.5, 1.5)) == 'stroke_distal'

    def test_raises_on_missing_type_metadata_column(self):
        df = pd.DataFrame({
            'contact_location_x': [1.0, 2.0],
            'contact_location_y': [0.0, 0.0],
            'contact_location_z': [0.0, 0.0],
        })
        with pytest.raises(ValueError, match="type_metadata"):
            classify_gesture_type(df)

    def test_raises_on_unknown_type_metadata_value(self):
        df = pd.DataFrame({
            'type_metadata': ['swipe', 'swipe'],
            'contact_location_x': [1.0, 2.0],
            'contact_location_y': [0.0, 0.0],
            'contact_location_z': [0.0, 0.0],
        })
        with pytest.raises(ValueError, match="unknown type_metadata"):
            classify_gesture_type(df)

    def test_raises_on_stroke_missing_contact_location_columns(self):
        df = pd.DataFrame({'type_metadata': ['stroke', 'stroke']})
        with pytest.raises(ValueError, match="contact_location"):
            classify_gesture_type(df)

    def test_stroke_proximal_nan_at_start(self):
        # NaN at iloc[0]; first valid x=1.0, last valid x=3.0 → proximal
        group = _stroke_group_with_nans([float('nan'), 1.0, 2.0, 3.0])
        assert classify_gesture_type(group) == 'stroke_proximal'

    def test_stroke_distal_nan_at_end(self):
        # NaN at iloc[-1]; first valid x=3.0, last valid x=1.0 → distal
        group = _stroke_group_with_nans([3.0, 2.0, 1.0, float('nan')])
        assert classify_gesture_type(group) == 'stroke_distal'

    def test_stroke_nan_at_both_ends(self):
        # NaN at both boundaries; interior goes 1.0→4.0 → proximal
        group = _stroke_group_with_nans([float('nan'), 1.0, 4.0, float('nan')])
        assert classify_gesture_type(group) == 'stroke_proximal'

    def test_stroke_unknown_when_all_nan_contact_location(self):
        group = _stroke_group_with_nans([float('nan'), float('nan')])
        assert classify_gesture_type(group) == 'stroke_unknown'


class TestAssignGestureType:
    def _make_df(self):
        return pd.DataFrame({
            'block_order_id':    [1, 1, 1, 1, 2, 2],
            'trial_id':          [1, 1, 1, 1, 1, 1],
            'single_touch_id':   [0, 1, 1, 2, 1, 1],
            'type_metadata':     ['tap', 'tap', 'tap', 'stroke', 'stroke', 'stroke'],
            'contact_location_x': [0.0, 0.0, 0.0, 2.0, 1.0, 3.0],
            'contact_location_y': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'contact_location_z': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        })

    def test_adds_gesture_type_column(self):
        df = self._make_df()
        result = assign_gesture_type(df)
        assert 'gesture_type' in result.columns

    def test_correct_values_per_group(self):
        df = self._make_df()
        result = assign_gesture_type(df)

        tap_rows = result[(result['block_order_id'] == 1) & (result['single_touch_id'] == 1)]
        assert (tap_rows['gesture_type'] == 'tap').all()

        # block 1, touch 2: single-row stroke with x=2.0 → only one frame, Δx=0 → distal
        stroke_distal = result[(result['block_order_id'] == 1) & (result['single_touch_id'] == 2)]
        assert (stroke_distal['gesture_type'] == 'stroke_distal').all()

        # block 2, touch 1: x goes 1.0 → 3.0, Δx > 0 → proximal
        stroke_proximal = result[(result['block_order_id'] == 2) & (result['single_touch_id'] == 1)]
        assert (stroke_proximal['gesture_type'] == 'stroke_proximal').all()

    def test_skips_single_touch_id_zero(self):
        df = self._make_df()
        result = assign_gesture_type(df)
        zero_rows = result[result['single_touch_id'] == 0]
        assert zero_rows['gesture_type'].isna().all()

    def test_does_not_mutate_input(self):
        df = self._make_df()
        _ = assign_gesture_type(df)
        assert 'gesture_type' not in df.columns
