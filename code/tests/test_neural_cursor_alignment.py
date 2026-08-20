"""Regression tests for the Neural+Kinect viewer's frame -> merged-row mapping.

The bug these guard against
---------------------------
The viewer used to place its timeseries cursor with a single multiplication:

    scale      = len(merged_df) / mkv_frame_count
    sample_idx = int(frame_idx * scale)

Once the visualisation pipeline was repointed at ``blocks_filtered/`` the merged
CSV no longer spanned the whole recording — neural-quality filtering removes
trials — and the cursor drifted linearly away from the displayed frame.
Measured on ``2022-06-14_ST13-02/block-order-01``: 1,734 of 3,186 frames
survived, so the scale was 18.14 against a true anchor spacing of 33.33; frame
437 (``contact_detected=0``) landed on row 7,927, which belongs to frame 238
(``contact_detected=1``, ``contact_depth=4.736``) instead of its own row 14,566.

Why the fixtures here are deliberately NON-UNIFORM
--------------------------------------------------
A fixture whose anchors are evenly spaced across the whole recording passes
under **both** the broken multiplication and the correct lookup, and therefore
proves nothing.  Every fixture below is built so the two implementations
disagree:

* ``truncated_and_gapped`` — truncated at the end *and* punched with a
  mid-recording gap, which is the shape ``discard_from_first_not2use: false``
  produces;
* the anchor spacing itself alternates 33/34 rows, as it does on real blocks;
* one fixture starts at a non-zero frame, which is what truncation from the
  first unusable trial leaves behind.

Each test that could be satisfied by a scale factor also asserts explicitly that
the scale factor would have given a *different* answer.

No Qt, no VTK: the mapping lives in the ``merging.frame_navigation`` leaf
precisely so it can be tested without a display.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from merging.frame_navigation import (  # noqa: E402
    FrameNavigation,
    build_frame_navigation_from_merged_df,
    build_frame_navigation_over_range,
)


# ---------------------------------------------------------------------------
# Synthetic merged-CSV builder
# ---------------------------------------------------------------------------


def _build_merged_df(
    frames: Sequence[int],
    spacings: Sequence[int] | None = None,
) -> Tuple[pd.DataFrame, List[int]]:
    """Return a merged-CSV-shaped DataFrame and the expected row of each frame.

    One anchor row per entry of *frames*, each followed by ``spacing - 1``
    interpolated nerve-rate rows that are NaN in every kinect column — the real
    artifact's shape.  *spacings* alternates 33/34 by default, matching what the
    merged CSVs on disk actually contain, so no constant multiplier can hit
    every anchor even before filtering enters the picture.
    """
    frames = list(frames)
    if spacings is None:
        spacings = [33 if i % 2 == 0 else 34 for i in range(len(frames))]
    assert len(spacings) == len(frames)

    rows: List[dict] = []
    expected_rows: List[int] = []
    for frame, spacing in zip(frames, spacings):
        expected_rows.append(len(rows))
        rows.append(
            {
                "frame_index": float(frame),
                "time_kinect": float(frame) / 30.0,
                "contact_detected": float(frame % 2),
                "contact_depth": float(frame) * 0.01,
                "contact_points": f"{frame}.0,0.0,0.0",
                "Nerve_freq": float(frame),
            }
        )
        for _ in range(spacing - 1):
            rows.append(
                {
                    "frame_index": np.nan,
                    "time_kinect": np.nan,
                    "contact_detected": np.nan,
                    "contact_depth": np.nan,
                    "contact_points": np.nan,
                    "Nerve_freq": float(frame),
                }
            )
    return pd.DataFrame(rows), expected_rows


def _truncated_and_gapped() -> Tuple[pd.DataFrame, List[int], List[int]]:
    """The adversarial fixture: a mid-recording gap AND end truncation.

    Recording is 400 frames.  The CSV holds 0..99 and 200..249 — 150 frames,
    one gap of 100 frames, and 150 frames of truncated tail.  Nothing about this
    is expressible as ``frame * constant``.
    """
    frames = list(range(0, 100)) + list(range(200, 250))
    df, rows = _build_merged_df(frames)
    return df, frames, rows


# ---------------------------------------------------------------------------
# 1. The lookup is exact on a non-uniform mapping
# ---------------------------------------------------------------------------


def test_every_frame_resolves_to_its_exact_row_with_a_gap_and_truncation():
    df, frames, expected_rows = _truncated_and_gapped()
    nav = build_frame_navigation_from_merged_df(df, "synthetic")

    assert nav.size == len(frames)
    for frame, expected in zip(frames, expected_rows):
        assert nav.row_of(frame) == expected, f"frame {frame}"

    # The anchor row is genuinely this frame's row: the CSV agrees.
    for frame in frames:
        row = nav.row_of(frame)
        assert df["frame_index"].iloc[row] == frame
        assert not np.isnan(df["time_kinect"].iloc[row])


def test_the_broken_scale_factor_would_have_disagreed():
    """Proof that the fixture is adversarial, not merely decorative.

    Without this the suite could pass with the multiplication restored.
    """
    df, frames, expected_rows = _truncated_and_gapped()
    nav = build_frame_navigation_from_merged_df(df, "synthetic")

    mkv_frame_count = 400  # the recording, not the CSV
    scale = len(df) / mkv_frame_count

    disagreements = sum(
        1
        for frame, expected in zip(frames, expected_rows)
        if int(frame * scale) != expected
    )
    assert disagreements > 0.9 * len(frames), (
        "the fixture is not adversarial: the broken scale factor agrees with "
        f"the lookup on {len(frames) - disagreements}/{len(frames)} frames"
    )

    # And the row the multiplication picks for frame 200 belongs to a
    # *different* frame — the exact pathology observed on ST13-02.
    broken_row = int(200 * scale)
    owner_pos = int(np.searchsorted(nav.rows, broken_row, side="right")) - 1
    assert owner_pos >= 0
    assert nav.frame_at(owner_pos) != 200
    assert nav.row_of(200) != broken_row


def test_positions_and_frames_are_not_the_same_thing():
    """Truncation from the first bad trial leaves a non-zero first frame."""
    frames = list(range(600, 900))
    df, expected_rows = _build_merged_df(frames)
    nav = build_frame_navigation_from_merged_df(df, "synthetic")

    assert nav.frame_at(0) == 600
    assert nav.position_of(600) == 0
    assert nav.frame_at(nav.size - 1) == 899
    assert nav.row_of(600) == 0
    assert nav.row_of(899) == expected_rows[-1]


def test_row_lookup_is_positional_not_label_based():
    """A non-RangeIndex must not change a single answer.

    ``_setup_axes`` plots against ``np.arange(len(merged_df))``, so the cursor
    lives in *positional* space.  Resolving by index label would silently offset
    the whole cursor track on any DataFrame that has been sliced or reindexed.
    """
    df, frames, expected_rows = _truncated_and_gapped()
    shifted = df.copy()
    shifted.index = pd.RangeIndex(start=100_000, stop=100_000 + len(df))

    nav = build_frame_navigation_from_merged_df(shifted, "synthetic")
    for frame, expected in zip(frames, expected_rows):
        assert nav.row_of(frame) == expected

    scrambled = df.copy()
    scrambled.index = pd.Index([f"r{i}" for i in range(len(df))])
    nav2 = build_frame_navigation_from_merged_df(scrambled, "synthetic")
    for frame, expected in zip(frames, expected_rows):
        assert nav2.row_of(frame) == expected


def test_a_csv_whose_anchors_run_backwards_raises():
    """The merged CSV is chronological by construction.

    A later frame anchored on an earlier row means the artifact is not what the
    viewer thinks it is; the cursor track (and the click-to-frame search, which
    binary-searches the rows) would be meaningless.  Raise rather than sort it
    into something plausible.
    """
    df, _ = _build_merged_df([0, 1, 2, 3])
    backwards = df.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match=r"strictly increasing"):
        build_frame_navigation_from_merged_df(backwards, "backwards.csv")


# ---------------------------------------------------------------------------
# 2. Frames the CSV does not contain (7.2)
# ---------------------------------------------------------------------------


def test_a_frame_inside_the_gap_is_not_navigable():
    df, frames, _ = _truncated_and_gapped()
    nav = build_frame_navigation_from_merged_df(df, "synthetic")

    for absent in (100, 150, 199, 250, 399):
        assert not nav.contains(absent)
        with pytest.raises(KeyError, match=r"not in the navigable set"):
            nav.position_of(absent)
        with pytest.raises(KeyError, match=r"no anchor row"):
            nav.row_of(absent)


def test_an_absent_frame_is_never_mapped_to_a_neighbour():
    """The failure mode being guarded: silently showing the next frame's data."""
    df, _, _ = _truncated_and_gapped()
    nav = build_frame_navigation_from_merged_df(df, "synthetic")

    # 99 and 200 both exist and bracket the gap; 150 must not resolve to either.
    assert nav.contains(99) and nav.contains(200)
    with pytest.raises(KeyError):
        nav.row_of(150)


def test_the_navigable_set_is_the_csv_not_the_recording():
    df, frames, _ = _truncated_and_gapped()
    nav = build_frame_navigation_from_merged_df(df, "synthetic")
    assert nav.frames.tolist() == frames
    assert nav.size == 150            # not 400, the recording length
    assert int(nav.frames[-1]) == 249  # a max frame alone would mis-describe it


def test_stepping_walks_the_set_and_jumps_the_gap():
    """next/prev and play move over positions, so a gap is one step wide."""
    df, frames, _ = _truncated_and_gapped()
    nav = build_frame_navigation_from_merged_df(df, "synthetic")

    pos_of_99 = nav.position_of(99)
    assert nav.frame_at(pos_of_99 + 1) == 200
    # Wrap-around at the end returns to the first navigable frame, not frame 0
    # of the recording (which here happens to coincide) — checked on the
    # truncated-start fixture instead.
    frames2 = list(range(600, 610))
    df2, _ = _build_merged_df(frames2)
    nav2 = build_frame_navigation_from_merged_df(df2, "synthetic")
    assert nav2.frame_at((nav2.size - 1 + 1) % nav2.size) == 600


# ---------------------------------------------------------------------------
# 3. Pure-3D mode (no merged CSV) falls back to the full MKV range
# ---------------------------------------------------------------------------


def test_no_merged_csv_navigates_the_whole_recording():
    nav = build_frame_navigation_over_range(3186)
    assert nav.size == 3186
    assert nav.frame_at(0) == 0
    assert nav.frame_at(3185) == 3185
    assert nav.contains(437)
    assert not nav.has_rows


def test_asking_for_a_row_without_a_csv_raises():
    nav = build_frame_navigation_over_range(10)
    with pytest.raises(ValueError, match=r"no merged CSV"):
        nav.row_of(3)


# ---------------------------------------------------------------------------
# 4. Malformed input fails loudly (no silent fallbacks)
# ---------------------------------------------------------------------------


def test_missing_frame_index_column_raises_and_names_the_columns():
    df = pd.DataFrame({"time_kinect": [0.0, 0.1], "Nerve_freq": [1.0, 2.0]})
    with pytest.raises(ValueError) as exc:
        build_frame_navigation_from_merged_df(df, "somefile.csv")
    assert "frame_index" in str(exc.value)
    assert "time_kinect" in str(exc.value)
    assert "somefile.csv" in str(exc.value)


def test_a_csv_with_no_anchor_at_all_raises():
    df = pd.DataFrame({"frame_index": [np.nan] * 5, "Nerve_freq": [1.0] * 5})
    with pytest.raises(ValueError, match=r"no non-NaN 'frame_index'"):
        build_frame_navigation_from_merged_df(df, "empty.csv")


def test_non_integral_frame_index_raises():
    df = pd.DataFrame({"frame_index": [0.0, 1.5, np.nan]})
    with pytest.raises(ValueError, match=r"non-integral"):
        build_frame_navigation_from_merged_df(df, "bad.csv")


def test_a_frame_anchored_twice_raises():
    df = pd.DataFrame({"frame_index": [7.0, np.nan, 7.0]})
    with pytest.raises(ValueError, match=r"more than one row"):
        build_frame_navigation_from_merged_df(df, "dupe.csv")


def test_negative_frame_index_raises():
    df = pd.DataFrame({"frame_index": [-1.0, 0.0]})
    with pytest.raises(ValueError, match=r"non-negative"):
        build_frame_navigation_from_merged_df(df, "neg.csv")


def test_frame_navigation_refuses_unsorted_frames():
    with pytest.raises(ValueError, match=r"sorted and strictly increasing"):
        FrameNavigation(np.array([3, 1, 2], dtype=np.int64), np.array([0, 1, 2]))


def test_frame_navigation_refuses_misaligned_rows():
    with pytest.raises(ValueError, match=r"one merged-CSV row per frame"):
        FrameNavigation(np.array([0, 1, 2], dtype=np.int64), np.array([0, 1]))


def test_frame_navigation_refuses_non_increasing_rows():
    with pytest.raises(ValueError, match=r"strictly increasing"):
        FrameNavigation(np.array([0, 1, 2], dtype=np.int64), np.array([0, 5, 3]))


def test_position_out_of_range_raises():
    nav = build_frame_navigation_over_range(5)
    with pytest.raises(IndexError, match=r"outside 0\.\.4"):
        nav.frame_at(5)
    with pytest.raises(IndexError):
        nav.frame_at(-1)


def test_frame_at_on_an_empty_set_raises():
    nav = build_frame_navigation_over_range(0)
    with pytest.raises(IndexError, match=r"nothing to display"):
        nav.frame_at(0)


# ---------------------------------------------------------------------------
# 5. The frame label keeps the real kinect frame visible
# ---------------------------------------------------------------------------


def test_label_shows_the_real_frame_not_only_the_position():
    df, _, _ = _truncated_and_gapped()
    nav = build_frame_navigation_from_merged_df(df, "synthetic")

    pos = nav.position_of(200)
    label = nav.format_label(pos)
    assert "200" in label, label            # the actual kinect frame
    assert f"{pos + 1}/150" in label, label  # position within the navigable set
    # It must not read as if the recording only had 150 frames at 0..149.
    assert label != "200 (200/150)"


def test_label_of_an_empty_set_is_explicit():
    assert build_frame_navigation_over_range(0).format_label(0) == "— (0/0)"


# ---------------------------------------------------------------------------
# 6. The contact-points fallback keys by frame, not position (7.3)
# ---------------------------------------------------------------------------


def test_contact_points_keyed_by_frame_survive_a_gap():
    """Reproduces the latent defect in ``_contact_pts_by_frame``.

    The viewer built that list positionally over the anchor rows and then
    indexed it with a kinect frame index.  Past the gap the two diverge, so the
    frame on screen and the contact points drawn for it were different frames.
    """
    # 0..99 then 110..149: 140 anchors, so frame 110 sits at position 100 and
    # the old positional read (index 110) would have returned frame 120's cell.
    frames = list(range(0, 100)) + list(range(110, 150))
    df, _ = _build_merged_df(frames)
    nav = build_frame_navigation_from_merged_df(df, "synthetic")

    cells = list(df["contact_points"].to_numpy()[nav.rows])
    by_frame = {int(f): c for f, c in zip(nav.frames, cells)}

    for frame in frames:
        assert by_frame[frame] == f"{frame}.0,0.0,0.0"

    assert by_frame[110] == "110.0,0.0,0.0"
    assert cells[110] == "120.0,0.0,0.0"      # what the positional read gave
    assert cells[110] != by_frame[110]
