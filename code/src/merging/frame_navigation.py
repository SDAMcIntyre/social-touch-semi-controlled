"""Frame navigation for the merged Neural+Kinect scene: which kinect frames
exist, and where each one lives in the merged CSV.

Why this is a module and not three lines in the viewer
------------------------------------------------------
The merged CSV is written at the nerve sampling rate: one *anchor* row carrying
kinect data (``frame_index``, ``time_kinect``, ``contact_points``, …) followed by
roughly 32 interpolated rows that are NaN in every kinect column.  The viewer
draws its three timeseries against ``np.arange(len(merged_df))``, so "where is
frame *f* on the plot" is the question *what positional row is frame f's anchor*.

That question used to be answered with a multiplication:

    scale      = len(merged_df) / mkv_frame_count
    sample_idx = int(frame_idx * scale)

which encodes two assumptions the artifact does not make:

1. that the CSV spans the whole recording, and
2. that anchors are evenly spaced.

Neural-quality filtering breaks (1) — it removes trials, so the CSV covers fewer
frames than the MKV — and the anchor spacing was never exactly uniform anyway
(measured 33 or 34 rows apart on real blocks).  On
``2022-06-14_ST13-02/block-order-01`` the CSV holds 1,734 of the recording's
3,186 frames, giving a scale of 18.14 where the true spacing is 33.33: frame 437
resolved to row 7,927 instead of 14,566, and the error grew linearly to ~790
frames by the end of the block.  The displayed contact and the plotted contact
were simply different frames.

A multiplication cannot express a non-uniform mapping.  A lookup can, and stays
correct under truncation, mid-recording gaps, and any future re-indexing.

The navigable set is an ARRAY, not a count
------------------------------------------
``filter_by_neural_quality`` has two modes.  With
``discard_from_first_not2use: true`` it truncates from the first unusable trial,
leaving a contiguous ``0..N-1``; with ``false`` it removes individual trials,
leaving **gaps** in the middle of the recording.  A maximum frame, or a count,
describes only the first case.  The set of frames that exist describes both.

Purity contract
---------------
This module knows about a DataFrame, numpy arrays and integers.  It knows
nothing about Qt, VTK, sliders, sessions, paths or why a trial was excluded — it
takes the frames it is given and answers questions about them.  That is what
makes it unit-testable without a display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

__all__ = [
    "FrameNavigation",
    "build_frame_navigation_from_merged_df",
    "build_frame_navigation_over_range",
]


@dataclass(frozen=True)
class FrameNavigation:
    """The frames a block can display, and where each one is in the CSV.

    Parameters
    ----------
    frames:
        Sorted, strictly increasing, non-negative ``int64`` kinect frame
        indices — every frame the viewer may navigate to, and no others.
    rows:
        Positional merged-CSV row for each entry of *frames*, or ``None`` when
        there is no merged CSV at all (pure-3D mode).  ``rows`` is *positional*
        — the space ``np.arange(len(merged_df))`` indexes — never an index label.

    A **position** is an offset into *frames* (what the slider moves over); a
    **frame** is a real kinect frame index (what the MKV, the stickers and the
    hand model are indexed by).  The two coincide only when nothing was filtered
    out, which is exactly the case this type exists to stop assuming.
    """

    frames: np.ndarray
    rows: Optional[np.ndarray] = None

    _position_by_frame: Dict[int, int] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )
    _row_by_frame: Dict[int, int] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        frames = np.asarray(self.frames)
        if frames.ndim != 1:
            raise ValueError(
                f"frames must be one-dimensional; got shape {frames.shape}."
            )
        frames = frames.astype(np.int64, copy=False)
        if frames.size and frames[0] < 0:
            raise ValueError(
                f"frames must be non-negative; got {int(frames.min())}."
            )
        if frames.size > 1 and not np.all(np.diff(frames) > 0):
            raise ValueError(
                "frames must be sorted and strictly increasing — the navigable "
                "set is a set, and the slider moves over it in order."
            )
        object.__setattr__(self, "frames", frames)

        rows = self.rows
        if rows is not None:
            rows = np.asarray(rows)
            if rows.shape != frames.shape:
                raise ValueError(
                    "rows must hold exactly one merged-CSV row per frame: got "
                    f"shape {rows.shape} for {frames.shape[0]} frames."
                )
            rows = rows.astype(np.int64, copy=False)
            if rows.size and rows[0] < 0:
                raise ValueError(
                    f"rows must be non-negative positional indices; got {int(rows.min())}."
                )
            if rows.size > 1 and not np.all(np.diff(rows) > 0):
                raise ValueError(
                    "rows must be strictly increasing: a later frame cannot "
                    "anchor on an earlier row of the same recording."
                )
            object.__setattr__(self, "rows", rows)

        object.__setattr__(
            self,
            "_position_by_frame",
            {int(f): i for i, f in enumerate(frames)},
        )
        object.__setattr__(
            self,
            "_row_by_frame",
            {} if rows is None else {int(f): int(r) for f, r in zip(frames, rows)},
        )

    # -- size ---------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of navigable frames."""
        return int(self.frames.size)

    def __len__(self) -> int:
        return self.size

    @property
    def has_rows(self) -> bool:
        """Whether a merged CSV row is known for each frame."""
        return self.rows is not None

    # -- lookups (fail-fast; never clamp to a neighbour) ---------------------

    def contains(self, frame: int) -> bool:
        """Whether *frame* is navigable."""
        return int(frame) in self._position_by_frame

    def frame_at(self, position: int) -> int:
        """Return the kinect frame at navigation *position*."""
        if self.size == 0:
            raise IndexError(
                "This block has no navigable frame; there is nothing to display."
            )
        if not 0 <= int(position) < self.size:
            raise IndexError(
                f"Navigation position {position} is outside 0..{self.size - 1}."
            )
        return int(self.frames[int(position)])

    def position_of(self, frame: int) -> int:
        """Return the navigation position of *frame*.

        Raises ``KeyError`` when the frame is not navigable.  It is never
        rounded to the nearest navigable frame: silently showing a neighbour is
        the failure mode this whole module exists to remove.
        """
        try:
            return self._position_by_frame[int(frame)]
        except KeyError:
            raise KeyError(
                f"Kinect frame {frame} is not in the navigable set "
                f"({self.size} frames"
                + (
                    f", {int(self.frames[0])}..{int(self.frames[-1])}"
                    if self.size
                    else ""
                )
                + "). It was removed by neural-quality filtering, so no neural "
                "data exists beside it."
            ) from None

    def row_of(self, frame: int) -> int:
        """Return the positional merged-CSV row anchoring *frame*."""
        if self.rows is None:
            raise ValueError(
                "This navigation has no merged CSV, so no frame has a row. "
                "Ask for a row only when a CSV was supplied."
            )
        try:
            return self._row_by_frame[int(frame)]
        except KeyError:
            raise KeyError(
                f"Kinect frame {frame} has no anchor row in the merged CSV, so "
                "the timeseries cursor has nowhere correct to go."
            ) from None

    # -- presentation --------------------------------------------------------

    def format_label(self, position: int) -> str:
        """``"<kinect frame> (<position>/<navigable>)"``.

        The real kinect frame comes first and unmodified.  When the navigable
        set is a filtered subset the position alone would hide which frame of
        the recording is on screen — and the frame index is what every other
        artifact (the MKV, the sidecar, the CSV) is keyed by.
        """
        if self.size == 0:
            return "— (0/0)"
        return f"{self.frame_at(position)} ({int(position) + 1}/{self.size})"


def build_frame_navigation_over_range(n_frames: int) -> FrameNavigation:
    """Every frame of a recording is navigable, and none has a CSV row.

    This is the pure-3D case (``merged_csv_path is None``): nothing was
    filtered, so nothing is being hidden by offering the whole MKV.
    """
    if n_frames < 0:
        raise ValueError(f"n_frames must be non-negative; got {n_frames}.")
    return FrameNavigation(np.arange(int(n_frames), dtype=np.int64), None)


def build_frame_navigation_from_merged_df(
    merged_df: pd.DataFrame,
    source: Any = "<merged CSV>",
) -> FrameNavigation:
    """Derive the navigable frames and their rows from a merged CSV.

    *source* is only used to name the offending artifact in error messages.

    Anchors are the rows with a non-NaN ``frame_index``; the ~32 interpolated
    nerve-rate rows between them are NaN in every kinect column and are not
    frames.  Row positions are resolved with ``np.flatnonzero``, i.e.
    positionally, because the plots are drawn against
    ``np.arange(len(merged_df))`` and nothing promises the DataFrame index is a
    clean ``RangeIndex``.
    """
    if "frame_index" not in merged_df.columns:
        raise ValueError(
            f"Merged CSV '{source}' has no 'frame_index' column, so no kinect "
            f"frame can be located in it. Columns found: {list(merged_df.columns)}"
        )

    values = merged_df["frame_index"].to_numpy(dtype=np.float64, na_value=np.nan)
    anchor_rows = np.flatnonzero(~np.isnan(values))
    if anchor_rows.size == 0:
        raise ValueError(
            f"Merged CSV '{source}' has no non-NaN 'frame_index' value: it "
            "contains no kinect frame at all. An empty recording is not "
            "something the viewer can display."
        )

    frames_f = values[anchor_rows]
    if not np.all(frames_f == np.floor(frames_f)):
        bad = frames_f[frames_f != np.floor(frames_f)][:5]
        raise ValueError(
            f"Merged CSV '{source}' has non-integral 'frame_index' values (e.g. "
            f"{bad.tolist()}). A frame index that is not a whole frame cannot "
            "address the MKV."
        )
    frames = frames_f.astype(np.int64)

    order = np.argsort(frames, kind="stable")
    frames = frames[order]
    rows = anchor_rows[order].astype(np.int64)

    if frames.size > 1:
        dupes = frames[:-1][np.diff(frames) == 0]
        if dupes.size:
            raise ValueError(
                f"Merged CSV '{source}' anchors kinect frame {int(dupes[0])} on "
                "more than one row. A frame that maps to two rows has no single "
                "cursor position."
            )

    return FrameNavigation(frames, rows)
