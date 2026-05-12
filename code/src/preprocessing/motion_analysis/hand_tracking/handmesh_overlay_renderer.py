from __future__ import annotations
import pickle
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Callable, Optional

from preprocessing.common import VideoMP4Manager, ColorFormat

logger = logging.getLogger(__name__)


def resolve_hand_motion_npz_path(raw_npz_path: Path) -> Path:
    """Return the best available hand-motion NPZ for *raw_npz_path*.

    Checks for the stabilised variant (*_handmodel_motion_stabilised.npz)
    first; falls back to the raw NPZ (*_handmodel_motion.npz).  Raises
    FileNotFoundError if neither exists.
    """
    stabilised_path = raw_npz_path.with_name(
        raw_npz_path.name.replace("_handmodel_motion.npz", "_handmodel_motion_stabilised.npz")
    )
    if stabilised_path.exists():
        return stabilised_path
    if raw_npz_path.exists():
        return raw_npz_path
    raise FileNotFoundError(
        f"No hand-motion NPZ found. Checked:\n"
        f"  stabilised: {stabilised_path}\n"
        f"  raw:        {raw_npz_path}"
    )


class HandTrackingDataManager:
    def __init__(self, pkl_path: Path) -> None:
        self.pkl_path = pkl_path
        if not pkl_path.exists():
            raise FileNotFoundError(f"Hand-tracking data not found: {pkl_path}")
        logger.info("Loading hand-tracking data: %s", pkl_path)
        with open(pkl_path, "rb") as f:
            self._data: list[dict] = pickle.load(f)
        logger.info("Loaded %d frames.", len(self._data))

    def get_hand_geometry(self, frame_index: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if frame_index < 0 or frame_index >= len(self._data):
            return None
        api_resp = self._data[frame_index].get("api_response", {})
        if not api_resp or api_resp.get("error"):
            return None
        hands = api_resp.get("hands", [])
        if not hands:
            return None
        hand = hands[0]
        if "vertices_pixel" not in hand or "faces" not in hand:
            return None
        vertices = np.array(hand["vertices_pixel"], dtype=np.float32)
        faces = np.array(hand["faces"], dtype=np.int32)
        if vertices.size == 0 or faces.size == 0:
            return None
        return vertices, faces


class BatchVideoRenderer:
    def __init__(
        self,
        video_path: Path,
        data_path: Path,
        output_path: Optional[Path] = None,
        target_fps: int = 30,
    ) -> None:
        self.video_path = video_path
        self.output_path = output_path
        self.target_fps = target_fps
        self._video = VideoMP4Manager(str(video_path), color_format=ColorFormat.RGB)
        self._data = HandTrackingDataManager(data_path)
        self._total_frames = len(self._video)

    @property
    def frame_count(self) -> int:
        return self._total_frames

    def render_frame(self, frame_idx: int) -> np.ndarray:
        """Return a single BGR frame with the hand overlay drawn."""
        frame_rgb = self._video[frame_idx]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return self._draw_overlay(frame_bgr, frame_idx)

    def _draw_overlay(self, image: np.ndarray, frame_idx: int) -> np.ndarray:
        geometry = self._data.get_hand_geometry(frame_idx)
        if geometry is None:
            cv2.putText(image, "No Tracking Data", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2)
            return image
        vertices, faces = geometry
        pts_int = vertices.astype(np.int32)
        triangles = pts_int[faces]
        cv2.polylines(image, list(triangles), isClosed=True, color=(0, 255, 255), thickness=1)
        return image

    def render(
        self,
        progress_cb: Optional[Callable[[int, int], bool]] = None,
    ) -> None:
        """Render the overlay video.

        progress_cb is called per frame with (current, total). If it returns
        True the render is cancelled, the writer released, and a partial MP4
        remains on disk. On any other failure the writer is released and the
        exception re-raised (fail-fast convention).
        """
        if self.output_path is None:
            raise ValueError("output_path must be set to call render()")
        first = self._video[0]
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(self.output_path), fourcc, self.target_fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for: {self.output_path}")
        try:
            for i in range(self._total_frames):
                frame_rgb = self._video[i]
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                processed = self._draw_overlay(frame_bgr, i)
                writer.write(processed)
                if progress_cb is not None and progress_cb(i + 1, self._total_frames):
                    break
        finally:
            writer.release()
        logger.info("Render complete: %s", self.output_path)
