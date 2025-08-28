"""
Load a single .tiff frame from a folder, mirroring the
single-frame reading flow from xyz_extractor.py (without PyK4A).

This is a minimal first step: it discovers .tif/.tiff files,
loads the first frame as a NumPy array (via OpenCV), and returns
its data and path. You can later extend this into an iterator to
stream frames like a playback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


# --- Data structures to loosely mimic a playback/capture ---

@dataclass
class TIFFCapture:
    """Lightweight stand-in for a Capture object."""
    path: Path
    image: np.ndarray  # raw image data loaded from the .tiff file


class TIFFPlayback:
    """
    Minimal playback-like wrapper for a directory of .tif/.tiff frames.

    This intentionally mirrors the iteration style of PyK4APlayback so
    you can later swap it into extractor-like flows.
    """

    def __init__(self, frames_dir: str | Path):
        self.frames_dir = Path(frames_dir)
        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {self.frames_dir}")
        self._frame_paths = self._discover_frames(self.frames_dir)
        if not self._frame_paths:
            raise FileNotFoundError(
                f"No .tif/.tiff files found in: {self.frames_dir}")

    @staticmethod
    def _discover_frames(frames_dir: Path) -> List[Path]:
        # Match both .tif and .tiff (case-insensitive)
        candidates = list(frames_dir.glob("**/*.tif")) + list(frames_dir.glob("**/*.tiff"))
        # Natural sort by numbers in filename if present
        def sort_key(p: Path):
            nums = re.findall(r"\d+", p.stem)
            return [int(n) for n in nums] if nums else [p.stem]
        return sorted(candidates, key=sort_key)

    def __len__(self) -> int:
        return len(self._frame_paths)

    def get_first_capture(self) -> TIFFCapture:
        """Load and return the first frame as a TIFFCapture."""
        first_path = self._frame_paths[0]
        image = self._read_image(first_path)
        return TIFFCapture(path=first_path, image=image)

    def get_capture(self, index: int) -> TIFFCapture:
        """Load and return capture at a given index."""
        if index < 0 or index >= len(self._frame_paths):
            raise IndexError(f"Frame index out of range: {index} (0..{len(self._frame_paths)-1})")
        path = self._frame_paths[index]
        image = self._read_image(path)
        return TIFFCapture(path=path, image=image)

    def _read_image(self, path: Path) -> np.ndarray:
        """
        Reads a .tiff image using OpenCV. Preserves depth/bitness when possible.
        - Returns a NumPy array with shape (H, W) for grayscale/depth,
          or (H, W, C) for multi-channel images.
        """
        # IMREAD_UNCHANGED keeps 16-bit depth if the file is 16-bit.
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        return img


# --- Minimal single-frame flow ---

def load_single_tiff_frame(frames_dir: str | Path) -> Tuple[np.ndarray, Path]:
    """
    Discover .tiff frames and load the first one, returning (image, path).

    This mirrors the outer structure of xyz_extractor's frame loop by
    constructing a playback-like object and retrieving one capture.
    """
    playback = TIFFPlayback(frames_dir)
    capture = playback.get_first_capture()
    # For parity with xyz_extractor's printouts
    print(f"Processing frame: 0", end='\r')
    # You can add additional per-frame logic here later
    return capture.image, capture.path


# --- Tracking CSV helpers ---

def load_roi_centers_for_frame(csv_path: str | Path, frame_index: int) -> Dict[str, Tuple[float, float]]:
    """
    Load ROI rectangles for a specific frame and compute center (px, py) per object.

    Expects columns: object_name, frame_id, roi_x, roi_y, roi_width, roi_height.
    Returns: { object_name: (px, py), ... }
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {csv_path}")

    # If the provided file is empty, raise a clear error; user handles it manually.
    try:
        if csv_path.is_file() and csv_path.stat().st_size == 0:
            raise ValueError(
                f"Tracking CSV is empty (0 bytes): {csv_path}. "
                f"Please regenerate it or pass a non-empty CSV path.")
    except OSError:
        # If we cannot stat the file for some reason, proceed to read and let pandas raise.
        pass

    df = pd.read_csv(csv_path)

    required = {"object_name", "frame_id", "roi_x", "roi_y", "roi_width", "roi_height"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    df_f = df[df["frame_id"] == frame_index].copy()
    if df_f.empty:
        return {}

    # Compute centers; pandas handles Int64/nullable dtypes
    df_f["px"] = df_f["roi_x"].astype(float) + df_f["roi_width"].astype(float) / 2.0
    df_f["py"] = df_f["roi_y"].astype(float) + df_f["roi_height"].astype(float) / 2.0

    centers: Dict[str, Tuple[float, float]] = {}
    for _, r in df_f.iterrows():
        name = str(r["object_name"]) if pd.notna(r["object_name"]) else "unknown"
        px = float(r["px"]) if pd.notna(r["px"]) else np.nan
        py = float(r["py"]) if pd.notna(r["py"]) else np.nan
        centers[name] = (px, py)
    return centers


def point_cloud_to_display(img_pc: np.ndarray) -> np.ndarray:
    """
    Convert a point-cloud TIFF (H,W,3 int16) to a displayable BGR image.
    Uses Z channel normalized to [0,255] and applies a colormap.
    """
    if img_pc.ndim == 3 and img_pc.shape[2] >= 3:
        z = img_pc[..., 2]
    else:
        # Fallback for grayscale inputs
        z = img_pc
    # Normalize ignoring zeros if possible
    mask = z > 0
    if np.any(mask):
        z_valid = z[mask]
        z_min, z_max = float(z_valid.min()), float(z_valid.max())
    else:
        z_min, z_max = float(z.min()), float(z.max())
    if z_max <= z_min:
        z8 = np.zeros_like(z, dtype=np.uint8)
    else:
        z8 = np.clip((z.astype(np.float32) - z_min) * (255.0 / (z_max - z_min)), 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(z8, cv2.COLORMAP_JET)
    return color


def draw_centers(img_bgr: np.ndarray, centers: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Draw center points with labels on a copy of the image."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for i, (name, (px, py)) in enumerate(centers.items()):
        if np.isnan(px) or np.isnan(py):
            continue
        ix, iy = int(round(px)), int(round(py))
        if 0 <= ix < w and 0 <= iy < h:
            color = (0, 255, 255) if i % 3 == 0 else (0, 255, 0) if i % 3 == 1 else (255, 0, 0)
            cv2.circle(out, (ix, iy), 6, color, 2, cv2.LINE_AA)
            cv2.putText(out, name, (ix + 8, iy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def visualize_orientation_check(frames_dir: str | Path, csv_path: str | Path, frame_index: int = 0) -> None:
    """
    Show the selected frame with ROI centers overlaid on both the raw TIFF orientation
    and its transposed version to decide whether a transpose is needed.
    """
    playback = TIFFPlayback(frames_dir)
    # Pick capture by index; assumes sorted order aligns with frame_id
    capture = playback.get_capture(frame_index if frame_index < len(playback) else 0)
    centers = load_roi_centers_for_frame(csv_path, frame_index)

    raw_disp = point_cloud_to_display(capture.image)
    raw_overlay = draw_centers(raw_disp, centers)

    transposed = np.transpose(capture.image, (1, 0, 2)) if capture.image.ndim == 3 else capture.image.T
    trans_disp = point_cloud_to_display(transposed)
    trans_overlay = draw_centers(trans_disp, centers)

    # Add labels
    cv2.putText(raw_overlay, f"Raw {raw_overlay.shape[1]}x{raw_overlay.shape[0]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(trans_overlay, f"Transposed {trans_overlay.shape[1]}x{trans_overlay.shape[0]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Fit both overlays into equal-sized canvases for a clean side-by-side view
    def fit_into_canvas(img: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(box_w / w, box_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        canvas = np.zeros((box_h, box_w, 3), dtype=img.dtype)
        y0 = (box_h - new_h) // 2
        x0 = (box_w - new_w) // 2
        canvas[y0:y0+new_h, x0:x0+new_w] = resized
        return canvas

    box_w, box_h = 960, 540
    left = fit_into_canvas(raw_overlay, box_w, box_h)
    right = fit_into_canvas(trans_overlay, box_w, box_h)
    mosaic = np.hstack([left, right])

    # Auto-resize mosaic to fit on screen (keep ~95% of screen size)
    def get_screen_size() -> Tuple[Optional[int], Optional[int]]:
        try:
            import tkinter as tk  # standard library
            root = tk.Tk()
            root.withdraw()
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            return int(w), int(h)
        except Exception:
            return None, None

    scr_w, scr_h = get_screen_size()
    if scr_w and scr_h:
        max_w = int(scr_w * 0.95)
        max_h = int(scr_h * 0.95)
        if mosaic.shape[1] > max_w or mosaic.shape[0] > max_h:
            scale = min(max_w / mosaic.shape[1], max_h / mosaic.shape[0])
            new_w = max(1, int(round(mosaic.shape[1] * scale)))
            new_h = max(1, int(round(mosaic.shape[0] * scale)))
            mosaic = cv2.resize(mosaic, (new_w, new_h), interpolation=cv2.INTER_AREA)

    window = "TIFF Orientation Check"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, mosaic)
    print("Press any key (window focused) to close...")
    cv2.waitKey(0)
    cv2.destroyWindow(window)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TIFF utilities for Kinect point clouds")
    parser.add_argument("frames_dir", type=str, help="Path to folder with .tif/.tiff frames")
    parser.add_argument("--tracking-csv", type=str, default=None, help="ROI tracking CSV path for overlay")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index to visualize")
    args = parser.parse_args()

    if args.tracking_csv:
        visualize_orientation_check(args.frames_dir, args.tracking_csv, args.frame_index)
    else:
        image, path = load_single_tiff_frame(args.frames_dir)
        h, w = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        dtype = image.dtype
        print(f"\nLoaded: {path}")
        print(f"Shape: {h}x{w}x{channels}, dtype: {dtype}")
