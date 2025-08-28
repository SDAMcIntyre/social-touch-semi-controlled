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
import sys
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


def transpose_point_cloud(img: np.ndarray) -> np.ndarray:
    """
    Transpose a point-cloud image by swapping width and height.

    - For 3D arrays (H, W, C), returns (W, H, C).
    - For 2D arrays  (H, W),   returns (W, H).

    This is a pure array transpose: dtype/values are unchanged and
    the result is typically a view (no copy) unless later operations
    require a contiguous buffer.
    """
    if img.ndim == 3:
        return np.transpose(img, (1, 0, 2))
    if img.ndim == 2:
        return img.T
    raise ValueError("transpose_point_cloud expects a 2D or 3D numpy array")


def _color_from_name(name: str) -> Tuple[int, int, int]:
    n = name.lower()
    mapping = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'cyan': (255, 255, 0),
        'teal': (255, 255, 0),
        'magenta': (255, 0, 255),
        'purple': (255, 0, 255),
        'pink': (255, 0, 255),
        'orange': (0, 165, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'grey': (128, 128, 128),
        'gray': (128, 128, 128),
        'brown': (19, 69, 139),
    }
    for key, bgr in mapping.items():
        if key in n:
            return bgr
    return (0, 255, 255)


def _draw_text_with_bg(img: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int], alpha: float = 0.6) -> None:
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    pad = 4
    x0, y0 = max(0, x - pad), max(0, y - th - pad)
    x1, y1 = min(img.shape[1], x + tw + pad), min(img.shape[0], y + base + pad)
    roi = img[y0:y1, x0:x1]
    if roi.size:
        overlay = roi.copy()
        overlay[:] = (0, 0, 0)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_centers(img_bgr: np.ndarray, centers: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Draw center markers and color-matched labels with semi-opaque backgrounds.

    Attempts to avoid overlapping labels by trying a set of offsets.
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]
    placed: List[Tuple[int, int, int, int]] = []
    for name, (px, py) in centers.items():
        if np.isnan(px) or np.isnan(py):
            continue
        ix, iy = int(round(px)), int(round(py))
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        color = _color_from_name(name)
        cv2.circle(out, (ix, iy), 6, color, 2, cv2.LINE_AA)

        # Place label with anti-overlap attempts
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        (tw, th), base = cv2.getTextSize(name, font, scale, thickness)
        candidates = [(8, -8), (8, 16), (-8, -8), (-8, 16), (8, 32), (-8, 32)]
        chosen = None
        for dx, dy in candidates:
            x = min(max(0, ix + dx), max(0, w - tw - 1))
            y = min(max(th + 1, iy + dy), max(th + 1, h - 1))
            box = (x - 4, y - th - 4, x + tw + 4, y + base + 4)
            if all(box[2] < bx0 or box[0] > bx1 or box[3] < by0 or box[1] > by1 for bx0, by0, bx1, by1 in placed):
                chosen = (x, y, box)
                break
        if chosen is None:
            x, y = ix + 8, iy - 8
            box = (x - 4, y - th - 4, x + tw + 4, y + base + 4)
        else:
            x, y, box = chosen
        placed.append(box)
        _draw_text_with_bg(out, name, (x, y), color, alpha=0.6)

    return out


def visualize_orientation_check(
    frames_dir: str | Path,
    csv_path: str | Path,
    frame_index: int = 0,
    transpose: bool = True,
) -> None:
    """
    Show the selected frame with ROI centers overlaid.

    - If `transpose` is True (default), display the TRANSPOSED TIFF
      so indexing matches (py, px) coordinates.
    - If False, display the raw TIFF orientation.
    """
    playback = TIFFPlayback(frames_dir)
    # Pick capture by index; assumes sorted order aligns with frame_id
    capture = playback.get_capture(frame_index if frame_index < len(playback) else 0)
    centers = load_roi_centers_for_frame(csv_path, frame_index)

    img = transpose_point_cloud(capture.image) if transpose else capture.image
    disp = point_cloud_to_display(img)
    overlay = draw_centers(disp, centers)

    # Add label
    label = ("Transposed" if transpose else "Raw") + f" {overlay.shape[1]}x{overlay.shape[0]}"
    cv2.putText(overlay, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Auto-resize image to fit on screen (keep ~95% of screen size)
    display_img = _fit_to_screen(overlay, margin=0.95)

    window = "TIFF Orientation Check"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, display_img)
    print("Press any key (window focused) to close...")
    cv2.waitKey(0)
    cv2.destroyWindow(window)


def _get_screen_size_safe() -> Tuple[Optional[int], Optional[int]]:
    """Get main screen size without importing tkinter (prevents macOS Tk crashes).

    macOS: use CoreGraphics via ctypes. Others: return None to skip resizing.
    """
    try:
        if sys.platform == 'darwin':
            from ctypes import cdll, Structure, c_double, c_uint32

            class CGSize(Structure):
                _fields_ = [("width", c_double), ("height", c_double)]

            class CGPoint(Structure):
                _fields_ = [("x", c_double), ("y", c_double)]

            class CGRect(Structure):
                _fields_ = [("origin", CGPoint), ("size", CGSize)]

            cg = cdll.LoadLibrary("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
            cg.CGMainDisplayID.restype = c_uint32
            cg.CGDisplayBounds.argtypes = [c_uint32]
            cg.CGDisplayBounds.restype = CGRect
            did = cg.CGMainDisplayID()
            bounds = cg.CGDisplayBounds(did)
            return int(bounds.size.width), int(bounds.size.height)
    except Exception:
        pass
    return None, None


def _fit_to_screen(img: np.ndarray, margin: float = 0.95) -> np.ndarray:
    scr_w, scr_h = _get_screen_size_safe()
    if not (scr_w and scr_h):
        return img
    max_w = int(scr_w * margin)
    max_h = int(scr_h * margin)
    if img.shape[1] <= max_w and img.shape[0] <= max_h:
        return img
    scale = min(max_w / img.shape[1], max_h / img.shape[0])
    new_w = max(1, int(round(img.shape[1] * scale)))
    new_h = max(1, int(round(img.shape[0] * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def visualize_tiff_sequence(
    frames_dir: str | Path,
    csv_path: str | Path,
    start_index: int = 0,
    transpose: bool = True,
    play: bool = False,
    fps: int = 15,
) -> None:
    """Interactive viewer to step frames and play back the sequence.

    Controls: SPACE play/pause, n/→ next, b/← prev, +/- speed, q/ESC quit.
    """
    playback = TIFFPlayback(frames_dir)
    idx = max(0, min(start_index, len(playback) - 1))
    delay_ms = max(1, int(1000 / max(1, fps)))
    window = "TIFF Sequence Viewer"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        capture = playback.get_capture(idx)
        centers = load_roi_centers_for_frame(csv_path, idx)
        img = transpose_point_cloud(capture.image) if transpose else capture.image
        disp = point_cloud_to_display(img)
        overlay = draw_centers(disp, centers)

        # Info panel
        info = [
            f"Frame {idx+1}/{len(playback)}",
            f"{'Transposed' if transpose else 'Raw'} {overlay.shape[1]}x{overlay.shape[0]}",
            "SPACE play/pause  n/→ next  b/← prev  +/- speed  q quit",
        ]
        y = 28
        for line in info:
            _draw_text_with_bg(overlay, line, (10, y), (255, 255, 255), alpha=0.5)
            y += 24

        display_img = _fit_to_screen(overlay, margin=0.95)
        cv2.imshow(window, display_img)

        wait = delay_ms if play else 0
        key = cv2.waitKey(wait) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord(' '):
            play = not play
        elif key in (ord('n'), 83):  # right arrow
            idx = min(idx + 1, len(playback) - 1)
        elif key in (ord('b'), 81):  # left arrow
            idx = max(idx - 1, 0)
        elif key in (ord('+'), ord('=')):
            fps = min(120, fps + 1); delay_ms = max(1, int(1000 / fps))
        elif key in (ord('-'), ord('_')):
            fps = max(1, fps - 1); delay_ms = max(1, int(1000 / fps))

        if play and key == 255:  # no key pressed
            if idx + 1 < len(playback):
                idx += 1
            else:
                play = False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TIFF utilities for Kinect point clouds")
    parser.add_argument("frames_dir", type=str, help="Path to folder with .tif/.tiff frames")
    parser.add_argument("--tracking-csv", type=str, default=None, help="ROI tracking CSV path for overlay")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index to visualize/start from")
    parser.add_argument("--no-transpose", action="store_true", help="Display raw TIFF without transposing")
    parser.add_argument("--play", action="store_true", help="Start in playback mode")
    parser.add_argument("--fps", type=int, default=15, help="Playback FPS when --play is enabled")
    args = parser.parse_args()

    if args.tracking_csv:
        # Use interactive sequence viewer by default
        visualize_tiff_sequence(
            args.frames_dir,
            args.tracking_csv,
            start_index=args.frame_index,
            transpose=(not args.no_transpose),
            play=args.play,
            fps=args.fps,
        )
    else:
        image, path = load_single_tiff_frame(args.frames_dir)
        h, w = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        dtype = image.dtype
        print(f"\nLoaded: {path}")
        print(f"Shape: {h}x{w}x{channels}, dtype: {dtype}")
