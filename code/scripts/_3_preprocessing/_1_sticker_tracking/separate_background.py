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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

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


def load_all_roi_centers(csv_path: str | Path) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """
    Load the entire tracking CSV once and build centers per frame.
    Returns: { frame_id: { object_name: (px, py) } }
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {csv_path}")
    try:
        if csv_path.is_file() and csv_path.stat().st_size == 0:
            raise ValueError(
                f"Tracking CSV is empty (0 bytes): {csv_path}. "
                f"Please regenerate it or pass a non-empty CSV path.")
    except OSError:
        pass

    df = pd.read_csv(csv_path)
    required = {"object_name", "frame_id", "roi_x", "roi_y", "roi_width", "roi_height"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Compute centers once
    df["px"] = df["roi_x"].astype(float) + df["roi_width"].astype(float) / 2.0
    df["py"] = df["roi_y"].astype(float) + df["roi_height"].astype(float) / 2.0

    centers_by_frame: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for frame_id, grp in df.groupby("frame_id"):
        frame_centers: Dict[str, Tuple[float, float]] = {}
        for _, r in grp.iterrows():
            name = str(r["object_name"]) if pd.notna(r["object_name"]) else "unknown"
            frame_centers[name] = (float(r["px"]) if pd.notna(r["px"]) else np.nan,
                                   float(r["py"]) if pd.notna(r["py"]) else np.nan)
        centers_by_frame[int(frame_id)] = frame_centers
    return centers_by_frame


def point_cloud_to_display(img_pc: np.ndarray, z_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Convert a point-cloud TIFF (H,W,3 int16) to a displayable BGR image.
    Uses Z channel normalized to [0,255] and applies a colormap.
    """
    if img_pc.ndim == 3 and img_pc.shape[2] >= 3:
        z = img_pc[..., 2]
    else:
        # Fallback for grayscale inputs
        z = img_pc
    # Normalize ignoring zeros AND keep zeros black to reveal invalid areas.
    # If a global z_range is provided, use it; otherwise compute per-frame.
    mask = (z > 0)
    if z_range is not None:
        z_min, z_max = z_range
    else:
        if np.any(mask):
            min_val, max_val, _, _ = cv2.minMaxLoc(z, mask=mask.astype(np.uint8))
            z_min, z_max = float(min_val), float(max_val)
        else:
            z_min, z_max = float(z.min()), float(z.max())

    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        z8 = np.zeros_like(z, dtype=np.uint8)
    else:
        scale = 255.0 / (z_max - z_min)
        zf = (z.astype(np.float32) - z_min) * scale
        # Clip negatives and cap at 255
        zf = np.clip(zf, 0.0, 255.0)
        z8 = zf.astype(np.uint8)
        # Keep invalid pixels black
        z8[~mask] = 0
    color = cv2.applyColorMap(z8, cv2.COLORMAP_JET)
    # Force invalid pixels to true black in the colored image
    if mask.ndim == 2:
        color[~mask] = (0, 0, 0)
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
    z_range: Optional[Tuple[float, float]] = None,
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
    disp = point_cloud_to_display(img, z_range=z_range)
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


def _normalize_depth_mode_name(mode_str: str) -> str:
    s = (mode_str or "").strip()
    if s.startswith("DepthMode."):
        s = s.split(".", 1)[1]
    return s.upper()


def _depth_mode_to_zrange(mode_str: str) -> Optional[Tuple[float, float]]:
    # Approximate working ranges from Azure Kinect documentation (in mm)
    ranges = {
        "NFOV_UNBINNED": (500.0, 3865.0),
        "NFOV_2X2BINNED": (500.0, 5460.0),
        "WFOV_UNBINNED": (250.0, 2945.0),
        "WFOV_2X2BINNED": (250.0, 3865.0),
    }
    return ranges.get(_normalize_depth_mode_name(mode_str))


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
    fps: int = 30,
    cache_size: int = 24,
    depth_mode: str = "DepthMode.NFOV_UNBINNED",
) -> None:
    """Interactive viewer to step frames and play back the sequence.

    Controls: SPACE play/pause, n/→ next, b/← prev, +/- speed, q/ESC quit.
    """
    playback = TIFFPlayback(frames_dir)
    idx = max(0, min(start_index, len(playback) - 1))
    delay_ms = max(1, int(1000 / max(1, fps)))
    frame_period = 1.0 / max(1, fps)
    last_step_ts = time.time()
    window = "TIFF Sequence Viewer"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # Preload tracking once
    centers_all = load_all_roi_centers(csv_path)

    # Fixed Z range based on depth mode
    fixed_z_range = _depth_mode_to_zrange(depth_mode)
    mode_label = _normalize_depth_mode_name(depth_mode)

    # Simple LRU caches for decoded frames and colorized displays
    class LRU:
        def __init__(self, capacity: int):
            self.capacity = max(1, capacity)
            self._d: OrderedDict = OrderedDict()
        def get(self, key):
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
            return None
        def put(self, key, value):
            self._d[key] = value
            self._d.move_to_end(key)
            if len(self._d) > self.capacity:
                self._d.popitem(last=False)

    img_cache = LRU(cache_size)
    disp_cache = LRU(cache_size)

    # Add UI controls: progress slider + play/pause toggle
    cv2.createTrackbar('Frame', window, idx, len(playback) - 1, lambda v: None)
    cv2.createTrackbar('Play', window, 1 if play else 0, 1, lambda v: None)

    while True:
        # Sync with slider only when paused; during playback we drive the slider
        if not play:
            slider_pos = cv2.getTrackbarPos('Frame', window)
            if slider_pos != idx:
                idx = slider_pos
                last_step_ts = time.time()

        # Get base image (with/without transpose) from cache or disk
        key_img = (idx, transpose)
        img = img_cache.get(key_img)
        if img is None:
            capture = playback.get_capture(idx)
            base = transpose_point_cloud(capture.image) if transpose else capture.image
            img_cache.put(key_img, base)
            img = base

        # Determine which Z range to use: fixed from depth mode, else per-frame
        if fixed_z_range is not None:
            active_z_range = fixed_z_range
            disp_key_suffix = (round(active_z_range[0], 2), round(active_z_range[1], 2), 'F')
        else:
            z = img[..., 2] if img.ndim == 3 and img.shape[2] >= 3 else img
            mask = (z > 0).astype(np.uint8)
            if mask.any():
                mn, mx, _, _ = cv2.minMaxLoc(z, mask=mask)
                active_z_range = (float(mn), float(mx))
            else:
                active_z_range = (0.0, 1.0)
            disp_key_suffix = ('P',)

        # Get colorized display from cache
        key_disp = (idx, transpose, disp_key_suffix)
        disp = disp_cache.get(key_disp)
        if disp is None:
            disp = point_cloud_to_display(img, z_range=active_z_range)
            disp_cache.put(key_disp, disp)

        centers = centers_all.get(idx, {})
        overlay = draw_centers(disp, centers)

        # Info panel (with Z range readout and mode)
        mode_str = f"fixed:{mode_label}" if fixed_z_range is not None else 'per-frame'
        zr = active_z_range
        info = [
            f"Frame {idx+1}/{len(playback)}",
            f"{'Transposed' if transpose else 'Raw'} {overlay.shape[1]}x{overlay.shape[0]}",
            f"Z: [{zr[0]:.0f}, {zr[1]:.0f}] mm  mode: {mode_str}",
            "SPACE play/pause  n/→ next  b/← prev  +/- speed  q quit",
            "slider=Frame  Play(0/1)",
        ]
        y = 28
        for line in info:
            _draw_text_with_bg(overlay, line, (10, y), (255, 255, 255), alpha=0.5)
            y += 24

        display_img = _fit_to_screen(overlay, margin=0.95)
        cv2.imshow(window, display_img)

        # Keep the slider in sync
        cv2.setTrackbarPos('Frame', window, idx)

        # Sync play state from trackbar
        track_play = cv2.getTrackbarPos('Play', window) == 1
        if track_play != play:
            play = track_play
            last_step_ts = time.time()

        # Advance playback based on wall clock, not waitKey return value
        now = time.time()
        if play:
            # Step forward by the number of elapsed frame periods
            elapsed = now - last_step_ts
            steps = int(elapsed / frame_period)
            if steps > 0:
                idx = min(idx + steps, len(playback) - 1)
                last_step_ts += steps * frame_period
                if idx >= len(playback) - 1:
                    play = False
                    cv2.setTrackbarPos('Play', window, 0)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord(' '):
            play = not play
            cv2.setTrackbarPos('Play', window, 1 if play else 0)
            last_step_ts = time.time()
        elif key in (ord('n'), 83):  # right arrow
            idx = min(idx + 1, len(playback) - 1)
            last_step_ts = time.time()
        elif key in (ord('b'), 81):  # left arrow
            idx = max(idx - 1, 0)
            last_step_ts = time.time()
        elif key in (ord('+'), ord('=')):
            fps = min(120, fps + 1)
            delay_ms = max(1, int(1000 / fps))
            frame_period = 1.0 / max(1, fps)
        elif key in (ord('-'), ord('_')):
            fps = max(1, fps - 1)
            delay_ms = max(1, int(1000 / fps))
            frame_period = 1.0 / max(1, fps)
        # no 'z' toggle; using fixed or per-frame based on --depth-mode



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TIFF utilities for Kinect point clouds")
    parser.add_argument("frames_dir", type=str, help="Path to folder with .tif/.tiff frames")
    parser.add_argument("--tracking-csv", type=str, default=None, help="ROI tracking CSV path for overlay")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index to visualize/start from")
    parser.add_argument("--no-transpose", action="store_true", help="Display raw TIFF without transposing")
    parser.add_argument("--play", action="store_true", help="Start in playback mode")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS when --play is enabled (default: 30, Azure Kinect)" )
    parser.add_argument("--cache-size", type=int, default=64, help="LRU cache size for frames/displays")
    parser.add_argument("--depth-mode", type=str, default="DepthMode.NFOV_UNBINNED",
                        help="Fixed depth range by Azure Kinect mode (e.g., DepthMode.NFOV_UNBINNED, WFOV_UNBINNED). If unknown, falls back to per-frame.")
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
            cache_size=max(1, args.cache_size),
            depth_mode=args.depth_mode,
        )
    else:
        image, path = load_single_tiff_frame(args.frames_dir)
        h, w = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        dtype = image.dtype
        print(f"\nLoaded: {path}")
        print(f"Shape: {h}x{w}x{channels}, dtype: {dtype}")
