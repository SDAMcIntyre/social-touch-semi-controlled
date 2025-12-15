# Standard library imports
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import cv2
import numpy as np
import pandas as pd
import sys
import qimage2ndarray

# Local application/library specific imports
# Assumes these modules exist in the user's environment
from utils.should_process_task import should_process_task
from preprocessing.common import VideoMP4Manager
from preprocessing.stickers_analysis import (
    ColorSpaceFileHandler,
    ColorSpaceManager,
    ColorSpaceStatus,
    FittedEllipsesFileHandler,
    FittedEllipsesManager

)

import sys
import cv2
import numpy as np
import pandas as pd
import qimage2ndarray
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QSpinBox, QFormLayout)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap

def launch_monitor_gui(frames: np.ndarray, df: pd.DataFrame, threshold: int, title: str):
    """
    Launches a concise, hardware-accelerated PyQt5 GUI for reviewing fitting results.
    Refactored for PyQt5 compatibility (scoped enums removed).
    """
    
    class MonitorCanvas(QWidget):
        """Handles high-performance rendering of the frame and vector overlay."""
        def __init__(self):
            super().__init__()
            self.img, self.data = None, None
            self.setMinimumSize(640, 480)

        def update_view(self, img, data):
            self.img, self.data = img, data
            self.update()  # Triggers paintEvent

        def paintEvent(self, event):
            if self.img is None: return
            p = QPainter(self)
            # PyQt5 Enum Change: QPainter.RenderHint.Antialiasing -> QPainter.Antialiasing
            p.setRenderHint(QPainter.Antialiasing)

            # 1. Process and Draw Image
            _, bin_img = cv2.threshold(self.img, threshold, 255, cv2.THRESH_BINARY)
            # Ensure contiguous array for safe QImage conversion
            if not bin_img.flags['C_CONTIGUOUS']: bin_img = np.ascontiguousarray(bin_img)
            
            q_img = qimage2ndarray.array2qimage(bin_img, normalize=False)
            
            # PyQt5 Enum Change: Qt.AspectRatioMode.KeepAspectRatio -> Qt.KeepAspectRatio
            # PyQt5 Enum Change: Qt.TransformationMode.SmoothTransformation -> Qt.SmoothTransformation
            pix = QPixmap.fromImage(q_img).scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            # Center the image
            ox = (self.width() - pix.width()) // 2
            oy = (self.height() - pix.height()) // 2
            p.drawPixmap(ox, oy, pix)

            # 2. Draw Vector Overlay
            if self.data is not None and not pd.isna(self.data.get('center_x')):
                # Calculate scale factors based on original vs displayed size
                sx, sy = pix.width() / self.img.shape[1], pix.height() / self.img.shape[0]
                
                # Transform coordinate system to ellipse center
                p.translate(ox + self.data['center_x'] * sx, oy + self.data['center_y'] * sy)
                
                # Apply Rotation
                # OpenCV angle is degrees clockwise from horizontal. 
                # QPainter rotate is also clockwise degrees.
                p.rotate(self.data['angle'])
                
                # Draw Ellipse
                p.setPen(QPen(QColor(0, 255, 0), 2))  # Green
                
                # Retrieve raw dimensions (width/height of the rect)
                w_raw = self.data.get('axis_width', self.data.get('axes_major', 0))
                h_raw = self.data.get('axis_height', self.data.get('axes_minor', 0))

                w, h = w_raw * sx, h_raw * sy
                
                # Draw centered ellipse. QPainter expects top-left of the bounding rect, 
                # so we offset by -w/2, -h/2.
                p.drawEllipse(int(-w/2), int(-h/2), int(w), int(h))
                
                # Draw Center Point
                p.setBrush(QColor(255, 0, 0))  # Red
                p.drawEllipse(-3, -3, 6, 6)

    class MonitorWindow(QMainWindow):
        """Main controller for UI layout and data binding."""
        def __init__(self):
            super().__init__()
            self.setWindowTitle(title)
            self.canvas = MonitorCanvas()
            
            # UI Controls
            # PyQt5 Enum Change: Qt.Orientation.Horizontal -> Qt.Horizontal
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, len(frames) - 1)
            self.spin = QSpinBox()
            self.spin.setRange(0, len(frames) - 1)
            
            # Data Display
            self.labels = {k: QLabel("-") for k in ['center_x', 'center_y', 'axes_major', 'axes_minor', 'angle', 'score']}
            
            # Layout
            panel = QWidget()
            pl = QVBoxLayout(panel)
            pl.addWidget(QLabel("<b>Navigation</b>"))
            pl.addWidget(self.slider)
            pl.addWidget(self.spin)
            pl.addWidget(QLabel("<b>Metrics</b>"))
            
            form = QFormLayout()
            for k, v in self.labels.items(): form.addRow(k, v)
            pl.addLayout(form)
            pl.addStretch()

            root = QWidget()
            rl = QHBoxLayout(root)
            rl.addWidget(self.canvas, 3)  # Canvas takes 75% width
            rl.addWidget(panel, 1)
            self.setCentralWidget(root)

            # Event Binding
            self.slider.valueChanged.connect(self.sync)
            self.spin.valueChanged.connect(self.sync)
            self.sync(0)  # Init

        def sync(self, idx):
            # Update inputs without triggering recursion
            self.slider.blockSignals(True); self.spin.blockSignals(True)
            self.slider.setValue(idx); self.spin.setValue(idx)
            self.slider.blockSignals(False); self.spin.blockSignals(False)
            
            # Update Data
            row = df[df['frame_number'] == idx]
            data = row.iloc[0] if not row.empty else None
            
            # Update Canvas
            self.canvas.update_view(frames[idx], data)
            
            # Update Text
            if data is not None:
                for k, lbl in self.labels.items():
                    val = data.get(k, np.nan)
                    lbl.setText(f"{val:.2f}" if isinstance(val, (float, int)) else "N/A")
            else:
                for lbl in self.labels.values(): lbl.setText("-")

    # Application Lifecycle Management
    app = QApplication.instance()
    if not app: app = QApplication(sys.argv)
    
    window = MonitorWindow()
    window.resize(1100, 700)
    window.show()
    # PyQt5 Legacy Safety: exec_() avoids Python 2 keywords, mostly standard in PyQt5
    app.exec_()


def fit_ellipse_on_frame(binary_frame: np.ndarray) -> Optional[Dict]:
    """
    Fits an ellipse to the largest contour in a binary frame.
    
    Refactored to preserve raw OpenCV geometry (width/height) rather than 
    forcing major/minor assignment, ensuring compatibility with visualization tools.

    Args:
        binary_frame (np.ndarray): The input binary frame (single channel, 0 or 255).

    Returns:
        Optional[Dict]: A dictionary with the ellipse data or None if no suitable contour is found.
    """
    # --- Contour Detection and Validation ---
    if binary_frame.ndim != 2 or binary_frame.dtype != np.uint8:
        print("Error: Input frame must be a single-channel 8-bit image (CV_8UC1).")
        return None

    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    # An ellipse needs at least 5 points to be fitted
    if len(largest_contour) < 5:
        return None

    # --- Ellipse Fitting ---
    # cv2.fitEllipse returns ((center_x, center_y), (width, height), angle)
    ellipse = cv2.fitEllipse(largest_contour)
    
    # --- Scoring and Return Structure ---
    contour_area = cv2.contourArea(largest_contour)
    axes = ellipse[1] # (width, height)

    # A valid ellipse must have non-zero axes
    if axes[0] <= 0 or axes[1] <= 0:
        return None
        
    # Area = pi * (width/2) * (height/2)
    ellipse_area = math.pi * (axes[0] / 2.0) * (axes[1] / 2.0)
    
    # Score is the ratio of the smaller area to the larger area to detect fit quality
    score = min(contour_area, ellipse_area) / max(contour_area, ellipse_area)

    return {
        'center_x': ellipse[0][0],
        'center_y': ellipse[0][1],
        'axes_major': axes[1],
        'axes_minor': axes[0],
        'angle': ellipse[2],     # Angle in degrees
        'score': score
    }


def fit_ellipses_on_grayscale_frames(frames: List[np.ndarray], threshold: int) -> pd.DataFrame:
    """
    Processes a list of grayscale frames by first applying a binary threshold
    and then finding the best-fit ellipse on the result.

    Args:
        frames (List[np.ndarray]): A list of grayscale video frames.
        threshold (int): The pixel intensity value (0-255) to use for binary thresholding.

    Returns:
        pd.DataFrame: A DataFrame with the best-fit ellipse data for each frame.
    """
    all_frame_results = []

    for frame_number, frame in enumerate(frames):
        if frame is None or frame.size == 0:
            print(f"Frame {frame_number}: Frame is empty, adding NaN row.")
            best_result_for_frame = None
        else:
            # Convert the grayscale frame to a binary image using the provided threshold.
            _, binary_frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
            
            # Fit ellipse directly on the binary frame
            best_result_for_frame = fit_ellipse_on_frame(binary_frame)

        if best_result_for_frame:
            best_result_for_frame['frame_number'] = frame_number
            all_frame_results.append(best_result_for_frame)
        else:
            # If no ellipse was found, append a row with NaN values
            nan_result = {
                'frame_number': frame_number, 
                'center_x': np.nan, 'center_y': np.nan,
                'axes_major': np.nan, 'axes_minor': np.nan, 
                'angle': np.nan, 'score': np.nan,
            }
            all_frame_results.append(nan_result)
            
    return pd.DataFrame(all_frame_results)

def fit_ellipses_on_correlation_videos(
    video_path: Path,
    md_path: Path,
    output_path: Path,
    *,
    force_processing: bool = False,
    monitor: bool = False
):
    """
    Orchestrates the video processing workflow for grayscale videos.
    It loads grayscale frames, applies a binary threshold from metadata,
    fits ellipses, and saves the results using the dedicated manager and file handler.

    Args:
        video_path (Path): Path to the source video.
        md_path (Path): Path to the metadata file.
        output_path (Path): Path where results should be saved.
        force_processing (bool): If True, processes video even if checks suggest skipping.
        monitor (bool): If True, opens a GUI to visualize results after processing each object.
    """
    colorspace_manager: ColorSpaceManager = ColorSpaceFileHandler.load(md_path)
    print(f"--- Starting Video Processing ---")
    # 2. Initialize the manager to handle results in memory.
    try: 
        results_manager: FittedEllipsesManager = FittedEllipsesFileHandler.load(output_path)
    except:
        results_manager = FittedEllipsesManager()
    
    processed_occured = False
    # 3. Process each object sequentially
    for name in colorspace_manager.colorspace_names:
        if "discarded" in name:
            continue
        print(f"\nProcessing '{name}'...")
        current_colorspace = colorspace_manager.get_colorspace(name)
        input_video_path = video_path.parent / (video_path.stem + f"_{name}.mp4")
        
        # 4. Decide whether to process this specific object
        need_to_process = should_process_task(output_paths=output_path, input_paths=input_video_path, force=force_processing)
        if not need_to_process and not (current_colorspace.status == ColorSpaceStatus.TO_BE_PROCESSED.value):
            print(f"Skipping '{name}' (status: '{current_colorspace.status}').")
            continue
        
        print(f"Loading video '{input_video_path}'...")
        try:
            frames_grayscale = VideoMP4Manager(input_video_path).get_frames()
            frames_grayscale_array = np.array(frames_grayscale)
            # Ensure correct shape handling if get_frames returns (N, H, W, C)
            if frames_grayscale_array.ndim == 4:
                frames_grayscale_squeezed = frames_grayscale_array[:, :, :, 0]
            else:
                frames_grayscale_squeezed = frames_grayscale_array
        except Exception as e:
            print(f"Error loading frames for {name}: {e}")
            continue

        # Process the grayscale frames to get ellipse data for this object
        object_ellipse_df = fit_ellipses_on_grayscale_frames(
            frames=frames_grayscale_squeezed,
            threshold=current_colorspace.threshold
        )
        
        # Add the resulting DataFrame to the manager.
        results_manager.add_or_update_result(name, object_ellipse_df)
        processed_occured = True
        print(f"✅ Finished processing and stored results for '{name}' in manager.")
        
        # --- Monitor Invocation ---
        if monitor:
            launch_monitor_gui(
                frames=frames_grayscale_squeezed,
                df=object_ellipse_df,
                title=f"Monitor: {name}",
                threshold=current_colorspace.threshold
            )
        
    # 4. Save all results using the dedicated file handler.
    if processed_occured:
        print("\n--- Aggregating and Saving Results ---")
        FittedEllipsesFileHandler.save(results_manager, output_path)
        print(f"✅ Results successfully saved to '{output_path}'.")
        print(f"--- Processing Complete ---")
    else:
        print(f"--- All object skipped. No need to update the results. ---")