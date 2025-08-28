import cv2
import numpy as np
from typing import Dict, Any, Optional

from ..models.xyz_metadata_model import XYZMetadataConfig

class XYZVisualizationHandler:
    """Encapsulates all visualization logic using OpenCV."""

    def __init__(self, config: XYZMetadataConfig):
        self.is_enabled = config.monitor or config.source_video_path is not None
        self.video_path = config.source_video_path
        self.display_dims = config.display_dims
        self.writer: Optional[cv2.VideoWriter] = None

    def setup_writer(self, fps: int):
        if not self.video_path:
            return
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.video_path, fourcc, fps, (self.display_dims[1], self.display_dims[0]))
        if not self.writer.isOpened():
            raise IOError(f"Could not open video writer for path: {self.video_path}")
        print(f"Saving monitoring video to: {self.video_path} at {fps} FPS")

    def _resize_with_padding(self, image: np.ndarray, target_dims: tuple) -> np.ndarray:
        # (Implementation from original script)
        target_w, target_h = target_dims
        if image.shape[0] == 0 or image.shape[1] == 0:
            shape = (target_h, target_w, image.shape[2]) if len(image.shape) == 3 else (target_h, target_w)
            return np.zeros(shape, dtype=image.dtype)

        src_h, src_w = image.shape[:2]
        aspect_ratio = src_w / src_h
        target_ratio = target_w / target_h

        new_w, new_h = (target_w, int(target_w / aspect_ratio)) if aspect_ratio > target_ratio else (int(target_h * aspect_ratio), target_h)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        padded_shape = (target_h, target_w, image.shape[2]) if len(image.shape) == 3 else (target_h, target_w)
        padded_image = np.zeros(padded_shape, dtype=image.dtype)
        
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return padded_image

    def _color_from_name(self, name: str) -> tuple:
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

    def create_frame(self, frame_index: int, depth_image: np.ndarray, monitoring_data: Dict[str, Any]) -> np.ndarray:
        # (Combines logic from _create_monitoring_frame)
        display_h, display_w = self.display_dims
        panel_width = 450
        images_total_width = display_w - panel_width

        # 1. Prepare Depth Visualization
        depth_vis = np.zeros((display_h, images_total_width, 3), dtype=np.uint8)
        if depth_image is not None:
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            jet_map = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # Draw markers before resizing
            for name, data in monitoring_data.items():
                px, py = data.get('px', np.nan), data.get('py', np.nan)
                if not np.isnan(px) and not np.isnan(py):
                    color = self._color_from_name(name)
                    cv2.drawMarker(jet_map, (int(round(px)), int(round(py))), color, cv2.MARKER_CROSS, 20, 2)
            
            depth_vis = self._resize_with_padding(jet_map, (images_total_width, display_h))

        # 2. Create Canvas and Draw Text
        canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        canvas[:, :images_total_width] = depth_vis
        
        # 3. Draw Text Panel
        text_x, text_y = images_total_width + 15, 30
        cv2.putText(canvas, f"Frame: {frame_index}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 40
        for name, data in monitoring_data.items():
            color = self._color_from_name(name)
            cv2.putText(canvas, f"- Sticker: {name}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            text_y += 25
            px_str = f"Pixel: ({data.get('px', np.nan):.1f}, {data.get('py', np.nan):.1f})"
            cv2.putText(canvas, px_str, (text_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text_y += 25
            xyz_str = f"XYZ(mm): ({data.get('x_mm', np.nan):.1f}, {data.get('y_mm', np.nan):.1f}, {data.get('z_mm', np.nan):.1f})"
            cv2.putText(canvas, xyz_str, (text_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text_y += 40

        return canvas

    def process_frame(self, frame: np.ndarray):
        """Displays or writes the frame."""
        if self.writer:
            self.writer.write(frame)
        else: # Assumes monitor=True if writer is None
            cv2.imshow("XYZ Monitoring", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return True # Request to quit
        return False

    def release(self):
        """Releases video writer and destroys OpenCV windows."""
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
