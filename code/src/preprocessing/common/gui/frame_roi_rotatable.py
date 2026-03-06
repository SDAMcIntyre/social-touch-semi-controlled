import cv2
import numpy as np
from PIL import Image

try:
    from .frame_roi_square import FrameROISquare
except ImportError:
    import sys, os as _os
    sys.path.insert(0, _os.path.normpath(
        _os.path.join(_os.path.dirname(__file__), "..", "..", "..")))
    from preprocessing.common.gui.frame_roi_square import FrameROISquare


class FrameROIRotatable(FrameROISquare):
    """
    Extends FrameROISquare with rotation, resize, and drag-to-move capabilities.

    The ROI is stored internally as (center_x, center_y, width, height, angle_deg)
    instead of (x, y, w, h).  After a rectangle is drawn, four interactions
    become available via the left mouse button:

        - Drag a corner handle         → resize the ROI from its center.
        - Shift+Drag a corner handle   → rotate the ROI around its center.
        - Drag inside the rect         → translate the ROI.
        - Drag outside the rect        → discard the current ROI and draw a new one.

    Pan / zoom controls are unchanged from the parent class.

    Controls:
        - Left-Click Drag (empty area): Draw a new ROI.
        - Left-Click Drag (inside rect): Move the ROI.
        - Left-Click Drag (corner handle): Resize the ROI (width/height from center).
        - Shift+Left-Click Drag (corner handle): Rotate the ROI.
        - Right / Middle Mouse Button Drag: Pan the image.
        - Mouse Wheel: Zoom in / out.
        - CTRL+Z: Reset the ROI.
        - Enter / Space: Confirm the ROI.
        - ESC: Close the window.

    get_roi_data() returns::

        {
            'cx': float,        # centre x in original image coordinates
            'cy': float,        # centre y in original image coordinates
            'width': float,
            'height': float,
            'angle_deg': float, # counter-clockwise rotation in degrees
        }

    predefined_roi accepts either the base-class format::

        {'x': ..., 'y': ..., 'width': ..., 'height': ..., 'angle_deg': ...}

    or a centre-based format::

        {'cx': ..., 'cy': ..., 'width': ..., 'height': ..., 'angle_deg': ...}

    ``angle_deg`` is optional in both cases and defaults to 0.
    """

    # Radius (display pixels) within which a click activates a corner handle.
    HANDLE_RADIUS = 8

    # ------------------------------------------------------------------ #
    #  Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def __init__(self, image_input, is_rgb: bool = True,
                 window_title: str = None,
                 color_live: tuple = (0, 255, 0),
                 color_final: tuple = (0, 0, 255),
                 predefined_roi: dict = None):

        # Extract angle and normalise predefined_roi to x/y/w/h for the parent.
        self._pending_angle = 0.0
        parent_roi = predefined_roi
        if isinstance(predefined_roi, dict):
            self._pending_angle = float(predefined_roi.get('angle_deg', 0.0))
            if 'cx' in predefined_roi and 'cy' in predefined_roi:
                cx = predefined_roi['cx']
                cy = predefined_roi['cy']
                w  = predefined_roi['width']
                h  = predefined_roi['height']
                parent_roi = {'x': cx - w / 2.0, 'y': cy - h / 2.0,
                              'width': w, 'height': h}

        super().__init__(image_input, is_rgb=is_rgb, window_title=window_title,
                         color_live=color_live, color_final=color_final,
                         predefined_roi=parent_roi)

        # Convert any predefined roi_rect loaded by the parent into centre form.
        if self.roi_rect:
            x, y, w, h = self.roi_rect
            self._roi_cx: float | None = x + w / 2.0
            self._roi_cy: float | None = y + h / 2.0
            self._roi_w:  float | None = float(w)
            self._roi_h:  float | None = float(h)
            self.roi_rect = None  # subclass manages its own state
        else:
            self._roi_cx = None
            self._roi_cy = None
            self._roi_w  = None
            self._roi_h  = None

        self.roi_angle: float = self._pending_angle  # degrees, CCW

        # Interaction state
        self._mode: str | None = None          # 'drawing' | 'moving' | 'rotating' | 'resizing'
        self._drag_offset = (0.0, 0.0)         # (dx, dy) from ROI centre on move-start
        self._rotate_ref_angle  = 0.0          # angle(mouse→centre) on rotate-start
        self._rotate_init_angle = 0.0          # roi_angle on rotate-start

    # ------------------------------------------------------------------ #
    #  Geometry helpers                                                     #
    # ------------------------------------------------------------------ #

    def _has_roi(self) -> bool:
        return self._roi_cx is not None

    @staticmethod
    def _corners(cx: float, cy: float, w: float, h: float,
                 angle_deg: float) -> np.ndarray:
        """Return 4×2 array of corner coordinates (image space, CCW from TL)."""
        a = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(a), np.sin(a)
        hw, hh = w / 2.0, h / 2.0
        local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]],
                         dtype=np.float64)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return (rot @ local.T).T + np.array([cx, cy])

    def _roi_corners_img(self) -> np.ndarray | None:
        if not self._has_roi():
            return None
        return self._corners(self._roi_cx, self._roi_cy,
                             self._roi_w, self._roi_h, self.roi_angle)

    def _img_to_disp(self, pts: np.ndarray) -> np.ndarray:
        """Transform Nx2 image-space points to display (window) space."""
        return pts * self.zoom_factor + np.array([self.pan_offset_x,
                                                  self.pan_offset_y])

    def _disp_to_img(self, dx: float, dy: float) -> tuple[float, float]:
        return ((dx - self.pan_offset_x) / self.zoom_factor,
                (dy - self.pan_offset_y) / self.zoom_factor)

    def _is_inside_roi(self, img_x: float, img_y: float) -> bool:
        """True if (img_x, img_y) lies inside the rotated ROI."""
        if not self._has_roi():
            return False
        dx = img_x - self._roi_cx
        dy = img_y - self._roi_cy
        a = np.deg2rad(-self.roi_angle)
        local_x =  np.cos(a) * dx - np.sin(a) * dy
        local_y =  np.sin(a) * dx + np.cos(a) * dy
        return abs(local_x) <= self._roi_w / 2.0 and \
               abs(local_y) <= self._roi_h / 2.0

    def _nearest_corner_idx(self, disp_x: float, disp_y: float) -> int:
        """Index of the nearest corner handle in display space, or -1."""
        if not self._has_roi():
            return -1
        corners_disp = self._img_to_disp(self._roi_corners_img())
        for i, (cx, cy) in enumerate(corners_disp):
            if np.hypot(disp_x - cx, disp_y - cy) <= self.HANDLE_RADIUS:
                return i
        return -1

    # ------------------------------------------------------------------ #
    #  Mouse callback                                                       #
    # ------------------------------------------------------------------ #

    def _mouse_callback(self, event, x, y, flags, param):
        """Overrides the parent callback to add move and rotate modes."""

        # ── Pan / zoom: unconditionally delegate to parent ──────────────
        if event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN,
                     cv2.EVENT_RBUTTONUP,   cv2.EVENT_MBUTTONUP,
                     cv2.EVENT_MOUSEWHEEL):
            super()._mouse_callback(event, x, y, flags, param)
            return

        if event == cv2.EVENT_MOUSEMOVE and self.panning:
            super()._mouse_callback(event, x, y, flags, param)
            return

        # ── Convert to image coords ──────────────────────────────────────
        img_x, img_y = self._disp_to_img(x, y)

        # ── BUTTON DOWN ─────────────────────────────────────────────────
        if event == cv2.EVENT_LBUTTONDOWN:
            corner_idx = self._nearest_corner_idx(x, y)

            if corner_idx >= 0:
                if flags & cv2.EVENT_FLAG_SHIFTKEY:
                    self._mode = 'rotating'
                    dx = img_x - self._roi_cx
                    dy = img_y - self._roi_cy
                    self._rotate_ref_angle  = np.rad2deg(np.arctan2(dy, dx))
                    self._rotate_init_angle = self.roi_angle
                else:
                    self._mode = 'resizing'

            elif self._is_inside_roi(img_x, img_y):
                self._mode = 'moving'
                self._drag_offset = (img_x - self._roi_cx,
                                     img_y - self._roi_cy)

            else:
                self._mode = 'drawing'
                self._clear_roi()
                self.drawing = True
                self.roi_start_point = (img_x, img_y)
                self.roi_end_point   = (img_x, img_y)

            self._update_display()
            return

        # ── MOUSE MOVE ──────────────────────────────────────────────────
        if event == cv2.EVENT_MOUSEMOVE:
            if self._mode == 'rotating':
                dx = img_x - self._roi_cx
                dy = img_y - self._roi_cy
                current = np.rad2deg(np.arctan2(dy, dx))
                self.roi_angle = self._rotate_init_angle + (current - self._rotate_ref_angle)
                self._update_display()

            elif self._mode == 'resizing':
                # Project mouse position into the ROI local frame.
                dx = img_x - self._roi_cx
                dy = img_y - self._roi_cy
                a = np.deg2rad(-self.roi_angle)
                local_x = np.cos(a) * dx - np.sin(a) * dy
                local_y = np.sin(a) * dx + np.cos(a) * dy
                new_w = 2.0 * abs(local_x)
                new_h = 2.0 * abs(local_y)
                if new_w > 2:
                    self._roi_w = new_w
                if new_h > 2:
                    self._roi_h = new_h
                self._update_display()

            elif self._mode == 'moving':
                self._roi_cx = img_x - self._drag_offset[0]
                self._roi_cy = img_y - self._drag_offset[1]
                self._update_display()

            elif self._mode == 'drawing' and self.drawing:
                self.roi_end_point = (img_x, img_y)
                self._update_display()
            return

        # ── BUTTON UP ───────────────────────────────────────────────────
        if event == cv2.EVENT_LBUTTONUP:
            if self._mode == 'drawing' and self.drawing:
                self.drawing = False
                x1, y1 = self.roi_start_point
                x2, y2 = img_x, img_y
                w, h = abs(x2 - x1), abs(y2 - y1)
                if w > 2 and h > 2:
                    self._roi_cx = (x1 + x2) / 2.0
                    self._roi_cy = (y1 + y2) / 2.0
                    self._roi_w  = w
                    self._roi_h  = h
                    self.roi_angle = 0.0
            self._mode = None
            self._update_display()

    # ------------------------------------------------------------------ #
    #  Display                                                              #
    # ------------------------------------------------------------------ #

    def _update_display(self):
        """Redraws the image with zoom/pan and overlays the rotatable ROI."""
        M = np.float32([
            [self.zoom_factor, 0, self.pan_offset_x],
            [0, self.zoom_factor, self.pan_offset_y]
        ])
        display_image = cv2.warpAffine(
            self.original_image, M, (self.window_w, self.window_h))

        # ── Live drawing preview (axis-aligned) ──────────────────────────
        if self.drawing and self.roi_start_point and self.roi_end_point:
            sx = int(self.roi_start_point[0] * self.zoom_factor + self.pan_offset_x)
            sy = int(self.roi_start_point[1] * self.zoom_factor + self.pan_offset_y)
            ex = int(self.roi_end_point[0]   * self.zoom_factor + self.pan_offset_x)
            ey = int(self.roi_end_point[1]   * self.zoom_factor + self.pan_offset_y)
            cv2.rectangle(display_image, (sx, sy), (ex, ey), self.color_live, 2)
            cv2.line(display_image, (sx, sy), (ex, ey), self.color_live, 2)
            cv2.line(display_image, (sx, ey), (ex, sy), self.color_live, 2)

        # ── Finalized rotatable ROI ───────────────────────────────────────
        elif self._has_roi():
            corners_img  = self._roi_corners_img()
            corners_disp = self._img_to_disp(corners_img).astype(np.int32)

            # Outline
            cv2.polylines(display_image, [corners_disp], isClosed=True,
                          color=self.color_final, thickness=2)

            # Diagonals
            cv2.line(display_image,
                     tuple(corners_disp[0]), tuple(corners_disp[2]),
                     self.color_final, 2)
            cv2.line(display_image,
                     tuple(corners_disp[1]), tuple(corners_disp[3]),
                     self.color_final, 2)

            # Corner handles (filled circles)
            for pt in corners_disp:
                cv2.circle(display_image, tuple(pt),
                           self.HANDLE_RADIUS, self.color_final, -1)

            # Centre crosshair
            cx_d = int(self._roi_cx * self.zoom_factor + self.pan_offset_x)
            cy_d = int(self._roi_cy * self.zoom_factor + self.pan_offset_y)
            cv2.drawMarker(display_image, (cx_d, cy_d), self.color_final,
                           cv2.MARKER_CROSS, 14, 2)

            # Angle label
            label = f"{self.roi_angle % 360:.1f}\u00b0"
            cv2.putText(display_image, label, (cx_d + 8, cy_d - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_image, label, (cx_d + 8, cy_d - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Command overlay ───────────────────────────────────────────────
        commands = [
            "Pan: Right/Middle-Click Drag",
            "Zoom: Mouse Wheel",
            "Draw ROI: Left-Click Drag (empty area)",
            "Move ROI: Left-Click Drag (inside rect)",
            "Resize ROI: Left-Click Drag (corner handle)",
            "Rotate ROI: Shift+Left-Click Drag (corner handle)",
            "Confirm: 'Proceed' Button, Enter, or Space",
            "Reset: CTRL+Z",
            "Exit: ESC",
        ]
        y0, dy = 25, 20
        for i, line in enumerate(commands):
            yy = y0 + i * dy
            cv2.putText(display_image, line, (15, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_image, line, (15, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(self.window_name, display_image)

    # ------------------------------------------------------------------ #
    #  ROI management                                                       #
    # ------------------------------------------------------------------ #

    def _clear_roi(self):
        self._roi_cx = None
        self._roi_cy = None
        self._roi_w  = None
        self._roi_h  = None
        self.roi_angle       = 0.0
        self.roi_start_point = None
        self.roi_end_point   = None

    def reset_roi(self):
        """Clears the drawn ROI (overrides parent)."""
        print("ROI has been reset.")
        self._clear_roi()
        self.drawing = False
        self._mode   = None
        self._update_display()

    def confirm_roi(self):
        """Confirms the ROI and prepares to exit (overrides parent)."""
        if not self._has_roi():
            print("No ROI selected. Draw a rectangle first.")
            return
        print("✅ ROI Confirmed:")
        print(f"   - Centre (cx, cy): ({self._roi_cx:.1f}, {self._roi_cy:.1f})")
        print(f"   - Dimensions (w, h): ({self._roi_w:.1f}, {self._roi_h:.1f})")
        print(f"   - Angle: {self.roi_angle:.2f}°")
        self.roi_confirmed = True

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_roi_data(self) -> dict | None:
        """
        Returns the confirmed ROI as a dictionary, or None.

        Keys: ``cx``, ``cy``, ``width``, ``height``, ``angle_deg``.
        """
        if not self._has_roi():
            return None
        return {
            'cx':        self._roi_cx,
            'cy':        self._roi_cy,
            'width':     self._roi_w,
            'height':    self._roi_h,
            'angle_deg': self.roi_angle,
        }


# --------------------------------------------------------------------------- #
#  POC                                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Build a synthetic 800×600 test image with a visible grid and centre mark.
    H, W = 600, 800
    img = np.full((H, W, 3), 40, dtype=np.uint8)

    # Light grid
    for gx in range(0, W, 50):
        cv2.line(img, (gx, 0), (gx, H), (70, 70, 70), 1)
    for gy in range(0, H, 50):
        cv2.line(img, (0, gy), (W, gy), (70, 70, 70), 1)

    # Centre crosshair
    cv2.drawMarker(img, (W // 2, H // 2), (200, 200, 200),
                   cv2.MARKER_CROSS, 20, 1)

    # Load with a predefined rotated ROI (centre-based format, 30° tilt).
    predefined = {'cx': 400, 'cy': 300, 'width': 200, 'height': 100, 'angle_deg': 30}

    selector = FrameROIRotatable(
        img,
        is_rgb=False,
        window_title="FrameROIRotatable — POC",
        predefined_roi=predefined,
    )
    selector.run()

    roi = selector.get_roi_data()
    if roi:
        print("\nFinal ROI:", roi)
    else:
        print("\nNo ROI confirmed.")
