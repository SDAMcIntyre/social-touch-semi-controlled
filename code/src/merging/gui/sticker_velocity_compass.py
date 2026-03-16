# Standard library imports
from math import atan2, sqrt, cos, sin

# Third-party imports
import numpy as np
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget

from typing import Optional


class StickerVelocityCompass(QWidget):
    """
    A QPainter-based widget that displays a 2D velocity compass for a single
    sticker, showing direction (XY plane) and Z-speed (via arrow color).

    The outer ring is drawn in the sticker's assigned color.
    The arrow points in the XY direction of velocity.
    Arrow color shifts from blue (moving toward camera) to red (moving away).

    Usage::

        compass = StickerVelocityCompass("sticker_blue", "blue")
        # ... on each frame:
        compass.update_velocity(np.array([dx, dy, dz]))
    """

    def __init__(self, sticker_name: str, sticker_color: str, parent=None):
        super().__init__(parent)

        self._sticker_name: str = sticker_name
        self._sticker_color: QColor = QColor(sticker_color)
        self._v_xyz: Optional[np.ndarray] = None

        # Tunable: how many mm/frame equals a full-radius arrow
        self._scale_mm_per_frame: float = 5.0

        self.setFixedSize(120, 140)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_velocity(self, v_xyz: np.ndarray) -> None:
        """Store the latest per-frame displacement vector and schedule repaint."""
        self._v_xyz = v_xyz
        self.update()  # triggers paintEvent via Qt event loop

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def sizeHint(self) -> QSize:
        return QSize(120, 140)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # --- Outer border ring (sticker color) ---
        r = self.rect().adjusted(4, 4, -4, -4)
        painter.setPen(QPen(self._sticker_color, 4))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(r)

        # --- Dark background fill ---
        inner_r = r.adjusted(4, 4, -4, -4)
        painter.setBrush(QBrush(QColor(30, 30, 30)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(inner_r)

        # Compute geometry parameters
        center = r.center()
        radius = min(r.width(), r.height()) / 2.0 - 8.0

        # --- Velocity arrow ---
        speed_label = ""
        if self._v_xyz is not None:
            vx, vy, vz = float(self._v_xyz[0]), float(self._v_xyz[1]), float(self._v_xyz[2])
            mag_xy = sqrt(vx ** 2 + vy ** 2)
            mag_total = sqrt(vx ** 2 + vy ** 2 + vz ** 2)

            speed_label = f"{mag_total:.1f} mm/f"

            if mag_xy > 1e-6:
                # Arrow direction in screen space (flip Y because screen Y goes downward)
                theta = atan2(-vy, vx)

                # Arrow length proportional to XY speed, capped at 85% of radius
                arrow_len = min(
                    mag_xy / self._scale_mm_per_frame * radius,
                    radius * 0.85
                )

                # Z-to-color: positive vz → red (moving away), negative vz → blue (approaching)
                z_norm = float(np.clip(vz / 3.0, -1.0, 1.0))
                r_c = int(128 + 127 * z_norm)
                b_c = int(128 - 127 * z_norm)
                arrow_color = QColor(r_c, 50, b_c)

                tip_x = center.x() + cos(theta) * arrow_len
                tip_y = center.y() + sin(theta) * arrow_len

                # Arrow shaft
                painter.setPen(QPen(arrow_color, 2))
                painter.drawLine(
                    int(center.x()), int(center.y()),
                    int(tip_x), int(tip_y)
                )

                # Arrowhead triangle (filled)
                head_len = min(arrow_len * 0.35, radius * 0.28)
                head_angle = 0.45  # radians half-angle

                left_x = tip_x - head_len * cos(theta - head_angle)
                left_y = tip_y - head_len * sin(theta - head_angle)
                right_x = tip_x - head_len * cos(theta + head_angle)
                right_y = tip_y - head_len * sin(theta + head_angle)

                from PyQt5.QtGui import QPolygonF
                from PyQt5.QtCore import QPointF
                triangle = QPolygonF([
                    QPointF(tip_x, tip_y),
                    QPointF(left_x, left_y),
                    QPointF(right_x, right_y),
                ])
                painter.setBrush(QBrush(arrow_color))
                painter.setPen(Qt.NoPen)
                painter.drawPolygon(triangle)

        # --- Absolute velocity label (small gray text, bottom of circle interior) ---
        if speed_label:
            painter.setPen(QPen(QColor(160, 160, 160)))
            font = QFont()
            font.setPointSize(6)
            painter.setFont(font)
            label_y = int(center.y() + radius * 0.72)
            painter.drawText(
                int(center.x() - 38), label_y,
                76, 14,
                Qt.AlignCenter,
                speed_label
            )

        # --- Sticker name label (8pt white text, below circle) ---
        painter.setPen(QPen(QColor(255, 255, 255)))
        name_font = QFont()
        name_font.setPointSize(8)
        painter.setFont(name_font)
        painter.drawText(
            0, self.height() - 22,
            self.width(), 20,
            Qt.AlignCenter,
            self._sticker_name
        )
