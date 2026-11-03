"""Image panel: scaled raster, value readout, and a colorbar."""

from __future__ import annotations

import numpy as np

from ai_insar.gui.qt import Qt, QtCore, QtGui, QtWidgets, Signal


class ImageView(QtWidgets.QWidget):
    """Draws an RGB raster and reports the data pixel under a click."""

    pixelActivated = Signal(int, int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QtGui.QImage | None = None
        self._values: np.ndarray | None = None
        self.setMinimumSize(360, 280)
        self.setMouseTracking(True)

    def has_raster(self) -> bool:
        return self._image is not None and not self._image.isNull()

    def set_raster(self, rgb: np.ndarray | None, values: np.ndarray | None) -> None:
        if rgb is None:
            self._image = None
            self._values = None
        else:
            self._image = _rgb_to_qimage(rgb)
            self._values = None if values is None else np.asarray(values)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt name)
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#12171f"))
        if self._image is None or self._image.isNull():
            painter.setPen(QtGui.QColor("#9aa7bd"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Generate or load an SLC stack, then run the numbered stages.",
            )
            painter.end()
            return
        target = self._target_rect()
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(target, self._image)
        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        pixel = self._pixel_at(event.position())
        if pixel is not None:
            self.pixelActivated.emit(pixel[0], pixel[1])
        super().mousePressEvent(event)

    def _pixel_at(self, point: QtCore.QPointF) -> tuple[int, int] | None:
        if self._values is None:
            return None
        target = self._target_rect()
        if target.isEmpty() or not target.contains(point):
            return None
        height, width = self._values.shape[:2]
        col = int((point.x() - target.x()) / target.width() * width)
        row = int((point.y() - target.y()) / target.height() * height)
        col = min(max(col, 0), width - 1)
        row = min(max(row, 0), height - 1)
        return row, col

    def _target_rect(self) -> QtCore.QRectF:
        if self._image is None or self._image.isNull():
            return QtCore.QRectF()
        frame = self.rect().adjusted(12, 12, -12, -12)
        if frame.width() <= 0 or frame.height() <= 0:
            return QtCore.QRectF()
        scale = min(frame.width() / self._image.width(), frame.height() / self._image.height())
        drawn_w = self._image.width() * scale
        drawn_h = self._image.height() * scale
        x = frame.x() + (frame.width() - drawn_w) / 2.0
        y = frame.y() + (frame.height() - drawn_h) / 2.0
        return QtCore.QRectF(x, y, drawn_w, drawn_h)


class ColorBar(QtWidgets.QWidget):
    """Horizontal colorbar with min and max labels."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QtGui.QImage | None = None
        self._vmin = ""
        self._vmax = ""
        self._caption = ""
        self.setMinimumHeight(48)

    def set_bar(self, rgb_row: np.ndarray | None, vmin: str, vmax: str, caption: str) -> None:
        if rgb_row is None:
            self._image = None
        else:
            row = np.ascontiguousarray(rgb_row, dtype=np.uint8)
            if row.ndim == 1:
                row = row[None, :]
            self._image = _rgb_to_qimage(row if row.ndim == 3 else row.reshape(1, -1, 3))
        self._vmin = vmin
        self._vmax = vmax
        self._caption = caption
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#1b212c"))
        painter.setPen(QtGui.QColor("#c5d0e0"))
        if self._image is None:
            painter.drawText(self.rect().adjusted(8, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter, self._caption)
            painter.end()
            return
        bar = QtCore.QRect(12, 8, max(10, self.width() - 24), 14)
        painter.drawImage(bar, self._image)
        painter.drawText(12, 38, self._vmin)
        painter.drawText(self.rect().adjusted(0, 24, -12, 0), Qt.AlignmentFlag.AlignRight, self._vmax)
        painter.end()


def _rgb_to_qimage(rgb: np.ndarray) -> QtGui.QImage:
    array = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, channels = array.shape
    if channels != 3:
        raise ValueError("expected an RGB image")
    buffer = array.tobytes()
    view = QtGui.QImage(buffer, width, height, 3 * width, QtGui.QImage.Format.Format_RGB888)
    # copy() detaches from ``buffer``, which is released when this function returns.
    return view.copy()
