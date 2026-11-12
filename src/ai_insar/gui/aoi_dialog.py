"""Draw or type the processing area on the reference and secondary footprints."""

from __future__ import annotations

from ai_insar.gui.qt import Qt, QtCore, QtGui, QtWidgets
from ai_insar.pipeline.aoi import (
    AoiError,
    LonLatPolygon,
    format_wkt,
    geographic_bounds,
    parse_wkt,
    polygon_from_bounds,
    prepare_aoi,
)


class FootprintMap(QtWidgets.QWidget):
    """Quick-look of two SAFE footprints with a rectangle or polygon AOI.

    The drawing is in longitude/latitude. It is the map SNAP shows when a
    product is open, reduced to the frames and the region.
    """

    regionChanged = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("aoiFootprintMap")
        self.setMinimumSize(420, 280)
        self.setMouseTracking(True)
        self._reference: tuple[tuple[float, float], ...] = ()
        self._secondary: tuple[tuple[float, float], ...] = ()
        self._aoi: tuple[tuple[float, float], ...] = ()
        self._mode = "rectangle"
        self._draft: list[tuple[float, float]] = []
        self._press: tuple[float, float] | None = None

    def set_footprints(
        self,
        reference: tuple[tuple[float, float], ...] | None,
        secondary: tuple[tuple[float, float], ...] | None,
    ) -> None:
        self._reference = tuple(reference or ())
        self._secondary = tuple(secondary or ())
        self.update()

    def set_mode(self, mode: str) -> None:
        self._mode = "polygon" if mode == "polygon" else "rectangle"
        self._draft.clear()
        self._press = None
        self.update()

    def set_aoi(self, polygon: LonLatPolygon | None) -> None:
        self._aoi = () if polygon is None else polygon.closed()
        self._draft.clear()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#12171f"))
        bounds = self._view_bounds()
        if bounds is None:
            painter.setPen(QtGui.QColor("#9aa7bd"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load a SAFE footprint to draw an AOI.")
            painter.end()
            return
        self._stroke(painter, self._reference, QtGui.QColor("#7eb6ff"), bounds, fill=QtGui.QColor(80, 140, 220, 40))
        self._stroke(painter, self._secondary, QtGui.QColor("#f0b429"), bounds, fill=QtGui.QColor(240, 180, 40, 40))
        self._stroke(painter, self._aoi, QtGui.QColor("#ff5d5d"), bounds, fill=QtGui.QColor(255, 80, 80, 70))
        if self._draft:
            self._stroke(painter, tuple(self._draft), QtGui.QColor("#ffd0d0"), bounds, fill=None)
        painter.setPen(QtGui.QColor("#c5d0e0"))
        painter.drawText(12, 22, "Blue: reference    Amber: secondary    Red: AOI")
        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        point = self._geo_at(event.position())
        if point is None:
            return
        if self._mode == "rectangle":
            self._press = point
            self._draft = [point, point]
        else:
            if event.button() == Qt.MouseButton.RightButton and len(self._draft) >= 3:
                self._commit(self._draft)
                return
            self._draft.append(point)
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._mode != "rectangle" or self._press is None:
            return
        point = self._geo_at(event.position())
        if point is None:
            return
        self._draft = [self._press, point]
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._mode != "rectangle" or self._press is None:
            return
        point = self._geo_at(event.position())
        self._press = None
        if point is None:
            self._draft.clear()
            self.update()
            return
        self._commit_rectangle(self._draft[0], point)
        self._draft.clear()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._mode == "polygon" and len(self._draft) >= 3:
            self._commit(self._draft)

    def _commit_rectangle(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        west, east = sorted((start[0], end[0]))
        south, north = sorted((start[1], end[1]))
        if east - west < 1.0e-8 or north - south < 1.0e-8:
            return
        self._commit(list(polygon_from_bounds(north, south, west, east).closed()))

    def _commit(self, vertices: list[tuple[float, float]]) -> None:
        polygon = LonLatPolygon(tuple(vertices))
        self._aoi = polygon.closed()
        self._draft.clear()
        self.regionChanged.emit(format_wkt(polygon))
        self.update()

    def _all_points(self) -> list[tuple[float, float]]:
        return [point for ring in (self._reference, self._secondary, self._aoi, tuple(self._draft)) for point in ring]

    def _view_bounds(self) -> tuple[float, float, float, float] | None:
        points = self._all_points()
        if not points:
            return None
        lons = [lon for lon, _lat in points]
        lats = [lat for _lon, lat in points]
        west, east = min(lons), max(lons)
        south, north = min(lats), max(lats)
        pad_lon = max((east - west) * 0.08, 0.01)
        pad_lat = max((north - south) * 0.08, 0.01)
        return west - pad_lon, south - pad_lat, east + pad_lon, north + pad_lat

    def _project(self, lon: float, lat: float, bounds: tuple[float, float, float, float]) -> QtCore.QPointF:
        west, south, east, north = bounds
        frame = self.rect().adjusted(16, 32, -16, -16)
        x = frame.left() + (lon - west) / (east - west) * frame.width()
        y = frame.top() + (north - lat) / (north - south) * frame.height()
        return QtCore.QPointF(x, y)

    def _geo_at(self, point: QtCore.QPointF) -> tuple[float, float] | None:
        bounds = self._view_bounds()
        if bounds is None:
            return None
        west, south, east, north = bounds
        frame = self.rect().adjusted(16, 32, -16, -16)
        if frame.width() <= 0 or frame.height() <= 0 or not frame.contains(point):
            return None
        lon = west + (point.x() - frame.left()) / frame.width() * (east - west)
        lat = north - (point.y() - frame.top()) / frame.height() * (north - south)
        return float(lon), float(lat)

    def _stroke(self, painter: QtGui.QPainter, ring, color: QtGui.QColor, bounds, *, fill: QtGui.QColor | None) -> None:
        if len(ring) < 2:
            return
        path = QtGui.QPainterPath()
        first = self._project(ring[0][0], ring[0][1], bounds)
        path.moveTo(first)
        for lon, lat in ring[1:]:
            path.lineTo(self._project(lon, lat, bounds))
        if fill is not None and len(ring) >= 3:
            painter.fillPath(path, fill)
        pen = QtGui.QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawPath(path)


class AoiDialog(QtWidgets.QDialog):
    """Type a WKT or a lat/lon box, or draw it on the two footprints."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        reference: tuple[tuple[float, float], ...] | None = None,
        secondary: tuple[tuple[float, float], ...] | None = None,
        burst_footprints: tuple[LonLatPolygon, ...] | None = None,
        wkt: str = "",
        subswath: str = "IW1",
        first_burst_number: int = 1,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Processing area")
        self.setObjectName("aoiDialog")
        self.resize(880, 640)
        self._reference = self._as_polygon(reference)
        self._secondary = self._as_polygon(secondary)
        self._bursts = tuple(burst_footprints or ())
        self._subswath = subswath
        self._first_burst_number = int(first_burst_number)
        self._report = None
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "The area is clipped to the reference/secondary overlap. "
            "A region that misses that overlap is refused. "
            "TOPS Split then keeps the bursts that cover the clip."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.map = FootprintMap()
        self.map.set_footprints(reference, secondary)
        self.map.regionChanged.connect(self._on_drawn)
        layout.addWidget(self.map, stretch=1)
        modes = QtWidgets.QHBoxLayout()
        self.rectangle = QtWidgets.QRadioButton("Rectangle")
        self.rectangle.setObjectName("aoiRectangle")
        self.polygon = QtWidgets.QRadioButton("Polygon")
        self.polygon.setObjectName("aoiPolygon")
        self.rectangle.setChecked(True)
        self.rectangle.toggled.connect(lambda checked: checked and self.map.set_mode("rectangle"))
        self.polygon.toggled.connect(lambda checked: checked and self.map.set_mode("polygon"))
        modes.addWidget(self.rectangle)
        modes.addWidget(self.polygon)
        modes.addStretch(1)
        layout.addLayout(modes)
        form = QtWidgets.QFormLayout()
        self.north = self._coord("aoiNorth")
        self.south = self._coord("aoiSouth")
        self.west = self._coord("aoiWest")
        self.east = self._coord("aoiEast")
        self.wkt = QtWidgets.QPlainTextEdit()
        self.wkt.setObjectName("aoiWkt")
        self.wkt.setFixedHeight(72)
        self.wkt.setPlaceholderText("POLYGON ((lon lat, lon lat, ...))")
        form.addRow("North latitude bound:", self.north)
        form.addRow("West longitude bound:", self.west)
        form.addRow("South latitude bound:", self.south)
        form.addRow("East longitude bound:", self.east)
        form.addRow("Geographic region (WKT):", self.wkt)
        layout.addLayout(form)
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("aoiStatus")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for box in (self.north, self.south, self.west, self.east):
            box.editingFinished.connect(self._on_bounds)
        self.wkt.textChanged.connect(self._on_wkt_typed)
        if wkt:
            self.wkt.setPlainText(wkt)
            self._apply_text(wkt)

    def report(self):
        return self._report

    def _coord(self, name: str) -> QtWidgets.QLineEdit:
        box = QtWidgets.QLineEdit()
        box.setObjectName(name)
        return box

    def _as_polygon(self, ring):
        if not ring or len(ring) < 3:
            return None
        try:
            return LonLatPolygon(tuple((float(lon), float(lat)) for lon, lat in ring))
        except AoiError:
            return None

    def _on_drawn(self, wkt: str) -> None:
        self._apply_text(wkt, from_map=True)

    def _on_bounds(self) -> None:
        try:
            polygon = polygon_from_bounds(
                float(self.north.text()),
                float(self.south.text()),
                float(self.west.text()),
                float(self.east.text()),
            )
        except (AoiError, ValueError):
            return
        self._apply_text(format_wkt(polygon), from_bounds=True)

    def _on_wkt_typed(self) -> None:
        text = self.wkt.toPlainText().strip()
        if not text.upper().startswith("POLYGON"):
            return
        self._apply_text(text, from_wkt=True)

    def _apply_text(self, text: str, *, from_map: bool = False, from_bounds: bool = False, from_wkt: bool = False) -> None:
        try:
            polygon = parse_wkt(text)
            report = prepare_aoi(
                polygon,
                reference=self._reference,
                secondary=self._secondary,
                burst_footprints=self._bursts or None,
                subswath=self._subswath,
                first_burst_number=self._first_burst_number,
            )
        except AoiError as exc:
            self.status.setText(str(exc))
            self._report = None
            return
        self._report = report
        self.status.setText(report.message)
        shown = report.clipped_wkt or report.requested_wkt
        if not from_wkt and self.wkt.toPlainText().strip() != shown:
            self.wkt.blockSignals(True)
            self.wkt.setPlainText(shown)
            self.wkt.blockSignals(False)
        if not from_bounds and report.north is not None:
            for box, value in (
                (self.north, report.north),
                (self.south, report.south),
                (self.west, report.west),
                (self.east, report.east),
            ):
                box.blockSignals(True)
                box.setText(f"{value:.8f}")
                box.blockSignals(False)
        if not from_map:
            try:
                self.map.set_aoi(parse_wkt(shown))
            except AoiError:
                self.map.set_aoi(polygon)
        self._fill_bounds_from(parse_wkt(shown) if shown else polygon)

    def _fill_bounds_from(self, polygon: LonLatPolygon) -> None:
        bounds = geographic_bounds(polygon)
        if self.north.text().strip():
            return
        for box, value in (
            (self.north, bounds.north),
            (self.south, bounds.south),
            (self.west, bounds.west),
            (self.east, bounds.east),
        ):
            box.setText(f"{value:.8f}")

    def _accept(self) -> None:
        if self._report is None:
            self._apply_text(self.wkt.toPlainText())
        if self._report is None:
            self.status.setText(self.status.text() or "Enter a WKT polygon or latitude/longitude bounds.")
            return
        if self._report.outside:
            self.status.setText(self._report.message)
            return
        self.accept()
