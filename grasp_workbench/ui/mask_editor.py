"""Frozen pixel-coordinate SAM3 prompts; segmentation remains in its model worker."""
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets


class MaskCanvas(QtWidgets.QLabel):
    def __init__(self, rgb, mask=None, parent=None):
        super().__init__(parent)
        self.rgb, self.mask = rgb.copy(), None if mask is None else mask.copy()
        self.boxes, self.points, self.mode, self.positive = [], [], "point", True
        self._start = None
        self.setMinimumSize(400, 250)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.redraw()

    def image_position(self, position):
        pixmap = self.pixmap()
        if pixmap is None:
            return None
        x = position.x() - (self.width() - pixmap.width()) / 2
        y = position.y() - (self.height() - pixmap.height()) / 2
        if 0 <= x < pixmap.width() and 0 <= y < pixmap.height():
            return [x * self.rgb.shape[1] / pixmap.width(), y * self.rgb.shape[0] / pixmap.height()]
        return None

    def mousePressEvent(self, event):
        point = self.image_position(event.position())
        if point is None:
            return
        positive = self.positive and event.button() != QtCore.Qt.MouseButton.RightButton
        if self.mode == "point":
            self.points.append({"xy": point, "label": int(positive)})
            self.redraw()
        else:
            self._start = point, positive

    def mouseReleaseEvent(self, event):
        end = self.image_position(event.position())
        if self.mode == "box" and self._start is not None and end is not None:
            start, positive = self._start
            low, high = np.minimum(start, end), np.maximum(start, end)
            if np.all(high > low):
                self.boxes.append({"xyxy": [*low.tolist(), *high.tolist()], "label": positive})
        self._start = None
        self.redraw()

    def redraw(self):
        rgb = self.rgb.copy()
        if self.mask is not None:
            rgb[self.mask] = (rgb[self.mask] * 0.6 + np.array([40, 220, 130]) * 0.4).astype(np.uint8)
        image = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format.Format_RGB888).copy()
        painter = QtGui.QPainter(image)
        for box in self.boxes:
            painter.setPen(QtGui.QPen(QtGui.QColor("lime" if box["label"] else "red"), 2))
            x1, y1, x2, y2 = box["xyxy"]
            painter.drawRect(QtCore.QRectF(x1, y1, x2 - x1, y2 - y1))
        for point in self.points:
            painter.setPen(QtGui.QPen(QtGui.QColor("lime" if point["label"] else "red"), 3))
            painter.drawEllipse(QtCore.QPointF(*point["xy"]), 3, 3)
        painter.end()
        self.setPixmap(QtGui.QPixmap.fromImage(image).scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                                           QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.redraw()


class MaskEditor(QtWidgets.QDialog):
    def __init__(self, rgb, mask=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("고정 RGB · SAM3 박스 / 포인트 보정")
        self.resize(850, 600)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("좌클릭 포함 / 우클릭 제외 · 박스는 드래그 · mask는 SAM3 worker가 다시 추론합니다"))
        row = QtWidgets.QHBoxLayout()
        mode = QtWidgets.QComboBox()
        mode.addItem("포인트", "point")
        mode.addItem("박스", "box")
        row.addWidget(mode)
        positive = QtWidgets.QCheckBox("포함 prompt")
        positive.setChecked(True)
        row.addWidget(positive)
        clear = QtWidgets.QPushButton("박스 / 포인트 지우기")
        row.addWidget(clear)
        layout.addLayout(row)
        self.canvas = MaskCanvas(rgb, mask)
        layout.addWidget(self.canvas, 1)
        mode.currentIndexChanged.connect(lambda _index: setattr(self.canvas, "mode", mode.currentData()))
        positive.toggled.connect(lambda value: setattr(self.canvas, "positive", value))
        clear.clicked.connect(self.clear_prompts)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("SAM3 보정 요청")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def clear_prompts(self):
        self.canvas.boxes.clear()
        self.canvas.points.clear()
        self.canvas.redraw()
