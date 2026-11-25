"""imshow.py defines the main functions imshow and wait_key that mimic cv2 versions."""
# pylint: disable=import-error,no-name-in-module

import atexit
from dataclasses import dataclass
import os
import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QMainWindow,
    QLabel,
)
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QColor,
    QPen,
    QFontDatabase,
)
from PySide6.QtCore import Qt, Signal, QRectF, QLineF, QEventLoop, QTimer

_MODULE_STATE = {"atexit_registered": False}


def _cleanup_qt_on_exit():
    """Prevents macOS Cocoa / cv2 C++ destructor deadlocks when Python exits."""
    app = QApplication.instance()
    if app is not None:
        app.closeAllWindows()
        app.processEvents()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _get_monospace_font(point_size: int = 11):
    """Returns the OS native fixed-width font (e.g. Menlo/SF Mono on macOS)."""
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    font.setPointSize(point_size)
    return font


@dataclass
class ViewerConfig:
    """Configuration parameters for ImageViewer overlays and appearance."""

    grid_tone: int | tuple[int, int] | None = None
    grid_size: int = 10
    rgb_size: int = 80
    rgb_font_size: int = 10
    show_grid: bool = True
    show_rgb_text: bool = True

    @property
    def tone_pair(self) -> tuple[int, int]:
        """Returns (tone_for_light_pixels, tone_for_dark_pixels) clamped to [0, 255]."""
        tone = 64 if self.grid_tone is None else self.grid_tone
        first, second = (tone, 255 - tone) if isinstance(tone, int) else tone
        light_tone = int(np.clip(min(first, second), 0, 255))
        dark_tone = int(np.clip(max(first, second), 0, 255))
        return light_tone, dark_tone


class ImageViewer(QGraphicsView):  # pylint: disable=too-many-instance-attributes
    """Interactive graphics view for displaying and inspecting image pixels."""

    # pylint: disable=invalid-name
    pixelClicked = Signal(int, int, tuple)
    pixelHovered = Signal(int, int, tuple)
    zoomChanged = Signal(float)
    windowClosed = Signal()
    keyPressed = Signal(int)
    # pylint: enable=invalid-name

    MAX_PIXEL_SIZE = 200.0
    ZOOM_STEP = 1.08

    def __init__(self, image: np.ndarray, config: ViewerConfig | None = None):
        """Initializes the image viewer with grid and RGB overlay settings."""
        super().__init__()
        self.config = config or ViewerConfig()
        self.image_array = self._normalize_image(image)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self._update_pixmap()

        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self._press_pos = None

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Converts input NumPy array into a contiguous uint8 3-channel RGB array."""
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy.ndarray")

        img = image.copy()
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating) and img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        return np.ascontiguousarray(img)

    def _update_pixmap(self):
        """Updates the scene pixmap and bounding rectangle from the image array."""
        height, width, channels = self.image_array.shape
        qimg = QImage(
            self.image_array.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))
        self.setSceneRect(QRectF(0, 0, width, height))

    def set_image(self, image: np.ndarray):
        """Replaces the displayed image and refreshes the viewport."""
        self.image_array = self._normalize_image(image)
        self._update_pixmap()
        self.viewport().update()

    @property
    def current_zoom(self) -> float:
        """Returns the current horizontal zoom scale factor."""
        return self.transform().m11()

    def wheelEvent(self, event):  # pylint: disable=invalid-name
        """Handles smooth mouse wheel and trackpad zooming anchored under the cursor."""
        delta = event.angleDelta().y()
        if delta == 0:
            return

        current_scale = self.current_zoom
        height, width, _ = self.image_array.shape
        min_scale = max(1.0 / max(1, width), 1.0 / max(1, height))
        raw_target = current_scale * (self.ZOOM_STEP ** (delta / 120.0))

        if delta > 0:
            if current_scale >= self.MAX_PIXEL_SIZE:
                return
            target_scale = min(self.MAX_PIXEL_SIZE, raw_target)
        else:
            if current_scale <= min_scale:
                return
            target_scale = max(min_scale, raw_target)

        factor = target_scale / current_scale
        self.scale(factor, factor)
        self.zoomChanged.emit(self.current_zoom)
        self._emit_hover_at(event.position().toPoint())

    def _emit_hover_at(self, viewport_pos):
        """Emits the pixel coordinates and RGB value under the given viewport position."""
        scene_pos = self.mapToScene(viewport_pos)
        x, y = int(np.floor(scene_pos.x())), int(np.floor(scene_pos.y()))
        height, width, _ = self.image_array.shape
        if 0 <= x < width and 0 <= y < height:
            rgb = tuple(int(v) for v in self.image_array[y, x])
            self.pixelHovered.emit(x, y, rgb)

    def mouseMoveEvent(self, event):  # pylint: disable=invalid-name
        """Tracks cursor movement to emit hovered pixel information."""
        super().mouseMoveEvent(event)
        self._emit_hover_at(event.pos())

    def mousePressEvent(self, event):  # pylint: disable=invalid-name
        """Records the initial press position to distinguish clicks from drags."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):  # pylint: disable=invalid-name
        """Emits pixelClicked if the left mouse button was released without dragging."""
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton and self._press_pos:
            if (event.pos() - self._press_pos).manhattanLength() < 4:
                scene_pos = self.mapToScene(event.pos())
                x, y = int(np.floor(scene_pos.x())), int(np.floor(scene_pos.y()))
                height, width, _ = self.image_array.shape
                if 0 <= x < width and 0 <= y < height:
                    rgb = tuple(int(v) for v in self.image_array[y, x])
                    self.pixelClicked.emit(x, y, rgb)
        self._press_pos = None

    def _draw_grid(self, painter: QPainter, grid_data: tuple):
        """Draws shared 1px grid lines with adaptive luminance contrast."""
        xs, ys, is_light, _ = grid_data
        rows_vis, cols_vis = is_light.shape
        lines: list[list[QLineF]] = [[], []]  # [0]=dark lines, [1]=light lines

        for col in range(cols_vis + 1):
            ref_c = min(col, cols_vis - 1)
            for row in range(rows_vis):
                lines[0 if is_light[row, ref_c] else 1].append(
                    QLineF(xs[col], ys[row], xs[col], ys[row + 1])
                )

        for row in range(rows_vis + 1):
            ref_r = min(row, rows_vis - 1)
            for col in range(cols_vis):
                lines[0 if is_light[ref_r, col] else 1].append(
                    QLineF(xs[col], ys[row], xs[col + 1], ys[row])
                )

        for tone, segment_list in zip(self.config.tone_pair, lines):
            if segment_list:
                painter.setPen(QPen(QColor(tone, tone, tone), 1))
                painter.drawLines(segment_list)

    def _draw_rgb_text(self, painter: QPainter, grid_data: tuple):
        """Renders centered R/G/B text values inside each visible pixel cell."""
        xs, ys, is_light, (top, left) = grid_data
        painter.setFont(_get_monospace_font(self.config.rgb_font_size))
        colors = (QColor(0, 0, 0), QColor(255, 255, 255))

        for row in range(is_light.shape[0]):
            for col in range(is_light.shape[1]):
                painter.setPen(colors[0 if is_light[row, col] else 1])
                rgb = self.image_array[top + row, left + col]
                rect = QRectF(
                    xs[col], ys[row], xs[col + 1] - xs[col], ys[row + 1] - ys[row]
                )
                painter.drawText(
                    rect,
                    Qt.AlignmentFlag.AlignCenter,
                    f"R={rgb[0]:>3}\nG={rgb[1]:>3}\nB={rgb[2]:>3}",
                )

    def _compute_grid_data(self) -> tuple | None:
        """Computes visible screen coordinates and luminance mask for the viewport."""
        height, width, _ = self.image_array.shape
        vis = self.mapToScene(self.viewport().rect()).boundingRect()
        left = max(0, int(np.floor(vis.left())))
        right = min(width, int(np.ceil(vis.right())))
        top = max(0, int(np.floor(vis.top())))
        bottom = min(height, int(np.ceil(vis.bottom())))
        if left >= right or top >= bottom:
            return None

        patch = self.image_array[top:bottom, left:right].astype(np.float32)
        is_light = (
                           0.299 * patch[:, :, 0]
                           + 0.587 * patch[:, :, 1]
                           + 0.114 * patch[:, :, 2]
                   ) > 128.0
        return (
            [self.mapFromScene(x, top).x() for x in range(left, right + 1)],
            [self.mapFromScene(left, y).y() for y in range(top, bottom + 1)],
            is_light,
            (top, left),
        )

    def paintEvent(self, event):  # pylint: disable=invalid-name
        """Renders the image along with optional pixel grid lines and RGB text overlays."""
        super().paintEvent(event)

        draw_grid = self.config.show_grid and (self.current_zoom > self.config.grid_size)
        draw_rgb = self.config.show_rgb_text and (self.current_zoom > self.config.rgb_size)
        if not draw_grid and not draw_rgb:
            return

        grid_data = self._compute_grid_data()
        if grid_data is None:
            return

        painter = QPainter(self.viewport())
        if draw_grid:
            self._draw_grid(painter, grid_data)
        if draw_rgb:
            self._draw_rgb_text(painter, grid_data)
        painter.end()


class ImshowWindow(QMainWindow):  # pylint: disable=too-many-instance-attributes
    """Main window hosting the ImageViewer and live pixel status bar."""

    def __init__(
            self,
            window_title: str,
            image: np.ndarray,
            config: ViewerConfig | None = None,
    ):
        """Initializes the inspector window, status bar, and viewer signals."""
        super().__init__()
        self.setWindowTitle(window_title)
        self.viewer = ImageViewer(image, config)
        self.setCentralWidget(self.viewer)
        self.resize(800, 600)

        self._last_x = 0
        self._last_y = 0
        self._last_rgb = tuple(int(v) for v in self.viewer.image_array[0, 0])

        self.status_label = QLabel()
        self.status_label.setFont(_get_monospace_font(11))
        self.statusBar().addWidget(self.status_label)
        self._refresh_status_bar()

        self.viewer.pixelHovered.connect(self._on_pixel_hovered)
        self.viewer.zoomChanged.connect(self._on_zoom_changed)

        # pylint: disable=invalid-name
        self.pixelClicked = self.viewer.pixelClicked
        self.windowClosed = self.viewer.windowClosed
        self.keyPressed = self.viewer.keyPressed
        # pylint: enable=invalid-name
        self.is_closed = False

    def _refresh_status_bar(self):
        """Updates the status bar text with aligned coordinates, RGB, hex, and zoom."""
        rows, cols, _ = self.viewer.image_array.shape
        coord_w = max(len(str(x)) for x in (rows, cols))
        red, green, blue = self._last_rgb
        hex_code = f"#{red:02X}{green:02X}{blue:02X}"

        self.status_label.setText(
            f"(x,y)=({self._last_x:>{coord_w}},{self._last_y:>{coord_w}})   "
            f"(RGB)=({red:>3},{green:>3},{blue:>3})={hex_code}   "
            f"Zoom={self.viewer.current_zoom:0.2f}"
        )

    def _on_pixel_hovered(self, x: int, y: int, rgb: tuple):
        """Updates cached hover coordinates and refreshes the status bar."""
        self._last_x = x
        self._last_y = y
        self._last_rgb = rgb
        self._refresh_status_bar()

    def _on_zoom_changed(self, _zoom: float):
        """Refreshes the status bar when the viewer zoom level changes."""
        self._refresh_status_bar()

    def update_image(self, image: np.ndarray):
        """Updates the displayed image and clamps the status bar coordinates."""
        self.viewer.set_image(image)
        height, width, _ = self.viewer.image_array.shape
        self._last_x = min(self._last_x, width - 1)
        self._last_y = min(self._last_y, height - 1)
        self._last_rgb = tuple(
            int(v) for v in self.viewer.image_array[self._last_y, self._last_x]
        )
        self._refresh_status_bar()

    def keyPressEvent(self, event):  # pylint: disable=invalid-name
        """Handles grid/text toggle shortcuts and emits non-modifier key presses."""
        key = event.key()
        if key == Qt.Key.Key_G:
            self.viewer.config.show_grid = not self.viewer.config.show_grid
            self.viewer.viewport().update()
            super().keyPressEvent(event)
            return
        if key == Qt.Key.Key_T:
            self.viewer.config.show_rgb_text = not self.viewer.config.show_rgb_text
            self.viewer.viewport().update()
            super().keyPressEvent(event)
            return

        if key not in (
                Qt.Key.Key_Shift,
                Qt.Key.Key_Control,
                Qt.Key.Key_Alt,
                Qt.Key.Key_Meta,
        ):
            self.keyPressed.emit(key)
        super().keyPressEvent(event)

    def closeEvent(self, event):  # pylint: disable=invalid-name
        """Marks the window as closed and emits the windowClosed signal once."""
        if not self.is_closed:
            self.is_closed = True
            self.windowClosed.emit()
        super().closeEvent(event)


def imshow(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        window_title: str,
        image: np.ndarray,
        grid_tone: int | tuple[int, int] = 127,
        show_grid_when_px_gt: int = 10,
        show_rgb_text_when_px_gt: int = 80,
        rgb_text_font_size: int = 10,
        show_grid: bool = True,
        show_rgb_text: bool = True,
) -> ImshowWindow:
    """Displays an image in an interactive inspection window with zoom and pixel overlays."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(True)

    if not _MODULE_STATE["atexit_registered"]:
        atexit.register(_cleanup_qt_on_exit)
        _MODULE_STATE["atexit_registered"] = True

    config = ViewerConfig(
        grid_tone=grid_tone,
        grid_size=show_grid_when_px_gt,
        rgb_size=show_rgb_text_when_px_gt,
        rgb_font_size=rgb_text_font_size,
        show_grid=show_grid,
        show_rgb_text=show_rgb_text,
    )
    window = ImshowWindow(window_title, image, config)
    window.show()
    window.raise_()
    window.activateWindow()
    return window


def wait_key(win: ImshowWindow, seconds: float = 0) -> int:
    """Blocks until a key is pressed in the window, timeout elapses, or window closes."""
    if win is None or win.is_closed:
        return -1

    app = QApplication.instance()
    if app is None:
        return -1

    loop = QEventLoop()
    result = {"key": -1}

    def finish(key_code: int = -1):
        """Records the pressed key and exits the local event loop."""
        if result["key"] == -1 and key_code != -1:
            result["key"] = key_code
        if loop.isRunning():
            loop.exit(0)

    def on_key(key_code: int):
        """Handles keyPressed signal by finishing the wait loop with the key code."""
        finish(key_code)

    def on_close():
        """Handles windowClosed or timeout events by exiting the wait loop."""
        finish(-1)

    win.keyPressed.connect(on_key)
    win.windowClosed.connect(on_close)

    timer = None
    if seconds and seconds > 0:
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(on_close)
        timer.start(int(seconds * 1000))

    loop.exec()

    if timer is not None:
        timer.stop()
    try:
        win.keyPressed.disconnect(on_key)
        win.windowClosed.disconnect(on_close)
    except RuntimeError:
        pass

    if not win.is_closed:
        win.close()

    app.processEvents()
    return result["key"]
