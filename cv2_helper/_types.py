"""Type definitions and enumerations for cv2_helper."""

# pylint: disable=no-member,too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,too-few-public-methods,invalid-name

from enum import Enum
import cv2


class FitOption(Enum):
    """Image resizing strategy options."""

    FIT_FULL = 0  # Fill entire canvas, stretching image if necessary
    FIT_AUTO = 1  # Automatically adjust scale to preserve aspect ratio and pad empty space


class TextPos(Enum):
    """Enumeration for text overlay positioning relative to image dimensions."""

    TOP_LEFT = 0
    TOP_RIGHT = 1
    TOP_CENTER = 2
    BOTTOM_LEFT = 3
    BOTTOM_RIGHT = 4
    BOTTOM_CENTER = 5
    CENTER_LEFT = 6
    CENTER_RIGHT = 7
    CENTER = 8


class TextAttr:
    """Text attributes and styling parameters with dynamic runtime default support."""

    org: tuple[int, int] | TextPos = TextPos.TOP_LEFT
    fontFace: int = cv2.FONT_HERSHEY_PLAIN
    fontScale: float = 1.0
    color: tuple[int, int, int] = (0, 0, 255)
    thickness: int = 1
    lineType: int = cv2.LINE_AA
    margin: int = 8
    margin_x: int = 0
    margin_y: int = 0

    def __init__(
            self,
            org: tuple[int, int] | TextPos | None = None,
            font_face: int | None = None,
            font_scale: float | None = None,
            color: tuple[int, int, int] | None = None,
            thickness: int | None = None,
            line_type: int | None = None,
            margin: int | None = None,
            margin_x: int | None = None,
            margin_y: int | None = None,
    ):
        """Initialize TextAttr instance with default or custom overrides."""
        self.org = TextAttr.org if org is None else org
        self.fontFace = TextAttr.fontFace if font_face is None else font_face
        self.fontScale = TextAttr.fontScale if font_scale is None else font_scale
        self.color = TextAttr.color if color is None else color
        self.thickness = TextAttr.thickness if thickness is None else thickness
        self.lineType = TextAttr.lineType if line_type is None else line_type

        self.margin = TextAttr.margin if margin is None else margin
        self.margin_x = self.margin if margin_x is None else margin_x
        self.margin_y = self.margin if margin_y is None else margin_y
