"""Utility functions for image grid calculation, random string generation, and text placement."""
# pylint: disable=no-member

from math import sqrt
import secrets
from typing import Tuple

import cv2

from ._types import TextAttr, TextPos


def _get_quasi_square(n: int) -> Tuple[int, int]:
    """Calculate row and column dimensions to arrange n items into a quasi-square grid.

    Args:
        n: Total number of items in the grid.

    Returns:
        Tuple containing (rows, cols).
    """
    if n < 1:
        return 0, 0
    if n == 1:
        return 1, 1
    if n == 2:
        return 1, 2

    rows, cols = 1, n
    while not 0.6 < float(rows) / cols <= 1:
        for k in range(int(sqrt(n)), 0, -1):
            if n % k == 0:
                rows, cols = k, n // k
                break
        n += 1
    return rows, cols


def _get_random_title() -> str:
    """Generate a random 16-character hexadecimal string.

    Returns:
        Random title string.
    """
    return secrets.token_hex(8)


def _get_text_info(
        shape: Tuple[int, ...],
        text: str,
        ta: TextAttr,
) -> Tuple[Tuple[int, int], int, float, Tuple[int, int, int], int, int]:
    """Calculate calculated origin coordinates and text attributes for rendering.

    Args:
        shape: Target image shape dimensions (height, width, ...).
        text: Text string to be rendered.
        ta: Text attribute configuration object.

    Returns:
        Tuple containing (origin, font_face, font_scale, color, thickness, line_type).
    """
    if isinstance(ta.org, TextPos):
        (text_w, text_h), baseline = cv2.getTextSize(text, ta.fontFace, ta.fontScale, ta.thickness)
        mx, my = ta.margin_x, ta.margin_y
        ks = 0.81

        xl = mx
        xc = (shape[1] - text_w) // 2
        xr = shape[1] - text_w - mx

        yt = my + int(text_h * ks)
        yc = (shape[0] + int(text_h * ks) - baseline) // 2
        yb = shape[0] - my - baseline

        if ta.org == TextPos.TOP_LEFT:
            ta.org = (xl, yt)
        elif ta.org == TextPos.TOP_CENTER:
            ta.org = (xc, yt)
        elif ta.org == TextPos.TOP_RIGHT:
            ta.org = (xr, yt)
        elif ta.org == TextPos.CENTER_LEFT:
            ta.org = (xl, yc)
        elif ta.org == TextPos.CENTER:
            ta.org = (xc, yc)
        elif ta.org == TextPos.CENTER_RIGHT:
            ta.org = (xr, yc)
        elif ta.org == TextPos.BOTTOM_LEFT:
            ta.org = (xl, yb)
        elif ta.org == TextPos.BOTTOM_CENTER:
            ta.org = (xc, yb)
        elif ta.org == TextPos.BOTTOM_RIGHT:
            ta.org = (xr, yb)
        else:
            ta.org = (xl, yt)

    return ta.org, ta.fontFace, ta.fontScale, ta.color, ta.thickness, ta.lineType
