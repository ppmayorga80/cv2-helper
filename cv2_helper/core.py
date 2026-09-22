"""Core image processing helper functions including resizing, borders, labels, and display."""

# pylint: disable=no-member,too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches

from typing import Sequence
import cv2
import numpy as np

from ._types import FitOption, TextAttr
from ._utils import _get_quasi_square, _get_text_info


def resize(
        imgs: Sequence[np.ndarray],
        dsize: tuple[int, int] | np.ndarray | None = None,
        fx: float = 0.0,
        fy: float = 0.0,
        interpolation: int = cv2.INTER_LINEAR,
        fit_option: FitOption = FitOption.FIT_AUTO,
        fill_color: tuple[int, int, int] | int = (0, 0, 0),
) -> list[np.ndarray]:
    """Resize a sequence of images to a uniform target dimension.

    Args:
        imgs: Sequence of input images as NumPy arrays.
        dsize: Target size as (width, height) or a reference image array.
        fx: Scale factor along the horizontal axis.
        fy: Scale factor along the vertical axis.
        interpolation: OpenCV interpolation algorithm flag.
        fit_option: Resizing strategy (FIT_AUTO maintains aspect ratio with
            canvas padding, FIT_FULL stretches).
        fill_color: Background padding color for FIT_AUTO (BGR tuple or scalar).

    Returns:
        List of resized NumPy array images.
    """
    if not imgs:
        return []

    if dsize is None and fx == 0 and fy == 0:
        dsize = imgs[0]

    if fx > 0 and fy == 0:
        fy = fx
    elif fy > 0 and fx == 0:
        fx = fy

    common_w, common_h = None, None

    if isinstance(dsize, np.ndarray):
        common_w, common_h = dsize.shape[1], dsize.shape[0]
    elif isinstance(dsize, (tuple, list)) and len(dsize) >= 2 and dsize[0] > 0 and dsize[1] > 0:
        common_w, common_h = int(dsize[0]), int(dsize[1])

    if common_w is not None and common_h is not None:
        if fx > 0:
            common_w = int(round(common_w * fx))
        if fy > 0:
            common_h = int(round(common_h * fy))

    resized_imgs = []

    for img in imgs:
        h, w = img.shape[:2]

        if common_w is not None and common_h is not None:
            target_w, target_h = common_w, common_h
        elif fx > 0 and fy > 0:
            target_w, target_h = int(round(w * fx)), int(round(h * fy))
        else:
            raise ValueError("Set dsize or scaling factors fx, fy.")

        if fit_option == FitOption.FIT_FULL:
            resized = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
            resized_imgs.append(resized)

        elif fit_option == FitOption.FIT_AUTO:
            scale = min(target_w / w, target_h / h)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))

            scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

            if img.ndim == 2:
                color = fill_color[0] if isinstance(fill_color, (tuple, list)) else fill_color
                canvas = np.full((target_h, target_w), color, dtype=img.dtype)
            else:
                canvas = np.full((target_h, target_w, 3), fill_color, dtype=img.dtype)

            top = (target_h - new_h) // 2
            left = (target_w - new_w) // 2

            canvas[top: top + new_h, left: left + new_w] = scaled_img
            resized_imgs.append(canvas)

    return resized_imgs


def add_borders(
        img: np.ndarray | Sequence[np.ndarray],
        border: int = 1,
        border_color: int | tuple[int, int, int] = (255, 255, 255),
) -> list[np.ndarray]:
    """Add solid borders around an image or sequence of images.

    Args:
        img: Single image array or sequence of image arrays.
        border: Border thickness in pixels.
        border_color: Border color as a BGR tuple or scalar value.

    Returns:
        List of NumPy array images with applied borders.
    """
    if isinstance(img, (list, tuple)):
        img_list = list(img)
    else:
        img_list = [img]

    img_list = [x.copy() for x in img_list]
    for target_img in img_list:
        b_width = min(border, min(target_img.shape[:2]) // 2)
        target_img[:, 0:b_width] = border_color
        target_img[:, -b_width:] = border_color
        target_img[0:b_width, :] = border_color
        target_img[-b_width:, :] = border_color

    return img_list


def add_labels(
        imgs: tuple[np.ndarray, str] | tuple[np.ndarray, str, TextAttr] | list[tuple]
) -> list[np.ndarray]:
    """Draw text labels onto images based on specified positional and styling attributes.

    Args:
        imgs: Tuple or list of tuples formatted as (image, text) or (image, text, TAttr).

    Returns:
        List of labeled NumPy array images.
    """
    if isinstance(imgs, tuple):
        if len(imgs) == 1:
            img_list = [(imgs[0], "", TextAttr())]
        elif len(imgs) == 2:
            img_list = [(imgs[0], imgs[1], TextAttr())]
        elif len(imgs) == 3:
            img_list = [imgs]
        else:
            raise ValueError(f"Invalid tuple parameters count: {len(imgs)}")
    elif isinstance(imgs, list):
        img_list = [
            (x, "", TextAttr())
            if isinstance(x, np.ndarray)
            else (
                (x[0], "", TextAttr())
                if isinstance(x, (list, tuple)) and len(x) == 1
                else (
                    (x[0], x[1], TextAttr())
                    if isinstance(x, (list, tuple)) and len(x) == 2
                    else (x[0], x[1], x[2])
                )
            )
            for x in imgs
        ]
    else:
        raise ValueError("Invalid parameters structure provided for labels.")

    shape = img_list[0][0].shape
    images = []
    for mat, lbl, opts in img_list:
        if lbl:
            mat = mat.copy()
            info_org, info_font_face, info_font_scale, info_color, thickness, line_type = (
                _get_text_info(shape, lbl, opts)
            )
            cv2.putText(
                mat,
                lbl,
                info_org,
                info_font_face,
                info_font_scale,
                info_color,
                thickness=thickness,
                lineType=line_type,
            )
            images.append(mat)
        else:
            images.append(mat)

    return images


def imshow(
        title: str,
        img: np.ndarray | Sequence[np.ndarray],
        wait_time: int = 0,
) -> int:
    """Display an image or a sequence of images formatted as a grid in an OpenCV window.

    Args:
        title: Window title string.
        img: Single image array or sequence of image arrays.
        wait_time: Delay in milliseconds for cv2.waitKey (0 waits indefinitely).
    Returns:
        int: the key identifier or -1
    """
    if isinstance(img, np.ndarray):
        cv2.imshow(title, img)
        return cv2.waitKey(wait_time)

    if isinstance(img, (list, tuple)):
        img_list = list(img)
        rows, cols = _get_quasi_square(len(img_list))

        img_list += [np.zeros_like(img_list[-1]) for _ in range(rows * cols - len(img_list))]

        grid = [img_list[i * cols: (i + 1) * cols] for i in range(rows)]
        full_grid_image = np.vstack([np.hstack(row) for row in grid])

        cv2.imshow(title, full_grid_image)
        return cv2.waitKey(wait_time)

    return -1
