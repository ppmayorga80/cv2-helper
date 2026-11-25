"""Core image processing helper functions including resizing, borders, labels, and display."""

# pylint: disable=no-member,too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,line-too-long

from typing import Sequence, Any
import cv2
import numpy as np

from ._types import FitOption, KeyCode, ImgSpecListType, ImgSpecClass, PreloadImages
from ._utils import _get_quasi_square, _get_text_info


def resize(
        imgs: np.ndarray | Sequence[np.ndarray],
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


def add_text(
        imgs: ImgSpecListType
) -> list[np.ndarray]:
    """Draw text labels onto images based on specified positional and styling attributes.

    Args:
        imgs: Tuple or list of tuples formatted as (image, text) or (image, text, TAttr).

    Returns:
        List of labeled NumPy array images.
    """
    img_list = ImgSpecClass.normalize_to_list(imgs)
    shape = img_list[0].img.shape
    images = []
    for img_spec in img_list:
        img, lbl, opts = img_spec.img, img_spec.label, img_spec.ta

        if lbl:
            img = img.copy()
            info_org, info_font_face, info_font_scale, info_color, thickness, line_type = (
                _get_text_info(shape, lbl, opts)
            )
            cv2.putText(
                img,
                lbl,
                info_org,
                info_font_face,
                info_font_scale,
                info_color,
                thickness=thickness,
                lineType=line_type,
            )
            images.append(img)
        else:
            images.append(img)

    return images


def imcomp(
        img_or_list: ImgSpecListType,
        shape: tuple[int, int] | None = None,
        border: int = 0,
        border_color: int | tuple[int, int, int] = (255, 255, 255),
        fill_bg_color: int | tuple[int, int, int] = (255, 255, 0)
) -> np.ndarray:
    """The method shows images with labels in a quasi-grid filling spaces with black images

    Args:
        img_or_list: Single image array or sequence of image arrays.
        shape: the shape of the image to composite, if None, infer the optimal shape
        border: the border size to be drawn
        border_color: the border color
        fill_bg_color: the fill color used when scaling images and fill non defined region or for missing image tile

    Returns:
        A single image with all images arranged with shape and border information

    Examples:
        ```python
        import cv2
        from cv2_helper import imshow, TextAttr, TextPos
        A = cv2.imread("lion.jpg")
        B = cv2.imread("cat.jpg")
        imshow("CATS",imcomp([A,B]))
        imshow("CATS",imcomp([(A,"Lion"),B]))
        imshow("CATS",imcomp([(A,"Lion"),(B,"Cat")]))
        imshow("CATS",imcomp([(A,"Lion"),(B,"Cat",TextAttr(org=TextPos.TOP_CENTER))]))
        imshow("CATS",imcomp((A,B),border=1))
        key,im_grid = imshow("CATS",imcomp((A,B)), shape=(2,2),border=1, border_color=(0,255,255))
        ```
    """

    # 1. transform img_or_list to a list of tuples (ndarray,str,TextAttr).
    img_or_list = ImgSpecClass.normalize_to_list(img_or_list)

    # 2. ensure all images are of the same size
    first_shape = img_or_list[0].img.shape[0:2]
    if not all(x.img.shape[0:2] == first_shape for x in img_or_list):
        imgs = [x.img for x in img_or_list]
        imgs = resize(imgs, imgs[0], fx=1.0, fy=1.0, fill_color=fill_bg_color)
        img_or_list = [
            [img, x.label, x.ta]
            for img, x in zip(imgs, img_or_list)
        ]

    # 2.1 ensure all images are BGR
    for k in range(len(img_or_list)):
        img = img_or_list[k].img
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_or_list[k].img = img

    # 3. add labels
    imgs_with_labels = add_text(img_or_list)

    # 4. ensure we have borders
    if border > 0:
        border = int(border)
        if isinstance(border_color, int):
            border_color = (border_color, border_color, border_color)
        elif isinstance(border_color, (tuple, list)):
            if len(border_color) == 1:
                border_color = (border_color[0], border_color[0], border_color[0])
            elif len(border_color) == 3:
                border_color = tuple(border_color)
            else:
                raise ValueError("border_color should be a tuple|list of size 3 or a single integer")

        imgs_with_labels = add_borders(imgs_with_labels, border=border, border_color=border_color)

    if shape is not None:
        rows, cols = shape[:2]
    else:
        rows, cols = _get_quasi_square(len(imgs_with_labels))

    im0 = imgs_with_labels[0].imag
    bg_region = np.full(im0.shape, fill_value=fill_bg_color, dtype=im0.dtype)
    imgs_with_labels += [bg_region for _ in range(rows * cols - len(imgs_with_labels))]

    grid = [imgs_with_labels[i * cols: (i + 1) * cols] for i in range(rows)]
    full_grid_image = np.vstack([np.hstack(row) for row in grid])

    return full_grid_image


def wait_key(
        wait_time: int = 0,
        exit_key_codes: KeyCode | str | int | Sequence[Any] | None = KeyCode.ALL,
) -> int:
    """Wait for key presses within a specified delay or until targeted key codes are received.

    Args:
        wait_time: Delay in milliseconds for cv2.waitKeyEx. If 0, waits indefinitely.
        exit_key_codes: KeyCode constant, string(s), int keycode, sequence of codes, or KeyCode.ALL.

    Returns:
        The integer keycode of the pressed key, or -1 if no valid key was pressed or time elapsed.
    """
    result_key = -1

    if wait_time < 0:
        pass
    elif exit_key_codes is None:
        result_key = cv2.waitKeyEx(wait_time)
    else:
        valid_codes: set[int] = set()
        always_exit = KeyCode.parse_key_codes(exit_key_codes, valid_codes)

        if wait_time > 0:
            key = cv2.waitKeyEx(wait_time)
            if key != -1 and (always_exit or key in valid_codes):
                result_key = key
        else:
            while True:
                key = cv2.waitKeyEx(0)
                if key != -1 and (always_exit or key in valid_codes):
                    result_key = key
                    break

    return result_key


if __name__ == '__main__':
    A=PreloadImages.COLOR.CAT_01
    B=PreloadImages.COLOR.CAT_02
    C=PreloadImages.COLOR.CAT_03
    D=PreloadImages.GRAY.LENA
    E=PreloadImages.BINARY.HORSE

    G0 = imcomp([A,B,C,D,E], border=1, border_color=(255,255,0), fill_bg_color=(127,127,127))

    cv2.imshow("G0", G0)
    cv2.waitKey(0)