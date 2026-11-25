"""Type definitions and enumerations for cv2_helper."""
# pylint: disable=no-member,too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,too-few-public-methods,invalid-name

from enum import Enum
from typing import Any, Sequence, TypeAlias
from pathlib import Path
import cv2
import numpy as np

# Get assets path
ASSETS_DIR = Path(__file__).parent / "assets"


class FitOption(Enum):
    """Image resizing strategy options."""

    FIT_FULL = 0  # Fill entire canvas, stretching image if necessary
    FIT_AUTO = 1  # Automatically adjust scale to preserve aspect ratio and pad empty space


class Image2DType(Enum):
    """Image assets type."""
    AUTO = 0
    COLOR = 1
    GRAY = 2
    BINARY = 3


class _LazyImage2D:
    """Lazy image, just load images the 1st time they are accessed."""

    def __init__(self, asset_rel_path: str, convert_to: Image2DType = Image2DType.AUTO):
        # constructor
        self.path = ASSETS_DIR / asset_rel_path
        self.image_type = Image2DType.AUTO
        self.convert_to = convert_to
        self.cached_image = None

    def __get__(self, instance, owner):
        # accessor
        if self.cached_image is None:
            self.cached_image = cv2.imread(str(self.path))
            ndim = len(self.cached_image.shape)

            if ndim == 2:
                self.image_type = Image2DType.GRAY
            elif ndim == 3:
                self.image_type = Image2DType.COLOR
            else:
                raise ValueError(f"Unsupported image dimension: {ndim} from shape={self.cached_image.shape}")

            if self.convert_to == Image2DType.AUTO:
                pass

            if self.convert_to == Image2DType.COLOR:
                if self.image_type == Image2DType.COLOR:
                    pass
                if self.image_type == Image2DType.GRAY:
                    self.cached_image = cv2.cvtColor(self.cached_image, cv2.COLOR_GRAY2BGR)
            elif self.convert_to == Image2DType.GRAY:
                if self.image_type == Image2DType.COLOR:
                    self.cached_image = cv2.cvtColor(self.cached_image, cv2.COLOR_BGR2GRAY)
                elif self.image_type == Image2DType.GRAY:
                    pass
            elif self.convert_to == Image2DType.BINARY:
                if self.image_type == Image2DType.COLOR:
                    self.cached_image = cv2.cvtColor(self.cached_image, cv2.COLOR_BGR2GRAY)
                    _, self.cached_image = cv2.threshold(self.cached_image,
                                                         127,
                                                         255,
                                                         cv2.THRESH_BINARY)
                elif self.image_type.value == Image2DType.GRAY:
                    _, self.cached_image = cv2.threshold(self.cached_image,
                                                         127,
                                                         255,
                                                         cv2.THRESH_BINARY)

        return self.cached_image


class PreloadImages:
    """Preloaded image assets grouped by color space."""

    class COLOR:
        """Color space preloading options."""
        LION = _LazyImage2D("color/lion.png")
        LENA = _LazyImage2D("color/lena.png")
        CAT_01 = _LazyImage2D("color/cat-01.png")
        CAT_02 = _LazyImage2D("color/cat-02.png")
        CAT_03 = _LazyImage2D("color/cat-03.png")

    class GRAY:
        """Gray space preloading options."""
        LION = _LazyImage2D("color/lion.png", convert_to=Image2DType.GRAY)
        LENA = _LazyImage2D("color/lena.png", convert_to=Image2DType.GRAY)
        CAT_01 = _LazyImage2D("color/cat-01.png", convert_to=Image2DType.GRAY)
        CAT_02 = _LazyImage2D("color/cat-02.png", convert_to=Image2DType.GRAY)
        CAT_03 = _LazyImage2D("color/cat-03.png", convert_to=Image2DType.GRAY)

    class BINARY:
        """Binary space preloading options."""
        ANT = _LazyImage2D("binary/ant.png", convert_to=Image2DType.BINARY)
        BAT = _LazyImage2D("binary/bat.png", convert_to=Image2DType.BINARY)
        HORSE = _LazyImage2D("binary/horse.png", convert_to=Image2DType.BINARY)
        PERSON = _LazyImage2D("binary/person.png", convert_to=Image2DType.BINARY)


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

    def __str__(self):
        return f"TextAttr(org={self.org},fontFace={self.fontFace},fontScale={self.fontScale},color={self.color},thickness={self.thickness},lineType={self.lineType},margin={self.margin},margin_x={self.margin_x},margin_y={self.margin_y})"

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


# A single image specifier:
# - np.ndarray
# - 2-element tuple/list: (img, "title") or [img, "title"]
# - 3-element tuple/list: (img, "title", attr) or [img, "title", attr]
# Declare explicit TypeAlias annotations
ImgSpecType: TypeAlias = (
        np.ndarray
        | tuple[np.ndarray, str]
        | tuple[np.ndarray, str, TextPos]
        | tuple[np.ndarray, str, TextAttr]
        | list
)
ImgSpecListType: TypeAlias = ImgSpecType | list[ImgSpecType] | tuple[ImgSpecType]


class ImgSpecClass:
    img: np.ndarray = None
    label: str = ""
    ta: TextAttr = None

    def __str__(self):
        return f"np.ndarray[{self.img.shape}], '{self.label}', {self.ta}"

    def __init__(self, img_w_spec: ImgSpecType):
        if isinstance(img_w_spec, np.ndarray):
            self.img = img_w_spec
            self.label = ""
            self.ta = TextAttr()
        elif isinstance(img_w_spec, (tuple, list)):
            if len(img_w_spec) == 1:
                self.img = img_w_spec[0]
                self.label = ""
                self.ta = TextAttr()
            elif len(img_w_spec) == 2:
                self.img = img_w_spec[0]
                self.label = img_w_spec[1]
                self.ta = TextAttr()
            elif len(img_w_spec) == 3:
                self.img = img_w_spec[0]
                self.label = img_w_spec[1]
                if isinstance(img_w_spec[2], TextPos):
                    self.ta = TextAttr(org=img_w_spec[2])
                else:
                    self.ta = img_w_spec[2]

        if not all([
            isinstance(self.img, np.ndarray),
            isinstance(self.label, str),
            isinstance(self.ta, TextAttr)]):
            raise TypeError(f"img_w_spec must be of type {ImgSpecType}")

    @classmethod
    def normalize_to_list(cls, img_or_list: ImgSpecListType) -> list[ImgSpecClass]:
        """Given an image or list specification
        this function normalizes it to a list[(image,str,TextAttr)] s.t. if no text or
        attributes are given, an empty/default values are used

        Args:
            img_or_list: Image or list [image|(image,str)|(image,str,TextAttr)]

        Returns:
            The normalized list of image sequences where each element
            has length 3 (image, str, TextAttr).

        ---
        Examples:
            ```python
                A=cv2.imread('./test.jpg')
                X=ImgSpec.normalize_to_list(A)
                Y=ImgSpec.normalize_to_list([(A,"HELLO")])
                Z=ImgSpec.normalize_to_list([(A,"HELLO"),(A,"WORLD"), A])
                # X = [[A,"",TextAttr()]]
                # Y = [[A,"HELLO",TextAttr()]]
                # Z = [[A,"HELLO",TextAttr()],[A,"WORLD",TextAttr()], [A,"",TextAttr()]]
            ```
        """
        if isinstance(img_or_list, np.ndarray):
            return [cls(img_or_list)]
        elif isinstance(img_or_list, (tuple, list)):
            try:
                object0 = cls(img_or_list)
                return [object0]
            except TypeError:
                pass
            return [cls(x) for x in img_or_list]
        raise TypeError(f"image_or_list must be of type Sequence[{ImgSpecType}]")


class KeyCode:
    """Cross-platform OpenCV key code sets for cv2.waitKeyEx()."""

    ALL = 0

    ESC = (27,)
    QUIT = (ord("q"), ord("Q"))

    # (Linux GTK, Windows Win32, macOS Cocoa, X11 fallback)
    LEFT = (65361, 2424832, 63234)
    UP = (65362, 2490368, 63232)
    RIGHT = (65363, 2555904, 63235)
    DOWN = (65364, 2621440, 63233)

    @classmethod
    def parse_key_codes(cls, codes: Any, valid_codes: set[int]) -> bool:
        """Extract integer keycodes from ints, strings, KeyCode attributes, or sequences.

        Args:
            codes: Key code, string, sequence, or KeyCode representation.
            valid_codes: Set into which resolved integer keycodes are collected.

        Returns:
            True if KeyCode.ALL is present, False otherwise.
        """
        if (codes == cls.ALL or
                getattr(codes, "value", None) == "ALL" or
                getattr(codes, "name", None) == "ALL"):
            return True

        has_all = False

        if isinstance(codes, str):
            for char in codes:
                valid_codes.add(ord(char))
        elif isinstance(codes, int):
            valid_codes.add(codes)
        elif isinstance(codes, (list, tuple, set)):
            for item in codes:
                if cls.parse_key_codes(item, valid_codes):
                    has_all = True
                    break
        elif hasattr(codes, "value"):
            has_all = cls.parse_key_codes(codes.value, valid_codes)

        return has_all


if __name__ == '__main__':
    A = PreloadImages.COLOR.CAT_01
    print(ImgSpecClass(A))
    print(ImgSpecClass((A, "Hello")))
    print(ImgSpecClass((A, "Hello", TextPos.CENTER)))
    print(ImgSpecClass((A, "Hello", TextAttr(color=(0, 255, 255)))))

    print("[")
    for x in ImgSpecClass.normalize_to_list(A):
        print("\t", x, ",")
    print("]")

    print("[")
    for x in ImgSpecClass.normalize_to_list([A, A]):
        print("\t", x, ",")
    print("]")

    print("[")
    for x in ImgSpecClass.normalize_to_list(((A, "Hello"), A)):
        print("\t", x, ",")
    print("]")
