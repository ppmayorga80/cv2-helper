import cv2

from cv2_helper import PreloadImages, TextAttr, TextPos, imcomp

if __name__ == '__main__':
    A = PreloadImages.COLOR.CAT_01
    B = PreloadImages.COLOR.CAT_02
    C = PreloadImages.COLOR.CAT_03
    D = PreloadImages.GRAY.LENA
    E = PreloadImages.BINARY.HORSE

    ta1 = TextAttr(org=TextPos.TOP_CENTER, font_scale=2, color=(16, 240, 240))
    G0 = imcomp([
        (A, "Gauss-1", TextPos.TOP_CENTER),
        (B, "Gauss-2", ta1),
        (C, "Gauss-2", ta1),
        (D, "Lena", ta1),
        (E, "Horse", ta1),
    ], shape=None, border=1, border_color=0)

    cv2.namedWindow("G0", cv2.WINDOW_GUI_EXPANDED)
    cv2.imshow('G0', G0)
    cv2.waitKey(0)
