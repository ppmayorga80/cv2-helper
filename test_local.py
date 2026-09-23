import cv2
import numpy as np

from cv2_helper import imshow, TextPos, TextAttr, KeyCode
from cv2_helper import add_borders, add_text, resize, FitOption

if __name__ == '__main__':
    A = cv2.imread("/Users/pedro/Downloads/lion.jpg")
    B = cv2.imread("/Users/pedro/Downloads/cat.jpg")
    C = cv2.imread("/Users/pedro/Downloads/tiger.jpg")
    D = cv2.imread("/Users/pedro/Downloads/fermat.png")
    E = cv2.medianBlur(D, ksize=5)

    imshow("Hello", [
        (A,),
        (B, "Cute Cat", TextAttr(org=TextPos.TOP_CENTER)),
        (C, "Cute Tiger", TextAttr(org=TextPos.TOP_CENTER)),
        (D, "Cute Fermat", TextAttr(org=TextPos.CENTER, font_scale=2.0)),
        E
    ], wait_time=0, exit_key_codes=(KeyCode.QUIT, KeyCode.ESC), border=1)

    cv2.imwrite("/Users/pedro/Downloads/lion-x.jpg", A[6])
