import cv2
import numpy as np

from cv2_helper import imshow, TextPos, TextAttr, KeyCode
from cv2_helper import add_borders, add_labels, resize, FitOption


def custom_convolution(A, delta=0):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    X = cv2.filter2D(A, ddepth=-1, kernel=kernel, delta=delta)
    return X


if __name__ == '__main__':
    I = cv2.imread("/Users/pedro/Downloads/lion.jpg")
    G = cv2.cvtColor(cv2.cvtColor(I, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    E = cv2.imread("/Users/pedro/Downloads/fermat.png")

    A = [np.empty((1, 1))] * 10
    A[0] = I
    A[1] = cv2.GaussianBlur(I, (13, 13), 0)
    A[2] = cv2.bilateralFilter(I, d=0, sigmaColor=10, sigmaSpace=10)
    A[3] = cv2.medianBlur(I, ksize=11)
    A[4] = cv2.boxFilter(I, -1, (11, 11))
    A[5] = custom_convolution(I, delta=0)
    A[6] = G
    A[7] = cv2.GaussianBlur(G, (13, 13), 0)
    A[8] = cv2.bilateralFilter(G, d=0, sigmaColor=10, sigmaSpace=10)
    A[9] = E

    A = resize(A, A[0], fx=0.5, fy=0.5, fit_option=FitOption.FIT_AUTO, fill_color=(255, 255, 0))
    A = add_borders(A, border=1, border_color=(255, 255, 255))
    A = add_labels([
        (A[0], "XOrixinalX", TextAttr(org=TextPos.TOP_LEFT)),
        (A[1], "XGaussianX", TextAttr(org=TextPos.TOP_CENTER)),
        (A[2], "XBilateralX", TextAttr(org=TextPos.TOP_CENTER)),
        (A[3], "XMedianX", TextAttr(org=TextPos.CENTER_LEFT)),
        (A[4], "XBoxFilterX", TextAttr(org=TextPos.CENTER)),
        (A[5], "XCUSTOMX", TextAttr(org=TextPos.CENTER_RIGHT)),
        (A[6], "X-GRAY-X", TextAttr(org=TextPos.BOTTOM_LEFT)),
        (A[7], "X-GRAY-Y", TextAttr(org=TextPos.BOTTOM_CENTER)),
        (A[8], "X-GRAY-Z", TextAttr(org=TextPos.BOTTOM_RIGHT)),
        (A[9], "ABCDEFG", TextAttr(org=TextPos.BOTTOM_CENTER)),
    ])

    imshow("Hello", A, wait_time=3000, exit_key_codes=(KeyCode.QUIT, KeyCode.ESC))

    cv2.imwrite("/Users/pedro/Downloads/lion-x.jpg", A[6])
