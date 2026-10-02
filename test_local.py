import cv2
import numpy as np
from cv2_helper import imshow, TextPos, TextAttr, KeyCode, PreloadImages

if __name__ == '__main__':
    I = PreloadImages.GRAY.LENA
    K1 = np.array([
        [1, -1]
    ])
    K2 = np.array([
        [1],
        [-1]
    ])
    K3 = np.array([
        [0, 1],
        [-1, 0]
    ])
    K4 = np.array([
        [1, 0],
        [0, -1]
    ])

    A = cv2.filter2D(I, -1, K1)
    B = cv2.filter2D(I, -1, K2)
    C = cv2.filter2D(I, -1, K3)
    D = cv2.filter2D(I, -1, K4)

    S = A.astype("float32") ** 2 + B.astype("float32") ** 2
    S = (S / (np.max(S) - np.min(S)) * 255).astype("uint8")

    imshow("", [
        (I, "Original"),
        (A, "Derivative in X"),
        (B, "Derivative in Y"),
        (C, "Derivative in Diag /"),
        (D, "Derivative in Diag \\"),
        (S, "Sobel"),
    ])