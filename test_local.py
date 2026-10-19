import cv2
import numpy as np
from cv2_helper import imshow, TextPos, TextAttr, KeyCode, PreloadImages

if __name__ == '__main__':
    A = PreloadImages.GRAY.CAT_01
    imshow("Original", A)
