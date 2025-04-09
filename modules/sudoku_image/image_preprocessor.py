# modules/sudoku_image/image_preprocessor.py
import cv2
import imutils
import numpy as np

class ImagePreprocessor:
    def __init__(self, image: np.ndarray, debug=None):
        self.debug = debug
        self.original_image = imutils.resize(image, width=600)
        self.gray = None
        self.thresholded = None
        self.debug.add_image("Original_Image", self.original_image)

    def preprocess(self):
        self.gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(self.gray, (7, 7), 3)
        self.thresholded = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        self.debug.add_image("Thresholded_Edges", self.thresholded)

        return self.thresholded

    def get_original(self):
        return self.original_image

    def get_gray(self):
        return self.gray

    def get_thresholded(self):
        return self.thresholded
