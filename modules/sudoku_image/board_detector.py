# modules/board_detector.py
import cv2
import imutils
import numpy as np
from imutils.perspective import four_point_transform

class BoardDetector:
    def __init__(self, original_image: np.ndarray, thresholded: np.ndarray, debug=None):
        self.original_image = original_image
        self.thresholded = thresholded
        self.debug = debug
        self.warped = None

    def detect(self):
        contours = cv2.findContours(self.thresholded.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        height, width = self.original_image.shape[:2]
        min_area = 0.2 * height * width
        largest_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if not largest_contours:
            raise Exception("No large enough Sudoku grid found.")

        smallest_contour = min(largest_contours, key=cv2.contourArea)
        peri = cv2.arcLength(smallest_contour, True)
        approx = cv2.approxPolyDP(smallest_contour, 0.02 * peri, True)

        if len(approx) != 4:
            raise Exception("Detected contour is not quadrilateral.")

        outline_img = self.original_image.copy()
        cv2.drawContours(outline_img, [approx], -1, (0, 255, 0), 2)
        self.debug.add_image("Detected_Sudoku_Outline", outline_img)

        warped = four_point_transform(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY), approx.reshape(4, 2))
        self.warped = warped

        self.debug.add_image("Warped_Sudoku_Board", self.warped)

        return self.warped