# modules/digit_extractor.py
import cv2
import numpy as np
import imutils
from skimage.segmentation import clear_border
from typing import List, Optional

class DigitExtractor:
    def __init__(self, warped_board: np.ndarray, debug=None):
        self.board = cv2.resize(warped_board, (450, 450))
        self.debug = debug

    def split_into_cells(self, grid_size=9):
        rows = np.vsplit(self.board, grid_size)
        return [np.hsplit(row, grid_size) for row in rows]

    def extract_digit_from_cell(self, cell: np.ndarray) -> Optional[np.ndarray]:
        thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cleared = clear_border(thresh)

        contours = cv2.findContours(cleared.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        if len(contours) == 0:
            return None

        c = max(contours, key=cv2.contourArea)
        mask = np.zeros(cleared.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        (h, w) = cleared.shape
        percent_filled = cv2.countNonZero(mask) / float(w * h)

        if percent_filled < 0.01:
            return None

        return cv2.bitwise_and(cleared, cleared, mask=mask)

    def extract_digits(self):
        cells = self.split_into_cells()
        digits = [[self.extract_digit_from_cell(cell) for cell in row] for row in cells]

        self._visualize_digits(digits)

        return digits

    def _visualize_digits(self, digits: List[List[Optional[np.ndarray]]]):
        grid_image = None
        for row in digits:
            cells = [cv2.resize(cell, (32, 32)) if cell is not None else np.zeros((32, 32), dtype="uint8") for cell in row]
            row_img = np.concatenate(cells, axis=1)
            grid_image = row_img if grid_image is None else np.concatenate([grid_image, row_img], axis=0)
        self.debug.add_image("Extracted_Digits_Grid", grid_image)