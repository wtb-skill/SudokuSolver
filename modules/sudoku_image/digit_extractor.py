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

        digit = cv2.bitwise_and(cleared, cleared, mask=mask)

        return digit

    def extract_digits(self):
        cells = self.split_into_cells()

        # Extract digits and resize them to (32, 32)
        digits = []
        for row in cells:
            digit_row = []
            for cell in row:
                digit = self.extract_digit_from_cell(cell)
                if digit is not None:
                    digit = cv2.resize(digit, (32, 32))  # Resize to 32x32 right here
                digit_row.append(digit)
            digits.append(digit_row)

        # Visualize the digits (already resized)
        self._visualize_digits(digits)

        # Collect the resized digits for later processing
        self.debug.collect_digit_cells(digits)

        return digits

    def _visualize_digits(self, digits: List[List[Optional[np.ndarray]]]):
        grid_image = None
        for row in digits:
            cells = [cell if cell is not None else np.zeros((32, 32), dtype="uint8") for cell in row]
            row_img = np.concatenate(cells, axis=1)
            grid_image = row_img if grid_image is None else np.concatenate([grid_image, row_img], axis=0)
        self.debug.add_image("Extracted_Digits_Grid", grid_image)

    # def _collect_non_empty_cells_table(self) -> List[List[Optional[np.ndarray]]]:
    #     """
    #     Collect all digit images in a 2D grid where empty cells are None,
    #     and non-empty cells are 32x32 images.
    #     """
    #     result_grid = []
    #     cells = self.split_into_cells()
    #
    #     for row in cells:
    #         row_digits = []
    #         for cell in row:
    #             digit = self.extract_digit_from_cell(cell)
    #             if digit is not None:
    #                 digit = cv2.resize(digit, (32, 32))
    #             row_digits.append(digit)
    #         result_grid.append(row_digits)
    #
    #     self.debug.collect_digit_cells(result_grid)
    #
    #     return result_grid
