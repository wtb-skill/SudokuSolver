# modules/sudoku_image/step_3_digit_extractor.py
import cv2
import numpy as np
import imutils
from skimage.segmentation import clear_border
from typing import Optional
from modules.debug import ImageCollector
from modules.utils.types import *

class DigitExtractor:
    """
    Extracts digits from a warped Sudoku board by detecting cells, segmenting digits,
    and collecting non-empty digits for further processing.

    Attributes:
        board (np.ndarray): The warped Sudoku board resized to (450, 450).
        image_collector (ImageCollector): An instance of ImageCollector for storing debug images.
    """

    def __init__(self, warped_board: np.ndarray, image_collector: ImageCollector):
        """
        Initializes the DigitExtractor with a warped Sudoku board and an image collector.

        Args:
            warped_board (np.ndarray): The input warped Sudoku board (grayscale).
            image_collector (ImageCollector): The image collector used to store images.
        """
        self.board: np.ndarray = cv2.resize(warped_board, (450, 450))
        self.image_collector: ImageCollector = image_collector

    def split_into_cells(self, grid_size: int = 9) -> DigitGrid:
        """
        Splits the warped Sudoku board into smaller cells based on the grid size.

        Args:
            grid_size (int): The number of rows and columns in the Sudoku grid (default is 9).

        Returns:
            DigitGrid: A 9x9 grid where each element is a cell from the Sudoku grid.
        """
        rows = np.vsplit(self.board, grid_size)
        return [np.hsplit(row, grid_size) for row in rows]

    @staticmethod
    def extract_digit_image_from_cell(cell: np.ndarray) -> DigitImage:
        """
        Extracts a digit image from a given Sudoku cell by enhancing contrast,
        thresholding, clearing borders, filtering out edge-touching contours,
        and assuming all remaining contours are part of the digit.

        Args:
            cell (np.ndarray): The image of a single Sudoku cell.

        Returns:
            DigitImage: The extracted digit as a binary image, or None if no digit is found.
        """
        # Step 1: Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_enhanced = clahe.apply(cell)

        # Step 2: Reduce noise with Gaussian blur
        blurred = cv2.GaussianBlur(clahe_enhanced, (3, 3), 0)

        # Step 3: Threshold (invert so digit is white)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Step 4: Clear border artifacts
        cleared = clear_border(thresh)

        # Step 5: Find contours
        contours = cv2.findContours(cleared.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        if not contours:
            return None

        # Step 6: Filter out contours too close to edges
        h, w = cleared.shape
        margin = 3  # pixels
        valid_contours = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            if x <= margin or y <= margin or x + cw >= w - margin or y + ch >= h - margin:
                continue  # Skip contour touching the edges
            valid_contours.append(contour)

        if not valid_contours:
            return None

        # Step 7: Create mask from valid contours
        mask = np.zeros(cleared.shape, dtype="uint8")
        cv2.drawContours(mask, valid_contours, -1, (255,), -1)

        # Step 8: Filter by percent filled
        percent_filled = cv2.countNonZero(mask) / float(w * h)
        if percent_filled < 0.01:
            return None

        # Step 9: Extract digit using mask
        digit_image = cv2.bitwise_and(cleared, cleared, mask=mask)
        return digit_image

    def extract_digit_images(self) -> DigitGrid:
        """
        Extracts all digit images from the Sudoku grid by processing each cell individually,
        resizing non-empty digit images to (32, 32), and collecting them for later use.

        Returns:
            DigitGrid: A 9x9 grid of digit images (or None for empty cells).
        """
        cells = self.split_into_cells() # at this point digit image is faint but visible

        # Extract digit images and resize them to (32, 32)
        digit_images_grid: DigitGrid = []
        for row in cells:
            digit_image_row: DigitRow = []
            for cell in row:
                digit_image = self.extract_digit_image_from_cell(cell)
                if digit_image is not None:
                    digit_image = cv2.resize(digit_image, (32, 32))  # Resize to 32x32 right here
                digit_image_row.append(digit_image)
            digit_images_grid.append(digit_image_row)

        # Visualize the digit images (already resized)
        self._visualize_digits(digit_images_grid)

        # Collect the resized digit images for later processing
        self.image_collector.collect_digit_cells(digit_images_grid)

        return digit_images_grid

    def _visualize_digits(self, digits: DigitGrid) -> None:
        """
        Constructs a visual representation of all extracted digit images by stitching them together
        into a single grid image, mimicking the Sudoku layout. The resulting image is added to the
        image collector for debugging purposes.

        Args:
            digits (DigitGrid): A 9x9 grid of extracted digit images, where each element is either
                                a (32x32) NumPy array or None for empty cells.
        """
        grid_image: Optional[np.ndarray] = None
        for row in digits:
            cells = [cell if cell is not None else np.zeros((32, 32), dtype="uint8") for cell in row]
            row_img = np.concatenate(cells, axis=1)
            grid_image = row_img if grid_image is None else np.concatenate([grid_image, row_img], axis=0)
        self.image_collector.add_image("Extracted_Digits_Grid", grid_image)


