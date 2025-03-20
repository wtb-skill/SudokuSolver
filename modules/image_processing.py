import cv2
import numpy as np
import imutils
from skimage.segmentation import clear_border
from imutils.perspective import four_point_transform
from typing import Tuple, Optional, List


class ImageProcessor:
    def __init__(self, image_path: str):
        """
        Initialize with an image path and preprocess it.
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.warped_board = None

        if self.image is None:
            raise ValueError("Image could not be loaded. Check the file path.")

        self.image = imutils.resize(self.image, width=600)  # Resize for easier processing

    def preprocess_image(self) -> np.ndarray:
        """
        Convert the image to grayscale, apply Gaussian blur, and thresholding.
        """
        # Convert the image to grayscale (simplifies processing)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and smooth the image
        blurred = cv2.GaussianBlur(gray, (7, 7), 3)

        # Apply adaptive thresholding to emphasize edges and contrast
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return cv2.bitwise_not(thresh)

    def find_board(self, debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and extract the Sudoku puzzle from the image.
        """
        thresh = self.preprocess_image()

        # Find contours of the largest objects in the thresholded image
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        puzzle_contour = None

        # Loop through contours to find the puzzle outline (a large quadrilateral)
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                puzzle_contour = approx
                break

        # If no Sudoku grid is found, raise an error
        if puzzle_contour is None:
            raise Exception("Could not find Sudoku puzzle outline. Check thresholding and contours.")

        # Optionally visualize the detected puzzle outline
        if debug:
            output = self.image.copy()
            cv2.drawContours(output, [puzzle_contour], -1, (0, 255, 0), 2)
            cv2.imshow("Puzzle Outline", output)
            cv2.waitKey(0)

        # Apply a perspective transform to get a top-down view of the board
        self.warped_board = four_point_transform(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), puzzle_contour.reshape(4, 2))

        return self.warped_board

    # def find_board_new(self, debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Detect and extract the Sudoku puzzle from the image by selecting the smallest contour
    #     among the largest contours, with a dynamic threshold based on image size.
    #     """
    #     thresh = self.preprocess_image()
    #
    #     # Find contours of the largest objects in the thresholded image
    #     contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     contours = imutils.grab_contours(contours)
    #
    #     # Sort contours by area (descending)
    #     contours = sorted(contours, key=cv2.contourArea, reverse=True)
    #
    #     # Get the image dimensions
    #     height, width = self.image.shape[:2]
    #
    #     # Define a dynamic threshold based on image size
    #     # Use a percentage of the image area, e.g., 70% of the total image area
    #     min_area = 0.5 * height * width  # 70% of the image's area
    #
    #     # Find the largest contours that exceed the dynamic threshold
    #     largest_contours = []
    #     for contour in contours:
    #         area = cv2.contourArea(contour)
    #         if area > min_area:  # Only consider contours larger than the dynamic threshold
    #             largest_contours.append(contour)
    #
    #     # If no large contours are found, raise an exception
    #     if len(largest_contours) == 0:
    #         raise Exception("Could not find a large enough Sudoku grid. Check thresholding and contours.")
    #
    #     # Now we find the smallest contour among the largest ones
    #     smallest_contour = min(largest_contours, key=cv2.contourArea)
    #
    #     # Approximate the polygon (quadrilateral) from the smallest large contour
    #     peri = cv2.arcLength(smallest_contour, True)
    #     approx = cv2.approxPolyDP(smallest_contour, 0.02 * peri, True)
    #
    #     # If the approximated contour is not a quadrilateral, skip it
    #     # if len(approx) != 4:
    #     #     raise Exception("Could not find Sudoku puzzle outline. The contour is not quadrilateral.")
    #
    #     puzzle_contour = approx
    #
    #     # Optionally visualize the detected puzzle outline
    #     if debug:
    #         output = self.image.copy()
    #         cv2.drawContours(output, [puzzle_contour], -1, (0, 255, 0), 2)
    #         cv2.imshow("Puzzle Outline", output)
    #         cv2.waitKey(0)
    #
    #     # Apply a perspective transform to get a top-down view of the board
    #     self.warped_board = four_point_transform(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY),
    #                                              puzzle_contour.reshape(4, 2))
    #
    #     return self.warped_board

    def save_warped_board(self, save_path: str = 'uploads/warped_sudoku_board.jpg') -> str:
        """
        Saves the warped Sudoku board.
        Parameters:
            save_path (str): Path to the input Sudoku image.

        Returns:
            Tuple[np.ndarray, str]: The warped (grayscale) top-down view of board image.
        """
        if self.warped_board is None:
            raise ValueError("Warped board is not generated yet. Run find_puzzle() first.")

        cv2.imwrite(save_path, self.warped_board)
        return save_path

    def _extract_digit(self, cell: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts the digit from a Sudoku cell.
        """
        # Apply automatic thresholding
        thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = clear_border(thresh)

        # Find external contours
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # Return None if no contours (empty cell)
        if len(contours) == 0:
            return None

        # Find the largest contour (likely the digit)
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a mask to extract the digit
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        # Compute the percentage of masked pixels
        (h, w) = thresh.shape
        percent_filled = cv2.countNonZero(mask) / float(w * h)

        # If less than 1% of the cell is filled, ignore it (likely noise)
        if percent_filled < 0.01:
            return None

        # Apply the mask to extract the digit
        digit = cv2.bitwise_and(thresh, thresh, mask=mask)

        return digit

    def split_into_cells(self, grid_size=9):
        """
        Divides the Sudoku puzzle into (81) cells.
        """
        if self.warped_board is None:
            raise ValueError("Warped image not available. Run find_puzzle() first.")

        self.warped_board = cv2.resize(self.warped_board, (450, 450))

        rows = np.vsplit(self.warped_board, grid_size)

        puzzle_cells = [np.hsplit(r, grid_size) for r in rows]

        return puzzle_cells

    def extract_digits_from_cells(self, debug=False):
        """
        Extracts digits from each cell in the provided 9x9 grid.
        Returns a 9x9 list with extracted digits or None for empty cells.
        """
        puzzle_cells = self.split_into_cells()
        digits = [[self._extract_digit(cell) for cell in row] for row in puzzle_cells]

        if debug:
            self._show_extracted_cells(digits)

        return digits

    def _show_extracted_cells(self, puzzle_cells: List[List[Optional[np.ndarray]]]):
        """
        Helper function to visualize extracted Sudoku cells.
        """
        grid_image = None

        for row in puzzle_cells:
            row_cells = [cv2.resize(cell, (28, 28)) if cell is not None else np.zeros((28, 28), dtype="uint8") for cell in row]
            row_image = np.concatenate(row_cells, axis=1)

            grid_image = row_image if grid_image is None else np.concatenate([grid_image, row_image], axis=0)

        cv2.imshow("Sudoku Puzzle Cells", grid_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
