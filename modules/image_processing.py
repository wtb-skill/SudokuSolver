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

    def extract_cells(self, grid_size: int = 9, debug: bool = False) -> List[List[Optional[np.ndarray]]]:
        """
        Divides the Sudoku puzzle into (81) cells and extracts digits.
        """
        if self.warped_board is None:
            raise ValueError("Warped image not available. Run find_puzzle() first.")

        h, w = self.warped_board.shape
        cell_height = h // grid_size
        cell_width = w // grid_size
        puzzle_cells = []

        for row in range(grid_size):
            row_digits = []
            for col in range(grid_size):
                x_start, y_start = col * cell_width, row * cell_height
                x_end, y_end = (col + 1) * cell_width, (row + 1) * cell_height

                cell = self.warped_board[y_start:y_end, x_start:x_end]
                digit = self._extract_digit(cell)
                row_digits.append(digit)

            puzzle_cells.append(row_digits)

        if debug:
            self._show_extracted_cells(puzzle_cells)

        return puzzle_cells

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
