import cv2
import numpy as np
import imutils
from modules.debug import DebugVisualizer
from skimage.segmentation import clear_border
from imutils.perspective import four_point_transform
from tensorflow.keras.preprocessing.image import img_to_array
from typing import Tuple, Optional, List


class SudokuImageProcessor:
    def __init__(self, image_path: str, debug: DebugVisualizer):
        """
        Initialize the image processor with an image path and preprocess the image.
        """
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        self.debug = debug
        self.sudoku_board = None

        if self.original_image is None:
            raise ValueError("Image could not be loaded. Check the file path.")

        self.original_image = imutils.resize(self.original_image, width=600)  # Resize for easier processing
        self.debug.add_image("Original_Image", self.original_image)

    def preprocess_image_for_edge_detection(self) -> None:
        """
        Convert the image to grayscale, apply Gaussian blur, and perform edge detection.
        """
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 3)
        thresholded_image = cv2.adaptiveThreshold(
            blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        self.debug.add_image("Thresholded_Edges", thresholded_image)

        return None

    def detect_sudoku_board_contour(self) -> None:
        """
        Detect the Sudoku puzzle's outline from the image using contours.
        """
        thresholded_image = self.debug.images.get("Thresholded_Edges")

        contours = cv2.findContours(thresholded_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        height, width = self.original_image.shape[:2]
        min_contour_area = 0.3 * height * width  # 30% of the image's area

        largest_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        if len(largest_contours) == 0:
            raise Exception("No large enough Sudoku grid found.")

        smallest_contour = min(largest_contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(smallest_contour, True)
        approx_polygon = cv2.approxPolyDP(smallest_contour, 0.02 * perimeter, True)

        if len(approx_polygon) != 4:
            raise Exception("The detected contour is not quadrilateral.")

        sudoku_contour = approx_polygon

        detected_board_image = self.original_image.copy()
        cv2.drawContours(detected_board_image, [sudoku_contour], -1, (0, 255, 0), 2)
        self.debug.add_image("Detected_Sudoku_Outline", detected_board_image)

        self.sudoku_board = four_point_transform(
            cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY), sudoku_contour.reshape(4, 2)
        )

        self.debug.add_image("Warped_Sudoku_Board", self.sudoku_board)

        return None

    @staticmethod
    def extract_digit_from_cell(cell_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts the digit from a single Sudoku cell, if present.
        """
        binary_image = cv2.threshold(cell_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cleared_image = clear_border(binary_image)

        contours = cv2.findContours(cleared_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        if len(contours) == 0:
            return None

        largest_contour = max(contours, key=cv2.contourArea)

        mask = np.zeros(cleared_image.shape, dtype="uint8")
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        (height, width) = cleared_image.shape
        occupied_percentage = cv2.countNonZero(mask) / float(width * height)

        if occupied_percentage < 0.01:
            return None

        digit_image = cv2.bitwise_and(cleared_image, cleared_image, mask=mask)

        return digit_image

    def split_sudoku_board_into_cells(self, grid_size=9) -> List[List[np.ndarray]]:
        """
        Split the warped Sudoku board image into individual cells.
        """
        if self.sudoku_board is None:
            raise ValueError("Warped Sudoku board not found. Run detect_sudoku_board_contour() first.")

        resized_board = cv2.resize(self.sudoku_board, (450, 450))
        rows = np.vsplit(resized_board, grid_size)
        cells = [np.hsplit(row, grid_size) for row in rows]

        return cells

    def extract_digits_from_cells(self) -> List[List[Optional[np.ndarray]]]:
        """
        Extracts digits from each cell of the 9x9 Sudoku grid.
        """
        puzzle_cells = self.split_sudoku_board_into_cells()
        extracted_digits = [
            [self.extract_digit_from_cell(cell) for cell in row] for row in puzzle_cells
        ]

        self.display_extracted_digits(extracted_digits)

        return extracted_digits

    def display_extracted_digits(self, extracted_cells: List[List[Optional[np.ndarray]]]) -> None:
        """
        Visualizes the extracted digits from each cell.
        """
        grid_image = None

        for row in extracted_cells:
            resized_cells = [
                cv2.resize(cell, (28, 28)) if cell is not None else np.zeros((28, 28), dtype="uint8") for cell in row
            ]
            row_image = np.concatenate(resized_cells, axis=1)
            grid_image = row_image if grid_image is None else np.concatenate([grid_image, row_image], axis=0)

        self.debug.add_image("Extracted_Digits_Grid", grid_image)

    @staticmethod
    def preprocess_cell_image(cell_image: np.ndarray) -> np.ndarray:
        """
        Preprocess a Sudoku cell image for recognition.
        """
        resized_cell = cv2.resize(cell_image, (28, 28))  # Resize to 28x28 pixels
        normalized_cell = resized_cell.astype("float") / 255.0  # Normalize pixel values to [0,1]
        cell_array = img_to_array(normalized_cell)  # Convert to array
        cell_array = np.expand_dims(cell_array, axis=0)  # Add batch dimension
        return cell_array

    def preprocess_extracted_digits(self, extracted_digits: List[List[Optional[np.ndarray]]]) -> List[List[Optional[np.ndarray]]]:
        """
        Preprocesses the extracted digits from the Sudoku cells for model prediction.
        """
        preprocessed_cells = []

        for row in extracted_digits:
            preprocessed_row = [
                self.preprocess_cell_image(cell) if cell is not None else None for cell in row
            ]
            preprocessed_cells.append(preprocessed_row)

        return preprocessed_cells

    def process_sudoku_image(self) -> List[List[Optional[np.ndarray]]]:
        """
        Executes the entire image processing pipeline for Sudoku digit extraction.
        """
        try:
            print("[INFO] Preprocessing the image...")
            self.preprocess_image_for_edge_detection()

            print("[INFO] Detecting Sudoku board...")
            self.detect_sudoku_board_contour()

            print("[INFO] Extracting digits from Sudoku cells...")
            extracted_digits = self.extract_digits_from_cells()

            print("[INFO] Preprocessing the extracted cells...")
            preprocessed_digit_images = self.preprocess_extracted_digits(extracted_digits)

            print("[SUCCESS] Sudoku image processing completed.")
            return preprocessed_digit_images

        except Exception as e:
            print(f"[ERROR] {e}")
            return None
