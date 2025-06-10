import cv2
import os
import numpy as np
from typing import List

from modules.debug import ImageCollector
from modules.sudoku_image_pipeline.step_1_image_preprocessor import ImagePreprocessor
from modules.sudoku_image_pipeline.step_2_board_detector import BoardDetector
from modules.sudoku_image_pipeline.step_3_digit_extractor import DigitExtractor


class SudokuDigitImageExtractor:
    """
    Handles batch processing of Sudoku images from a folder, extracting individual digit images
    from each puzzle and saving them for later use (e.g., model training or analysis).
    """

    def __init__(self, input_folder: str, output_folder: str) -> None:
        """
        Initializes the extractor with input and output directories.

        Args:
            input_folder (str): Path to the folder containing Sudoku images to process.
            output_folder (str): Path to the folder where extracted digit images will be saved.
        """
        self.input_folder: str = input_folder
        self.output_folder: str = output_folder
        self.image_collector: ImageCollector = ImageCollector()

        os.makedirs(self.output_folder, exist_ok=True)

    def process_images(self) -> None:
        """
        Processes all valid Sudoku images in the input folder through a pipeline of:
        1. Preprocessing
        2. Board Detection
        3. Digit Extraction

        Extracted digits are saved as individual image files in the output folder.
        """
        for filename in os.listdir(self.input_folder):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                print(f"Processing {filename}...")
                img_path: str = os.path.join(self.input_folder, filename)

                # Step 1: Image Preprocessing
                original_image: np.ndarray = cv2.imread(img_path)
                preprocessor = ImagePreprocessor(original_image, self.image_collector)
                thresholded: np.ndarray = preprocessor.preprocess()

                # Step 2: Board Detection
                board_detector = BoardDetector(
                    preprocessor.get_original(), thresholded, self.image_collector
                )
                warped_board: np.ndarray = board_detector.detect()

                # Step 3: Digit Extraction
                digit_extractor = DigitExtractor(warped_board, self.image_collector)
                _ = digit_extractor.extract_digit_images()

                # Cut sudoku image into 81 equal size cells and make a list out of non-empty cells:
                digit_images_list: List[np.ndarray] = self.image_collector.digit_cells

                # Save each digit image
                self.save_digit_images(digit_images_list, filename)

                # Optionally display debug images
                # self.image_collector.display_images_in_grid()

                # Clear stored images for next run
                self.image_collector.reset()

    def save_digit_images(self, digit_images: List[np.ndarray], filename: str) -> None:
        """
        Saves each digit image to the output folder with a unique name based on the original file.

        Args:
            digit_images (List[np.ndarray]): List of digit cell images (typically 32x32).
            filename (str): Original filename used as a base for naming saved digits.
        """
        base_filename: str = os.path.splitext(filename)[0]

        for idx, cell in enumerate(digit_images):
            digit_filename: str = f"{base_filename}_{idx + 1}.png"
            output_path: str = os.path.join(self.output_folder, digit_filename)
            cv2.imwrite(output_path, cell)


if __name__ == "__main__":
    input_folder = "sudoku_images"
    output_folder = "extracted_digit_images"

    extractor = SudokuDigitImageExtractor(input_folder, output_folder)
    extractor.process_images()
