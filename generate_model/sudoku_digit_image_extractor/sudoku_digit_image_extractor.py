import cv2
import os
import numpy as np
from modules.debug import ImageCollector
from modules.sudoku_image_pipeline.step_1_image_preprocessor import ImagePreprocessor
from modules.sudoku_image_pipeline.step_2_board_detector import BoardDetector
from modules.sudoku_image_pipeline.step_3_digit_extractor import DigitExtractor
from typing import List


class SudokuDigitImageExtractor:
    def __init__(self, input_folder: str, output_folder: str):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.image_collector = ImageCollector()

        os.makedirs(self.output_folder, exist_ok=True)

    def process_images(self):
        """
        Process all images in the input folder by running the full image pipeline:
        1. Preprocessing
        2. Board Detection
        3. Digit Extraction
        """
        for filename in os.listdir(self.input_folder):
            img_path = os.path.join(self.input_folder, filename)
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                print(f"Processing {filename}...")

                # Step 1: Image Preprocessing
                original_image = cv2.imread(img_path)
                preprocessor = ImagePreprocessor(original_image, self.image_collector)
                thresholded = preprocessor.preprocess()

                # Step 2: Board Detection
                board_detector = BoardDetector(preprocessor.get_original(), thresholded, self.image_collector)
                warped_board = board_detector.detect()

                # Step 3: Digit Extraction
                digit_extractor = DigitExtractor(warped_board, self.image_collector)
                _ = digit_extractor.extract_digit_images()

                # Cut sudoku image into 81 equal size cells and make a list out of non-empty cells:
                digit_images_list = self.image_collector.digit_cells

                # Save digit images from the list
                self.save_digit_images(digit_images_list, filename)

                # debug:
                # self.image_collector.display_images_in_grid()

                # Clear digit cells before processing the next sudoku image
                self.image_collector.reset()

    def save_digit_images(self, digit_images: List[np.ndarray], filename: str):
        """
        Save the extracted digit images directly into the output folder.

        Args:
            digit_images (List[np.ndarray]): Flat list of extracted digit cell images (32x32).
            filename (str): The name of the original image (used to generate unique names).
        """
        base_filename = os.path.splitext(filename)[0]

        for idx, cell in enumerate(digit_images):
            digit_filename = f"{base_filename}_{idx + 1}.png"
            output_path = os.path.join(self.output_folder, digit_filename)
            cv2.imwrite(output_path, cell)


if __name__ == "__main__":
    input_folder = "sudoku_images"
    output_folder = "extracted_digit_images"

    extractor = SudokuDigitImageExtractor(input_folder, output_folder)
    extractor.process_images()

