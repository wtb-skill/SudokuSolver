import cv2
import os
from modules.debug import ImageCollector
from modules.sudoku_image.step_1_image_preprocessor import ImagePreprocessor
from modules.sudoku_image.step_2_board_detector import BoardDetector
from modules.sudoku_image.step_3_digit_extractor import DigitExtractor


class SudokuProcessor:
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
                thresholded_image = preprocessor.preprocess()

                # Step 2: Board Detection
                board_detector = BoardDetector(original_image, thresholded_image, self.image_collector)
                warped_board = board_detector.detect()

                # Step 3: Digit Extraction
                digit_extractor = DigitExtractor(warped_board, self.image_collector)
                digit_images_grid = digit_extractor.extract_digit_images()

                # Save digit images for manual labeling
                self.save_digit_images(digit_images_grid, filename)

    def save_digit_images(self, digit_images_grid, filename):
        """
        Save the extracted digit images directly into the output folder, no subfolders.

        Args:
            digit_images_grid: 9x9 grid of extracted digit images (or None for empty cells).
            filename: The name of the original image (used to make filenames unique).
        """
        base_filename = os.path.splitext(filename)[0]

        counter = 1
        for row_idx, row in enumerate(digit_images_grid):
            for col_idx, cell in enumerate(row):
                if cell is not None:
                    # Create a unique filename
                    digit_filename = f"{base_filename}_{row_idx}_{col_idx}.png"
                    output_path = os.path.join(self.output_folder, digit_filename)

                    # Save the image
                    cv2.imwrite(output_path, cell)
                    counter += 1


if __name__ == "__main__":
    input_folder = "sudoku_images"
    output_folder = "extracted_digit_images"

    processor = SudokuProcessor(input_folder, output_folder)
    processor.process_images()
