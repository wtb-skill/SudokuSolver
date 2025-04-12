# modules/sudoku_image/sudoku_pipeline.py
import cv2
import numpy as np
from werkzeug.datastructures import FileStorage

from modules.sudoku_image.step_1_image_preprocessor import ImagePreprocessor
from modules.sudoku_image.step_2_board_detector import BoardDetector
from modules.sudoku_image.step_3_digit_extractor import DigitExtractor
from modules.sudoku_image.step_4_digit_preprocessor import DigitPreprocessor
from modules.types import ProcessedDigitGrid
from modules.debug import ImageCollector


class SudokuPipeline:
    """
    Coordinates the full pipeline for processing a Sudoku image, including preprocessing,
    board detection, digit extraction, and digit preprocessing for model input.
    """

    def __init__(self, image_file: FileStorage, image_collector: ImageCollector):
        """
        Initializes the pipeline by reading and decoding the uploaded image file.

        Args:
            image_file (BinaryIO): A file-like object containing the uploaded image.
            image_collector (ImageCollector): A debug image collector to store intermediate images.
        """
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image could not be loaded. Check the file content.")

        self.image_collector: ImageCollector = image_collector
        self.image: np.ndarray = image

    def process_sudoku_image(self) -> ProcessedDigitGrid:
        """
        Executes the full Sudoku processing pipeline.

        Returns:
            ProcessedDigitGrid: A 9x9 grid of processed digit images ready for model inference.
        """
        print("[INFO] Preprocessing the image...")
        preprocessor = ImagePreprocessor(self.image, self.image_collector)
        thresholded = preprocessor.preprocess()

        print("[INFO] Detecting Sudoku board...")
        detector = BoardDetector(preprocessor.get_original(), thresholded, self.image_collector)
        warped = detector.detect()

        print("[INFO] Extracting digits from Sudoku cells...")
        extractor = DigitExtractor(warped, self.image_collector)
        digits = extractor.extract_digit_images()

        print("[INFO] Preprocessing the extracted cells...")
        return DigitPreprocessor.preprocess(digits)

