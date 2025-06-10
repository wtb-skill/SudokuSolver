# modules/sudoku_image/sudoku_pipeline.py
import cv2
import numpy as np
import logging

from werkzeug.datastructures import FileStorage

from modules.sudoku_image_pipeline.step_1_image_preprocessor import ImagePreprocessor
from modules.sudoku_image_pipeline.step_2_board_detector import BoardDetector
from modules.sudoku_image_pipeline.step_3_digit_extractor import DigitExtractor
from modules.sudoku_image_pipeline.step_4_digit_preprocessor import DigitPreprocessor
from modules.utils.types import ProcessedDigitGrid
from modules.debug import ImageCollector

# Create a logger for this module
logger = logging.getLogger(__name__)


class SudokuPipeline:
    """
    Coordinates the full pipeline for processing a Sudoku image, including preprocessing,
    board detection, digit extraction, and digit preprocessing for model input.
    """

    def __init__(self, image_file, image_collector: ImageCollector):
        """
        Initializes the pipeline by reading and decoding the uploaded image file.

        Args:
            image_file (str or FileStorage): Path to the uploaded image file or a FileStorage object.
            image_collector (ImageCollector): A debug image collector to store intermediate images.
        """
        self.image_collector: ImageCollector = image_collector

        # Check if image_file is a path (string) or FileStorage object
        if isinstance(image_file, str):
            self.image = cv2.imread(image_file)
            if self.image is None:
                raise ValueError(f"Image could not be loaded from path: {image_file}. Check the file content.")
        elif isinstance(image_file, FileStorage):
            file_bytes = np.frombuffer(image_file.read(), np.uint8)
            self.image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if self.image is None:
                raise ValueError("Image could not be loaded. Check the file content.")
        else:
            raise ValueError("image_file must be a string (path) or a FileStorage object.")

    def process_sudoku_image(self) -> ProcessedDigitGrid:
        """
        Executes the full Sudoku processing pipeline.

        Returns:
            ProcessedDigitGrid: A 9x9 grid of processed digit images ready for model inference.
        """
        logger.info("Preprocessing the image...")
        preprocessor = ImagePreprocessor(self.image, self.image_collector)
        thresholded = preprocessor.preprocess()

        logger.info("Detecting Sudoku board...")
        detector = BoardDetector(preprocessor.get_original(), thresholded, self.image_collector)
        warped = detector.detect()

        logger.info("Extracting digits from Sudoku cells...")
        extractor = DigitExtractor(warped, self.image_collector)
        digits = extractor.extract_digit_images()

        logger.info("Preprocessing the extracted cells...")
        return DigitPreprocessor.preprocess(digits)
