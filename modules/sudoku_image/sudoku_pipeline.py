# modules/sudoku_pipeline.py
import cv2
import numpy as np
from modules.sudoku_image.image_preprocessor import ImagePreprocessor
from modules.sudoku_image.board_detector import BoardDetector
from modules.sudoku_image.digit_extractor import DigitExtractor
from modules.sudoku_image.digit_preprocessor import DigitPreprocessor

class SudokuPipeline:
    def __init__(self, image_file, debug):
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image could not be loaded. Check the file path.")

        self.debug = debug
        self.image = image

    def process_sudoku_image(self):
        print("[INFO] Preprocessing the image...")
        preprocessor = ImagePreprocessor(self.image, self.debug)
        thresholded = preprocessor.preprocess()

        print("[INFO] Detecting Sudoku board...")
        detector = BoardDetector(preprocessor.get_original(), thresholded, self.debug)
        warped = detector.detect()

        print("[INFO] Extracting digits from Sudoku cells...")
        extractor = DigitExtractor(warped, self.debug)
        digits = extractor.extract_digits()

        print("[INFO] Preprocessing the extracted cells...")
        return DigitPreprocessor.preprocess(digits)
