# modules/sudoku_image/step_4_digit_preprocessor.py
import cv2
import numpy as np
from keras.api.preprocessing.image import img_to_array
from modules.types import *

class DigitPreprocessor:
    """
    A utility class for preprocessing digit images in a Sudoku grid.
    Converts raw digit images into normalized arrays suitable for input into a neural network.
    """

    @staticmethod
    def preprocess(digits: DigitGrid) -> ProcessedDigitGrid:
        """
        Preprocesses a grid of digit images by resizing them to (32, 32),
        normalizing pixel values to the range [0, 1], and converting them
        into a format compatible with Keras (i.e., (1, 32, 32, 1) arrays).

        Args:
            digits (DigitGrid): A 9x9 grid of digit images or None values.

        Returns:
            ProcessedDigitGrid: A 9x9 grid where each non-empty cell contains a preprocessed
                       4D NumPy array ready for model prediction. Empty cells remain None.
        """
        def prep(cell: DigitImage) -> ProcessedDigitImage:
            resized = cv2.resize(cell, (32, 32))
            normalized = resized.astype("float") / 255.0
            arr = img_to_array(normalized)
            return np.expand_dims(arr, axis=0)

        processed_grid: ProcessedDigitGrid = [
            [prep(cell) if cell is not None else None for cell in row]
            for row in digits
        ]
        return processed_grid

