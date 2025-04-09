# modules/digit_preprocessor.py
import cv2
import numpy as np
from keras.api.preprocessing.image import img_to_array
from typing import Optional, List

class DigitPreprocessor:
    @staticmethod
    def preprocess(digits: List[List[Optional[np.ndarray]]]) -> List[List[Optional[np.ndarray]]]:
        def prep(cell):
            resized = cv2.resize(cell, (32, 32))
            normalized = resized.astype("float") / 255.0
            arr = img_to_array(normalized)
            return np.expand_dims(arr, axis=0)

        return [[prep(cell) if cell is not None else None for cell in row] for row in digits]
