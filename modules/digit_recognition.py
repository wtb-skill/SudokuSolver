# modules/digit_recognition.py
import logging
from keras.api.models import load_model
import numpy as np
import os
from modules.utils.types import *

# Create a logger for this module
logger = logging.getLogger(__name__)


class SudokuDigitRecognizer:
    def __init__(self, model_path: str):
        """
        Initializes the SudokuDigitRecognizer with a pre-trained model.

        Parameters:
            model_path (str): Path to the trained digit recognition model.
        """
        self.model = self._load_model_from_directory(model_path)

    @staticmethod
    def _load_model_from_directory(directory: str):
        """
        Loads the most recent model from the given directory.

        Parameters:
            directory (str): Path to the directory containing model files.

        Returns:
            keras.Model: The loaded model.
        """
        valid_extensions = [".keras", ".h5"]
        model_files = [
            f
            for f in os.listdir(directory)
            if any(f.endswith(ext) for ext in valid_extensions)
        ]

        if len(model_files) == 0:
            raise ValueError("No valid model files found in the specified directory.")

        # Get the most recent model based on modification time
        latest_model_file = max(
            model_files, key=lambda f: os.path.getmtime(os.path.join(directory, f))
        )
        model_file_path = os.path.join(directory, latest_model_file)

        logger.info(f"Loading model: {model_file_path}")
        return load_model(model_file_path)

    def _predict_single_digit(self, cell: ProcessedDigitImage) -> int:
        """
        Predicts the digit in a given Sudoku cell image and logs the probabilities.

        Parameters:
            cell (ProcessedDigitImage): The preprocessed image of a digit, which should be a 2D numpy array.

        Returns:
            int: The predicted digit (1-9, where 0 represents an empty cell).
        """
        probabilities = self.model.predict(cell, verbose=0)[
            0
        ]  # Get prediction probabilities
        predicted_digit = (
            np.argmax(probabilities) + 1
        )  # Adjust for 1-based Sudoku digits

        # Formatting probabilities nicely
        prob_str = ", ".join(
            [f"{i+1}: {prob:.2%}" for i, prob in enumerate(probabilities)]
        )

        # Use internal _log method
        logger.info(f"   ✅ Predicted Digit: {predicted_digit}")
        logger.info(f"   📊 Probability Distribution: {prob_str}")

        return predicted_digit

    def convert_cells_to_digits(
        self, extracted_cells: ProcessedDigitGrid
    ) -> np.ndarray:
        """
        Converts extracted Sudoku cells into a 9x9 board with recognized digits.

        Parameters:
            extracted_cells (ProcessedDigitGrid): A 2D list (9x9 grid) containing 81 Sudoku cells,
                                                   where each cell is either an image of a digit or None.

        Returns:
            np.ndarray: A 9x9 numpy array representing the Sudoku board with recognized digits (0 for empty cells).
        """
        digit_board = np.zeros((9, 9), dtype=int)

        for row in range(9):
            for col in range(9):
                cell = extracted_cells[row][col]
                if cell is not None:
                    logger.info(f"[INFO] 🧩 Prediction on cell [{row + 1}][{col + 1}].")
                    digit_board[row, col] = self._predict_single_digit(cell)

        return digit_board
