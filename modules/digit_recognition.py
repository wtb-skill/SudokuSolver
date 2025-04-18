# modules/digit_recognition.py
from keras.api.models import load_model
import numpy as np
from modules.types import *

class SudokuDigitRecognizer:
    def __init__(self, model_path: str):
        """
        Initializes the SudokuDigitRecognizer with a pre-trained model.

        Parameters:
            model_path (str): Path to the trained digit recognition model.
        """
        self.model = load_model(model_path)

    def _predict_single_digit(self, cell: ProcessedDigitImage  ) -> int:
        """
        Predicts the digit in a given Sudoku cell image and prints the probabilities.

        Parameters:
            cell (ProcessedDigitImage  ): The preprocessed image of a digit, which should be a 2D numpy array.

        Returns:
            int: The predicted digit (1-9, where 0 represents an empty cell).
        """
        probabilities = self.model.predict(cell)[0]  # Get prediction probabilities
        predicted_digit = np.argmax(probabilities) + 1  # Adjusting for 1-based Sudoku digits

        # Formatting probabilities for better readability
        prob_str = ", ".join([f"{i+1}: {prob:.2%}" for i, prob in enumerate(probabilities)])

        print(f"   ✅ Predicted Digit: {predicted_digit}")
        print(f"   📊 Probability Distribution: {prob_str}")

        return predicted_digit

    def convert_cells_to_digits(self, extracted_cells: ProcessedDigitGrid) -> np.ndarray:
        """
        Converts extracted Sudoku cells into a 9x9 board with recognized digits.

        Parameters:
            extracted_cells (ProcessedDigitGrid): A 2D list (9x9 grid) containing 81 Sudoku cells,
                                            where each cell is either an image of a digit or None.

        Returns:
            np.ndarray: A 9x9 numpy array representing the Sudoku board with recognized digits (0 for empty cells).
        """
        digit_board = np.zeros((9, 9), dtype=int)  # Initialize a 9x9 Sudoku board

        for row in range(9):
            for col in range(9):
                cell = extracted_cells[row][col]
                if cell is not None:
                    print(f"[INFO] 🧩Prediction on cell [{row + 1}][{col + 1}].")
                    digit_board[row, col] = self._predict_single_digit(cell)

        return digit_board
