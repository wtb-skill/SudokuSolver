# modules/digit_recognition.py
from tensorflow.keras.models import load_model
import numpy as np


class DigitRecognizer:
    def __init__(self, model_path: str):
        """
        Initializes the DigitRecognizer with a pre-trained model.

        Parameters:
            model_path (str): Path to the trained digit recognition model.
        """
        self.model = load_model(model_path)


    def _predict_digit(self, cell: np.ndarray) -> int:
        """
        Predicts the digit in a given Sudoku cell image.

        Parameters:
            cell (np.ndarray): The preprocessed image of a digit.

        Returns:
            int: The predicted digit (0-9, where 0 represents an empty cell).
        """
        pred = self.model.predict(cell).argmax(axis=1)[0]
        return pred

    def cells_to_digits(self, puzzle_cells: list) -> np.ndarray:
        """
        Converts extracted Sudoku cells into a 9x9 board with recognized digits.

        Parameters:
            puzzle_cells (list): A 2D list containing 81 Sudoku cells. Each cell is either an image of a digit or None.

        Returns:
            np.ndarray: A 9x9 numpy array representing the Sudoku board with recognized digits (0 for empty cells).
        """
        board = np.zeros((9, 9), dtype=int)  # Initialize a 9x9 Sudoku board

        for row in range(9):
            for col in range(9):
                cell = puzzle_cells[row][col]
                if cell is not None:
                    board[row, col] = self._predict_digit(cell)
                else:
                    board[row, col] = 0  # Empty cell remains 0

        return board
