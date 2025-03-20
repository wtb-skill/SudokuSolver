# modules/digit_recognition.py
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2


class DigitRecognizer:
    def __init__(self, model_path: str):
        """
        Initializes the DigitRecognizer with a pre-trained model.

        Parameters:
            model_path (str): Path to the trained digit recognition model.
        """
        self.model = load_model(model_path)

    def _preprocess_cell(self, cell: np.ndarray) -> np.ndarray:
        """
        Preprocesses a Sudoku cell image for digit recognition.

        Parameters:
            cell (np.ndarray): The image of a digit.

        Returns:
            np.ndarray: Preprocessed image ready for model prediction.
        """
        roi = cv2.resize(cell, (28, 28))  # Resize to 28x28 pixels
        roi = roi.astype("float") / 255.0  # Normalize pixel values to [0,1]
        roi = img_to_array(roi)  # Convert image to array
        roi = np.expand_dims(roi, axis=0)  # Add batch dimension
        return roi

    def _predict_digit(self, cell: np.ndarray) -> int:
        """
        Predicts the digit in a given Sudoku cell image.

        Parameters:
            cell (np.ndarray): The preprocessed image of a digit.

        Returns:
            int: The predicted digit (0-9, where 0 represents an empty cell).
        """
        roi = self._preprocess_cell(cell)
        pred = self.model.predict(roi).argmax(axis=1)[0]
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
