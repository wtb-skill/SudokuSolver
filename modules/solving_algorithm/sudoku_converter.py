import numpy as np
from typing import Dict


class SudokuConverter:
    """
    Utility class for converting between different Sudoku representations:
    - From a 9x9 numpy array to a string format.
    - From a dictionary solution back to a 9x9 numpy array.
    """

    @staticmethod
    def board_to_string(digit_board: np.ndarray) -> str:
        """
        Converts a 9x9 numpy array into a single-line Sudoku grid string.

        Args:
            digit_board (np.ndarray): A 9x9 numpy array of integers (0 for empty, 1-9 for digits).

        Returns:
            str: A string of 81 characters representing the grid, row-wise.
        """
        assert digit_board.shape == (9, 9), "Input must be a 9x9 numpy array."
        return "".join(
            str(digit_board[row, col]) for row in range(9) for col in range(9)
        )

    @staticmethod
    def dict_to_board(sudoku_dict: Dict[str, str]) -> np.ndarray:
        """
        Converts a dictionary-based Sudoku solution into a 9x9 numpy array.

        Args:
            sudoku_dict (Dict[str, str]): A dict where keys are positions ('A1' to 'I9') and values are digits as strings.

        Returns:
            np.ndarray: A 9x9 numpy array representing the Sudoku board.
        """
        board: np.ndarray = np.zeros((9, 9), dtype=int)
        rows = "ABCDEFGHI"
        cols = "123456789"

        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(cols):
                key = row + col
                board[row_idx, col_idx] = int(sudoku_dict[key])

        return board

    @staticmethod
    def labels_to_board(labels: list[int]) -> np.ndarray:
        """
        Converts a flat list of 81 labels into a 9x9 numpy array.

        Args:
            labels (list[int]): A flat list of 81 integers representing cell values.

        Returns:
            np.ndarray: A 9x9 numpy array board.

        Raises:
            ValueError: If the input list is not of length 81.
        """
        if len(labels) != 81:
            raise ValueError("Expected exactly 81 labels to form a 9x9 board.")
        return np.array([labels[i : i + 9] for i in range(0, 81, 9)])
