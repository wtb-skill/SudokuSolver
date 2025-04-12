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
        return ''.join(str(digit_board[row, col]) for row in range(9) for col in range(9))

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
        rows = 'ABCDEFGHI'
        cols = '123456789'

        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(cols):
                key = row + col
                board[row_idx, col_idx] = int(sudoku_dict[key])

        return board
