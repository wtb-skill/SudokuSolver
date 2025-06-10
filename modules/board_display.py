# modules/board_display.py
import numpy as np
import cv2
from modules.debug import ImageCollector


class SudokuBoardDisplay:
    def __init__(self, image_collector: ImageCollector):
        """
        Initializes the Sudoku board display.

        Parameters:
            image_collector (DebugVisualizer, optional): Debugging tool to store intermediate results.
        """
        self.image_collector = image_collector  # Store image

    def draw_unsolved_board(
        self,
        board: np.ndarray,
        cell_size: int = 50,
        font_scale: float = 1,
        thickness: int = 2,
    ) -> None:
        """
        Draws the unsolved Sudoku board, only drawing the initial given digits.

        Parameters:
            board (np.ndarray): A 9x9 NumPy array representing the Sudoku grid.
            cell_size (int): The size of each cell in pixels. Default is 50.
            font_scale (float): The scale factor for the text size. Default is 1.
            thickness (int): The thickness of grid lines and text. Default is 2.
        """
        grid_size = board.shape[0]  # 9x9
        image_size = grid_size * cell_size  # Total image size

        # Create a blank white image
        board_img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

        # Draw the grid lines
        for i in range(grid_size + 1):
            thickness_line = 3 if i % 3 == 0 else 1  # Thicker lines for 3x3 boxes
            cv2.line(
                board_img,
                (0, i * cell_size),
                (image_size, i * cell_size),
                (0, 0, 0),
                thickness_line,
            )
            cv2.line(
                board_img,
                (i * cell_size, 0),
                (i * cell_size, image_size),
                (0, 0, 0),
                thickness_line,
            )

        # Write the digits in the cells (only the given ones, not the solved ones)
        for row in range(grid_size):
            for col in range(grid_size):
                num = board[row, col]
                if num != 0:  # Only draw numbers if they are not zero (given numbers)
                    text = str(num)
                    text_size = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                    )[0]
                    text_x = col * cell_size + (cell_size - text_size[0]) // 2
                    text_y = row * cell_size + (cell_size + text_size[1]) // 2
                    cv2.putText(
                        board_img,
                        text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),
                        thickness,
                    )

        self.image_collector.add_image("Unsolved_Board", board_img)

        return None

    def draw_solved_board(
        self,
        unsolved_board: np.ndarray,
        solved_board: np.ndarray,
        cell_size: int = 50,
        font_scale: float = 1,
        thickness: int = 2,
    ) -> None:
        """
        Draws the solved Sudoku board, highlighting the difference (solved part in green).

        Parameters:
            unsolved_board (np.ndarray): The initial unsolved board.
            solved_board (np.ndarray): The solved Sudoku board.
            cell_size (int): The size of each cell in pixels. Default is 50.
            font_scale (float): The scale factor for the text size. Default is 1.
            thickness (int): The thickness of grid lines and text. Default is 2.
        """
        grid_size = unsolved_board.shape[0]  # 9x9
        image_size = grid_size * cell_size  # Total image size

        # Create a blank white image
        board_img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

        # Draw the grid lines
        for i in range(grid_size + 1):
            thickness_line = 3 if i % 3 == 0 else 1  # Thicker lines for 3x3 boxes
            cv2.line(
                board_img,
                (0, i * cell_size),
                (image_size, i * cell_size),
                (0, 0, 0),
                thickness_line,
            )
            cv2.line(
                board_img,
                (i * cell_size, 0),
                (i * cell_size, image_size),
                (0, 0, 0),
                thickness_line,
            )

        # Draw the digits in the cells
        for row in range(grid_size):
            for col in range(grid_size):
                unsolved_num = unsolved_board[row, col]
                solved_num = solved_board[row, col]

                if (
                    solved_num != 0
                ):  # Only draw numbers if they are not zero (solved numbers)
                    text = str(solved_num)
                    text_size = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                    )[0]
                    text_x = col * cell_size + (cell_size - text_size[0]) // 2
                    text_y = row * cell_size + (cell_size + text_size[1]) // 2

                    # Check if the cell value is different from the unsolved value
                    if unsolved_num != solved_num:
                        # Highlight the solved part in green
                        cv2.putText(
                            board_img,
                            text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (0, 255, 0),
                            thickness,
                        )  # Green color for solved parts
                    else:
                        cv2.putText(
                            board_img,
                            text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (0, 0, 0),
                            thickness,
                        )  # Black color for unsolved given cells

        self.image_collector.add_image("Solved_Board", board_img)

        return None
