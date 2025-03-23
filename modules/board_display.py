# modules/board_display.py
import numpy as np
import cv2
from modules.debug import DebugVisualizer


class SudokuBoardDisplay:
    def __init__(self, debug: DebugVisualizer = None):
        """
        Initializes the Sudoku board display.

        Parameters:
            debug (DebugVisualizer, optional): Debugging tool to store intermediate results.
        """
        self.debug = debug  # Store debug object

    def draw_sudoku_board(self, board: np.ndarray, cell_size: int = 50, font_scale: float = 1,
                          thickness: int = 2, solved: bool = False) -> np.ndarray:
        """
        Draws a Sudoku board and stores it in the debug instance.

        Parameters:
            board (np.ndarray): A 9x9 NumPy array representing the Sudoku grid.
            cell_size (int): The size of each cell in pixels. Default is 50.
            font_scale (float): The scale factor for the text size. Default is 1.
            thickness (int): The thickness of grid lines and text. Default is 2.
            solved (bool): If True, marks the board as solved for debugging.

        Returns:
            np.ndarray: The generated Sudoku board image.
        """
        grid_size = board.shape[0]  # 9x9
        image_size = grid_size * cell_size  # Total image size

        # Create a blank white image
        board_img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

        # Draw the grid lines
        for i in range(grid_size + 1):
            thickness_line = 3 if i % 3 == 0 else 1  # Thicker lines for 3x3 boxes
            cv2.line(board_img, (0, i * cell_size), (image_size, i * cell_size), (0, 0, 0), thickness_line)
            cv2.line(board_img, (i * cell_size, 0), (i * cell_size, image_size), (0, 0, 0), thickness_line)

        # Write the digits in the cells
        for row in range(grid_size):
            for col in range(grid_size):
                num = board[row, col]
                if num != 0:  # Only draw numbers if they are not zero
                    text = str(num)
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    text_x = col * cell_size + (cell_size - text_size[0]) // 2
                    text_y = row * cell_size + (cell_size + text_size[1]) // 2
                    cv2.putText(board_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                                thickness)

        # Store the debug image if debugging is enabled
        if self.debug:
            debug_step_name = "Solved_Board" if solved else "Unsolved_Board"
            self.debug.add_image(debug_step_name, board_img)

        return board_img  # Return the board image instead of saving it
