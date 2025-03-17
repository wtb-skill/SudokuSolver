import numpy as np
import cv2
import os

def draw_sudoku_board(board, cell_size=50, font_scale=1, thickness=2, solved=False):
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
                cv2.putText(board_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # Ensure the uploads directory exists
    os.makedirs("uploads", exist_ok=True)

    # Save the image in the uploads folder
    filename = "sudoku_board_solved.jpg" if solved else "sudoku_board_unsolved.jpg"
    file_path = os.path.join("uploads", filename)
    cv2.imwrite(file_path, board_img)

    print(f"Sudoku board saved at: {file_path}")  # Print the save location for debugging
