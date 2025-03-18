# modules/digit_recognition.py
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2


# Load the pre-trained model
model_path = "models/sudoku_digit_recognizer.keras"
model = load_model(model_path)


def cells_to_digits(puzzle_cells: list) -> np.ndarray:
    """
    Converts the extracted Sudoku cells into a 9x9 Sudoku board with recognized digits using a pre-trained model.

    Parameters:
        puzzle_cells (list): A 2D list containing 81 Sudoku cells. Each cell is either an image of a digit or None (empty).

    Returns:
        np.ndarray: A 9x9 numpy array representing the Sudoku board with recognized digits (0 for empty cells).
    """
    # Initialize an empty 9x9 Sudoku board with zeros (empty cells)
    board = np.zeros((9, 9), dtype="int")

    # Loop through each row and column of the 9x9 grid
    for row in range(9):
        for col in range(9):
            cell = puzzle_cells[row][col]

            if cell is not None:
                # Preprocess the cell (resize to 28x28 pixels to match MNIST model input size)
                roi = cv2.resize(cell, (28, 28))
                roi = roi.astype("float") / 255.0  # Normalize pixel values to [0,1]
                roi = img_to_array(roi)  # Convert the image to an array
                roi = np.expand_dims(roi, axis=0)  # Add batch dimension (for model prediction)

                # Predict the digit using the trained model
                pred = model.predict(roi).argmax(axis=1)[0]  # Get the digit with highest probability
                board[row, col] = pred  # Store the predicted digit in the board
            else:
                # If the cell is empty (None), assign 0 to represent an empty cell
                board[row, col] = 0  # Empty cell remains 0

    return board
