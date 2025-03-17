from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

def cells_to_digits(puzzle_cells):
    board = np.zeros((9, 9), dtype="int")
    model_path = "models/output_model.h5"
    model = load_model(model_path)

    for row in range(9):
        for col in range(9):
            cell = puzzle_cells[row][col]

            if cell is not None:
                # Resize the cell to 28x28 pixels to match the MNIST model's input size
                roi = cv2.resize(cell, (28, 28))
                roi = roi.astype("float") / 255.0  # Normalize pixel values to [0,1]
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)  # Add batch dimension

                # Predict the digit using the trained model
                pred = model.predict(roi).argmax(axis=1)[0]
                board[row, col] = pred  # Store the predicted digit in the board
            else:
                board[row, col] = 0  # Empty cell remains 0
    return board


