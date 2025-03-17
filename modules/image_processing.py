# modules/image_processing.py
from imutils.perspective import four_point_transform  # Helps in transforming perspective to get a top-down view
from skimage.segmentation import clear_border  # Removes noise at the image borders
import numpy as np
import imutils
import cv2
import os


def find_puzzle(image, debug=False):
    """
    Detects and extracts the Sudoku puzzle from an image.

    Parameters:
        image: Input image containing a Sudoku puzzle.
        debug: If True, displays debugging images for each step.

    Returns:
        A tuple containing:
        - The color (RGB) version of the extracted puzzle.
        - The grayscale version of the extracted puzzle.
    """
    # Convert the image to grayscale (simplifies processing)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and smooth the image
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # Apply adaptive thresholding to emphasize edges and contrast
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    thresh = cv2.bitwise_not(thresh)  # Invert colors so the puzzle grid is white on black

    if debug:
        cv2.imshow("Puzzle Thresh", thresh)
        cv2.waitKey(0)

    # Find contours of the largest objects in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # Sort contours by size

    puzzleCnt = None  # Variable to store the Sudoku grid contour

    # Loop through contours to find the puzzle outline (a large quadrilateral)
    for c in cnts:
        peri = cv2.arcLength(c, True)  # Perimeter of the contour
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # Approximate to 4-sided shape

        if len(approx) == 4:  # If the contour has four points, it's likely the Sudoku grid
            puzzleCnt = approx
            break

    # If no Sudoku grid is found, raise an error
    if puzzleCnt is None:
        raise Exception("Could not find Sudoku puzzle outline. Check thresholding and contours.")

    # Optionally visualize the detected puzzle outline
    if debug:
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)

    # Apply a perspective transform to get a top-down view of the puzzle
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))  # Color version
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))  # Grayscale version

    if debug:
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)

    return puzzle, warped  # Return both color and grayscale versions


def create_warped_image(file_path):
    """
    Reads an image, finds the Sudoku puzzle, and saves the warped version.

    Parameters:
        file_path: Path to the input Sudoku image.

    Returns:
        The warped (grayscale) puzzle image and its filename.
    """
    warped_filename = 'warped_sudoku_board.jpg'
    warped_image_path = os.path.join('uploads', warped_filename)

    image = cv2.imread(file_path)  # Read the input image
    image = imutils.resize(image, width=600)  # Resize for easier processing

    puzzle, warped = find_puzzle(image)  # Extract the Sudoku grid

    cv2.imwrite(warped_image_path, warped)  # Save the warped (top-down) image

    return warped, warped_filename


def extract_digit(cell, debug=False):
    """
    Extracts the digit from a Sudoku cell (if present).

    Parameters:
        cell: An individual Sudoku cell (grayscale).
        debug: If True, shows intermediate processing steps.

    Returns:
        The processed digit as an image or None if no digit is detected.
    """
    # Apply automatic thresholding
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)  # Remove any unwanted borders touching the edge

    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)

    # Find external contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        return None  # Return None if no contours (empty cell)

    # Find the largest contour (likely the digit)
    c = max(cnts, key=cv2.contourArea)

    # Create a mask to extract the digit
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    # Compute the percentage of masked pixels
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)

    # If less than 3% of the cell is filled, ignore it (likely noise)
    if percentFilled < 0.01:
        return None

    # Apply the mask to extract the digit
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)

    return digit


def show_cells(warped, grid_size=9, debug=True):
    """
    Divides the Sudoku puzzle into 81 cells and displays them.

    Parameters:
        warped: The top-down grayscale Sudoku puzzle image.
        grid_size: The number of rows/columns (default: 9x9).
        debug: If True, visualizes the extracted cells.

    Returns:
        A 2D list containing extracted digits or None for empty cells.
    """
    h, w = warped.shape  # Get puzzle dimensions
    cell_height = h // grid_size  # Height of each cell
    cell_width = w // grid_size  # Width of each cell

    puzzle_cells = []

    for row in range(grid_size):
        row_digits = []
        for col in range(grid_size):
            # Define the region for the current cell
            x_start, y_start = col * cell_width, row * cell_height
            x_end, y_end = (col + 1) * cell_width, (row + 1) * cell_height

            cell = warped[y_start:y_end, x_start:x_end]  # Crop the cell

            digit = extract_digit(cell, debug=False)  # Extract digit (if any)
            row_digits.append(digit)

        puzzle_cells.append(row_digits)

    # Debugging visualization
    if debug:
        grid_image = None
        for row in range(grid_size):
            row_cells = []
            for col in range(grid_size):
                cell = puzzle_cells[row][col]
                if cell is not None:
                    resized_cell = cv2.resize(cell, (28, 28))  # Normalize size
                    row_cells.append(resized_cell)
                else:
                    blank_cell = np.zeros((28, 28), dtype="uint8")  # Empty cell
                    row_cells.append(blank_cell)

            row_image = np.concatenate(row_cells, axis=1)  # Row-wise concatenation

            if grid_image is None:
                grid_image = row_image
            else:
                grid_image = np.concatenate([grid_image, row_image], axis=0)  # Stack rows

        cv2.imshow("Sudoku Puzzle Cells", grid_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return puzzle_cells
