from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2
import os

def find_puzzle(image, debug=False):
    # Since color is not needed for edge detection, the image is converted to grayscale to simplify processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blurring reduces noise and smooths the image, which helps in detecting more consistent edges
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # apply adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # check to see if we are visualizing each step of the image
    # processing pipeline (in this case, thresholding)
    if debug:
        cv2.imshow("Puzzle Thresh", thresh)
        cv2.waitKey(0)

    # find contours in the thresholded image and sort them by size in
    # descending order
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # initialize a contour that corresponds to the puzzle outline
    puzzleCnt = None

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle
        if len(approx) == 4:
            puzzleCnt = approx
            break

    # if the puzzle contour is empty then our script could not find
    # the outline of the Sudoku puzzle so raise an error
    if puzzleCnt is None:
        raise Exception(("Could not find Sudoku puzzle outline. "
                         "Try debugging your thresholding and contour steps."))

    # check to see if we are visualizing the outline of the detected
    # Sudoku puzzle
    if debug:
        # draw the contour of the puzzle on the image and then display
        # it to our screen for visualization/debugging purposes
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)

    # apply a four point perspective transform to both the original
    # image and grayscale image to obtain a top-down bird's eye view
    # of the puzzle
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

    # check to see if we are visualizing the perspective transform
    if debug:
        # show the output warped image (again, for debugging purposes)
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)

    # return a 2-tuple of puzzle in both RGB and grayscale
    return (puzzle, warped)

def create_warped_image(file_path):
    # Fixed name for the warped image
    warped_filename = 'warped_sudoku_board.jpg'
    warped_image_path = os.path.join('uploads', warped_filename)

    # Read the uploaded image
    image = cv2.imread(file_path)

    # Call your find_puzzle function to get the warped image
    puzzle, warped = find_puzzle(image)

    # Save the warped image (this will overwrite the previous one)
    cv2.imwrite(warped_image_path, warped)

    return warped_image_path  # Return the fixed filename

def extract_digit(cell, debug=False):
    # Apply automatic thresholding to the cell and then clear any
    # connected borders that touch the border of the cell
    thresh = cv2.threshold(cell, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    # Check to see if we are visualizing the cell thresholding step
    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)

    # Find contours in the thresholded cell
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # If no contours were found, then this is an empty cell
    if len(cnts) == 0:
        return None

    # Otherwise, find the largest contour in the cell and create a mask for the contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    # Compute the percentage of masked pixels relative to the total area of the image
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)

    # If less than 3% of the mask is filled, then we are looking at noise and can safely ignore the contour
    if percentFilled < 0.03:
        return None

    # Apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    # Check to see if we should visualize the masking step
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)

    # Return the digit to the calling function
    return digit

def show_cells(warped, grid_size=9, debug=True):
    """
    Divides the Sudoku puzzle into cells and shows each cell for debugging.

    Parameters:
        warped: The top-down view of the Sudoku puzzle (warped image).
        grid_size: The size of the Sudoku grid (default is 9x9).
        debug: A flag to visualize the process (default is False).
    """
    # Get the height and width of the warped image
    h, w = warped.shape

    # Calculate the height and width of each cell
    cell_height = h // grid_size
    cell_width = w // grid_size

    # Initialize storage for extracted digits
    puzzle_cells = []

    for row in range(grid_size):
        row_digits = []
        for col in range(grid_size):
            # Define the coordinates of the current cell
            x_start = col * cell_width
            y_start = row * cell_height
            x_end = (col + 1) * cell_width
            y_end = (row + 1) * cell_height

            # Crop the cell from the warped image
            cell = warped[y_start:y_end, x_start:x_end]

            # Extract the digit from the cell (if any)
            digit = extract_digit(cell, debug=False)

            # Append the extracted digit (None if no digit found)
            row_digits.append(digit)

        # Append the row's digits to the main puzzle_cells list
        puzzle_cells.append(row_digits)

    # Debugging visualization
    if debug:
        grid_image = None
        for row in range(grid_size):
            row_cells = []
            for col in range(grid_size):
                cell = puzzle_cells[row][col]
                if cell is not None:
                    # Resize each digit image to a fixed size
                    resized_cell = cv2.resize(cell, (50, 50))
                    row_cells.append(resized_cell)
                else:
                    # If the cell is empty, create a blank image
                    blank_cell = np.zeros((50, 50), dtype="uint8")
                    row_cells.append(blank_cell)

            # Horizontally concatenate the cells for each row
            row_image = np.concatenate(row_cells, axis=1)

            if grid_image is None:
                grid_image = row_image
            else:
                # Vertically concatenate each row to form the grid
                grid_image = np.concatenate([grid_image, row_image], axis=0)

        # Show the entire grid of cells
        cv2.imshow("Sudoku Puzzle Cells", grid_image)
        cv2.waitKey(0)

    # Close any open image windows if debug is enabled
    if debug:
        cv2.destroyAllWindows()

    return puzzle_cells


