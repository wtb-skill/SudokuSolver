from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory
from modules.image_processing import create_warped_image, show_cells
from modules.digit_recognition import cells_to_digits
from modules.board_display import draw_sudoku_board
import os
import cv2


# Initialize Blueprint
main_bp = Blueprint('main', __name__)

# Ensure the uploads folder exists
os.makedirs('uploads', exist_ok=True)


@main_bp.route('/')
def home():
    return render_template('index.html')


@main_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file to the 'uploads' folder
        file_path = os.path.join('uploads/', file.filename)
        file.save(file_path)

        # Step 1: Warp the image to get a top-down view of the Sudoku grid
        warped_image_path = create_warped_image(file_path)

        # Step 2: Load the warped image
        image = cv2.imread(warped_image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return "Error: Could not load the warped Sudoku image."

        # Step 3: Extract Sudoku cells and recognize digits
        puzzle_cells = show_cells(image, debug=False)

        # Step 4: Convert extracted digits into a Sudoku board
        board = cells_to_digits(puzzle_cells)

        # Save a jpg of unsolved board
        draw_sudoku_board(board, solved=False)

        # Render solution.html with all required information
        return render_template('solution.html',
                               filename=file.filename,
                               warped_filename=warped_image_path.split('/')[-1],
                               board=board)


# Serve uploaded files properly from the 'uploads' directory
@main_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)
