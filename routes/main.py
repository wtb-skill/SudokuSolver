from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory
from modules.image_processing import create_warped_image, show_cells
from modules.digit_recognition import cells_to_digits
from modules.board_display import draw_sudoku_board
import os


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
        # Save the uploaded file
        file_path = os.path.join('uploads/', file.filename)
        file.save(file_path)

        # Step 1: Get the top-down view of the Sudoku grid
        warped_image, warped_filename = create_warped_image(file_path)

        # Step 2: Extract Sudoku cells and recognize digits (as pixels)
        puzzle_cells = show_cells(warped_image, debug=True)

        # Step 3: Convert extracted digits into a Sudoku board (pixels to numbers)
        board = cells_to_digits(puzzle_cells)

        # Save a jpg of the unsolved board
        draw_sudoku_board(board, solved=False)

        # Render solution.html with required information
        return render_template('solution.html',
                               filename=file.filename,
                               warped_filename=warped_filename,
                               board=board)



# Serve uploaded files properly from the 'uploads' directory
@main_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)
