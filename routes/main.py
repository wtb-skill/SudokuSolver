# main.py
from flask import Blueprint, render_template, request, redirect, send_from_directory
from modules.image_processing import ImageProcessor
from modules.digit_recognition import DigitRecognizer
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

        # Step 1: Process the image and extract the Sudoku board
        processor = ImageProcessor(file_path)

        try:
            processor.find_board(debug=True)  # Extract top-down view
            processor.save_warped_board()  # Save the top-down view of the board

            # Step 2: Extract Sudoku cells
            puzzle_cells = processor.extract_digits_from_cells(debug=False)  # Extract cells with digits

            # Step 3: Convert extracted digits into a Sudoku board
            recognizer = DigitRecognizer(model_path="models/sudoku_digit_recognizer.keras")
            board = recognizer.cells_to_digits(puzzle_cells)

            # Save a JPG of the unsolved board
            draw_sudoku_board(board, solved=False)

            # Render the solution page
            return render_template('solution.html',
                                   filename=file.filename)

        except Exception as e:
            return f"Error processing image: {str(e)}"


# Serve uploaded files properly from the 'uploads' directory
@main_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)
