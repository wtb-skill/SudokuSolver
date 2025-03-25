# main.py
import os

# Suppress TensorFlow oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Blueprint, render_template, request, redirect, send_from_directory, send_file, abort
from modules.image_processing import SudokuImageProcessor
from modules.digit_recognition import SudokuDigitRecognizer
from modules.debug import DebugVisualizer
from modules.board_display import SudokuBoardDisplay


# Initialize Blueprint
main_bp = Blueprint('main', __name__)

# Ensure the uploads folder exists
os.makedirs('uploads', exist_ok=True)

# Initialize DebugVisualizer instance
debug = DebugVisualizer()

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

        try:
            # Step 1: Process the image and extract the 2D list of digit images
            processor = SudokuImageProcessor(file, debug)
            preprocessed_digit_images = processor.process_sudoku_image()

            # Step 2: Categorize digit images into actual numbers
            recognizer = SudokuDigitRecognizer(model_path="models/sudoku_digit_recognizer.keras")
            board = recognizer.convert_cells_to_digits(preprocessed_digit_images)

            # Create an image of the unsolved board
            unsolved_board = SudokuBoardDisplay(debug)
            unsolved_board.draw_sudoku_board(board, solved=False)

            debug.display_images_in_grid()

            # Render the solution page
            return render_template('solution.html')

        except Exception as e:
            return f"Error processing image: {str(e)}"


# Serve uploaded files properly from the 'uploads' directory
@main_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@main_bp.route('/debug-image/<step_name>')
def get_debug_image(step_name):
    """
    Serves a debug image directly from memory.

    Parameters:
        step_name (str): Name of the image stored in DebugVisualizer.

    Returns:
        Flask response with image data.
    """
    image_bytes = debug.get_image_bytes(step_name)
    if image_bytes is None:
        abort(404)  # Return 404 if the image is not found

    return send_file(image_bytes, mimetype='image/jpeg')