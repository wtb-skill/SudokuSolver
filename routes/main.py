# main.py
import os

# Suppress TensorFlow oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import (Blueprint, render_template, request, redirect, send_from_directory, send_file, abort, session,
                   Response)
from modules.digit_recognition import SudokuDigitRecognizer
from modules.debug import ImageCollector
from modules.board_display import SudokuBoardDisplay
from modules.solving_algorithm.norvig_solver import NorvigSolver
from modules.solving_algorithm.sudoku_converter import SudokuConverter
from modules.sudoku_image.sudoku_pipeline import SudokuPipeline
from modules.user_data_collector import UserDataCollector
import pickle

# Initialize Blueprint
main_bp = Blueprint('main', __name__)

# Ensure the uploads folder exists
os.makedirs('uploads', exist_ok=True)

# Initialize ImageCollector instance
image_collector = ImageCollector()

@main_bp.route('/')
def home() -> str:
    """
    Clears the session and resets the image collector, then renders the home page.

    Returns:
        str: The rendered HTML template for the home page.
    """
    session.clear()  # Clears all session data
    image_collector.reset()
    return render_template('index.html')

def upload_file() -> str or Response:
    """
    Handles the uploaded file, processes the Sudoku image, recognizes digits,
    solves the Sudoku, and displays the solution or an error message.

    Returns:
        str: The rendered HTML template with the solution or an error message.
    """

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:

        try:
            # Step 1: Process the image and extract the 2D list of digit images
            sudoku_pipeline = SudokuPipeline(image_file=file, image_collector=image_collector)
            preprocessed_digit_images = sudoku_pipeline.process_sudoku_image()

            # Step 2: Categorize digit images into actual numbers
            recognizer = SudokuDigitRecognizer(model_path="models/sudoku_digit_recognizer.keras")
            unsolved_board = recognizer.convert_cells_to_digits(preprocessed_digit_images)

            # Create an image of the unsolved board
            sudoku_board_display = SudokuBoardDisplay(image_collector)
            sudoku_board_display.draw_unsolved_board(board=unsolved_board)

            # Convert the 2D digit board to a Sudoku grid string
            converter = SudokuConverter()
            sudoku_grid = converter.board_to_string(digit_board=unsolved_board)

            # Step 3: Solve the Sudoku puzzle
            solver = NorvigSolver()
            solved_grid = solver.solve(sudoku_grid)

            if not solved_grid:
                # debug:
                # debug.display_images_in_grid()
                # debug.save_images()

                # Store the digits grid temporarily
                digit_images = image_collector.digit_cells
                pickled_digit_images = pickle.dumps(digit_images)  # Pickle the digit grid
                session['digit_images'] = pickled_digit_images

                # Store the unsolved board in the session
                session['unsolved_board'] = unsolved_board.tolist()  # Convert to list to store in session

                #return "Sudoku puzzle could not be solved."
                return render_template('no_solution.html')

            # Convert the solved Sudoku string back to a 2D digit board
            solved_board = converter.dict_to_board(solved_grid)

            # Step 4: Create an image of the solved board
            sudoku_board_display.draw_solved_board(unsolved_board=unsolved_board, solved_board=solved_board)

            # debug:
            # debug.display_images_in_grid()
            # debug.save_images()

            # Render the solution page
            return render_template('solution.html')

        except Exception as e:
            return f"Error processing image: {str(e)}"


@main_bp.route('/uploads/<filename>')
def uploaded_file(filename: str) -> Response:
    """
    Serves an uploaded file from the 'uploads' directory.

    Parameters:
        filename (str): The name of the file to serve.

    Returns:
        Response: The file served from the 'uploads' directory.
    """
    return send_from_directory('uploads', filename)

@main_bp.route('/debug-image/<step_name>')
def get_debug_image(step_name) -> Response:
    """
    Serves a debug image directly from memory.

    Parameters:
        step_name (str): Name of the image stored in DebugVisualizer.

    Returns:
        Flask response with image data.
    """
    image_bytes = image_collector.get_image_bytes(step_name)
    if image_bytes is None:
        print(f"Image {step_name} not found!")  # Debugging line
        abort(404)  # Return 404 if the image is not found

    return send_file(image_bytes, mimetype='image/jpeg')


@main_bp.route('/handle-collect-decision', methods=['POST'])
def handle_collect_decision() -> str or Response:
    """
    Handles the user's decision to either collect data or go back to the home page.

    Returns:
        str: The rendered template based on the user's decision.
    """
    if request.form.get('collect_data') == 'YES':
        # Extract unsolved board from debug
        unsolved_board = session.pop('unsolved_board', None)

        if unsolved_board is None:
            return "Error: No unsolved board found.", 400

        return render_template('collect_user_data.html', sudoku_grid=unsolved_board)

    return redirect('/')

@main_bp.route('/collect-user-data', methods=['POST'])
def collect_user_data() -> str or Response:
    """
    Collects user-provided labels for the Sudoku puzzle and saves the labeled data.

    Returns:
        str: A message indicating the result of the operation (success or failure).
    """
    try:
        # Step 1: Get labels and convert them to integers
        labels = request.form.getlist('cell')
        labels = [int(label.strip()) for label in labels if label.strip().isdigit()]  # Convert to integers

        # Step 2: Load and validate digit images
        collector = UserDataCollector()
        digit_images = collector.load_digit_images_from_session(session)
        collector.validate_labels(digit_images, labels)

        # Step 3: Save data
        collector.save_labeled_data(digit_images, labels)

        # Step 4: Cleanup
        session.clear()
        return redirect('/')

    except ValueError as ve:
        return str(ve), 400
    except Exception as e:
        return f"Unexpected error: {str(e)}", 500