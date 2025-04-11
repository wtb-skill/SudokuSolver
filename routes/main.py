# main.py
import os

# Suppress TensorFlow oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Blueprint, render_template, request, redirect, send_from_directory, send_file, abort, session
from modules.digit_recognition import SudokuDigitRecognizer
from modules.debug import ImageCollector
from modules.board_display import SudokuBoardDisplay
from modules.solving_algorithm import NorvigSolver, SudokuConverter
from modules.sudoku_image.sudoku_pipeline import SudokuPipeline
from modules.user_data_collector import UserDataCollector
import pickle

# Initialize Blueprint
main_bp = Blueprint('main', __name__)

# Ensure the uploads folder exists
os.makedirs('uploads', exist_ok=True)

# Initialize ImageCollector instance
debug = ImageCollector()

@main_bp.route('/')
def home():
    session.clear()  # Clears all session data
    debug.reset()
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
            sudoku_pipeline = SudokuPipeline(image_file=file, debug=debug)
            preprocessed_digit_images = sudoku_pipeline.process_sudoku_image()

            digits_grid = debug.digit_cells
            # Debugging: Print size of the digits grid (rows and columns)
            print(f"RIGHT AFTER PREPROCESSING- Digits grid size: {len(digits_grid)} rows, {len(digits_grid[0])} columns")

            # Step 2: Categorize digit images into actual numbers
            recognizer = SudokuDigitRecognizer(model_path="models/sudoku_digit_recognizer.keras")
            unsolved_board = recognizer.convert_cells_to_digits(preprocessed_digit_images)

            # Create an image of the unsolved board
            sudoku_board_display = SudokuBoardDisplay(debug)
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
                digit_images = debug.digit_cells
                pickled_digit_images = pickle.dumps(digit_images)  # Pickle the digit grid
                session['digit_images'] = pickled_digit_images

                # Store the unsolved board in the session
                session['unsolved_board'] = unsolved_board.tolist()  # Convert to list to store in session

                #return "Sudoku puzzle could not be solved."
                return render_template('no_solution.html')

            # Convert the solved Sudoku string back to a 2D digit board
            solved_board = converter.dict_to_board(solved_grid)

            # Create an image of the solved board
            sudoku_board_display.draw_solved_board(unsolved_board=unsolved_board, solved_board=solved_board)

            # debug:
            # debug.display_images_in_grid()
            # debug.save_images()

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
        print(f"Image {step_name} not found!")  # Debugging line
        abort(404)  # Return 404 if the image is not found

    return send_file(image_bytes, mimetype='image/jpeg')


@main_bp.route('/handle-collect-decision', methods=['POST'])
def handle_collect_decision():
    if request.form.get('collect_data') == 'YES':
        # Extract unsolved board from debug
        unsolved_board = session.pop('unsolved_board', None)

        if unsolved_board is None:
            return "Error: No unsolved board found.", 400

        # Display the page with the extracted digits and sudoku grid
        return render_template('collect_user_data.html', sudoku_grid=unsolved_board)

    # If the user chooses NO, just return to the index page
    return redirect('/')

@main_bp.route('/collect-user-data', methods=['POST'])
def collect_user_data():
    try:
        # Step 1: Get labels
        labels = request.form.getlist('cell')
        labels = [label.strip() for label in labels if label.strip().isdigit()]

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