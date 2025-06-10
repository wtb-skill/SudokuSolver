# sudoku_solver.py
import os

# Suppress TensorFlow oneDNN warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    send_from_directory,
    send_file,
    abort,
    session,
    flash,
    get_flashed_messages,
    Response,
)
from modules.digit_recognition import SudokuDigitRecognizer
from modules.debug import ImageCollector
from modules.board_display import SudokuBoardDisplay
from modules.solving_algorithm.norvig_solver import NorvigSolver
from modules.solving_algorithm.sudoku_converter import SudokuConverter
from modules.sudoku_image_pipeline.sudoku_pipeline import SudokuPipeline
from modules.utils.user_data_collector import UserDataCollector
from modules.utils.session_utils import (
    store_digit_images,
    load_digit_images,
    store_board,
    load_board,
    store_filename,
    load_filename,
)
from modules.utils.test_data_manager import save_test_board


# Initialize Blueprint
sudoku_bp = Blueprint("sudoku_solver", __name__)

# Ensure the uploads folder exists
os.makedirs("uploads", exist_ok=True)

# Initialize ImageCollector instance
image_collector = ImageCollector()


@sudoku_bp.route("/")
def home() -> str:
    """
    Clears the session and resets the image collector, then renders the home page.

    Returns:
        str: The rendered HTML template for the home page.
    """
    flashes = get_flashed_messages(with_categories=True)
    session.clear()
    image_collector.reset()
    return render_template("index.html", flashes=flashes)


@sudoku_bp.route("/process-sudoku-image", methods=["POST"])
def process_sudoku_image() -> str or Response:
    """
    Handles the uploaded file, processes the Sudoku image, recognizes digits,
    solves the Sudoku, and displays the solution or an error message.

    Returns:
        str: The rendered HTML template with the solution or an error message.
    """

    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        return redirect(request.url)

    if file:

        try:
            # Step 1: Process the image and extract the 2D list of digit images
            sudoku_pipeline = SudokuPipeline(
                image_file=file, image_collector=image_collector
            )
            preprocessed_digit_images = sudoku_pipeline.process_sudoku_image()

            # Step 2: Categorize digit images into actual numbers
            recognizer = SudokuDigitRecognizer(model_path="models/currently_used/")
            unsolved_board = recognizer.convert_cells_to_digits(
                extracted_cells=preprocessed_digit_images
            )

            # Create an image of the unsolved board
            sudoku_board_display = SudokuBoardDisplay(image_collector=image_collector)
            sudoku_board_display.draw_unsolved_board(board=unsolved_board)

            # Convert the 2D digit board to a Sudoku grid string
            converter = SudokuConverter()
            sudoku_grid = converter.board_to_string(digit_board=unsolved_board)

            # Step 3: Solve the Sudoku puzzle
            solver = NorvigSolver()
            solved_grid = solver.solve(grid=sudoku_grid)

            if not solved_grid:
                # debug:
                # image_collector.display_images_in_grid()
                # image_collector.save_images()

                # Store data in the session
                digit_images = image_collector.digit_cells
                store_digit_images(session=session, digit_images=digit_images)
                store_board(session=session, board=unsolved_board)
                store_filename(session=session, filename=file.filename)

                return render_template("no_solution.html")

            # Convert the solved Sudoku string back to a 2D digit board
            solved_board = converter.dict_to_board(sudoku_dict=solved_grid)

            # Step 4: Create an image of the solved board
            sudoku_board_display.draw_solved_board(
                unsolved_board=unsolved_board, solved_board=solved_board
            )

            # debug:
            # image_collector.display_images_in_grid()
            # image_collector.save_images()

            # Render the solution page
            return render_template("solution.html")

        except Exception as e:
            return f"Error processing image: {str(e)}"


@sudoku_bp.route("/uploads/<filename>")
def uploaded_file(filename: str) -> Response:
    """
    Serves an uploaded file from the 'uploads' directory.

    Parameters:
        filename (str): The name of the file to serve.

    Returns:
        Response: The file served from the 'uploads' directory.
    """
    return send_from_directory("uploads", filename)


@sudoku_bp.route("/debug-image/<step_name>")
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

    return send_file(image_bytes, mimetype="image/jpeg")


@sudoku_bp.route("/handle-collect-decision", methods=["POST"])
def handle_collect_decision() -> str or Response:
    """
    Handles the user's decision to either collect data or go back to the home page.
    """
    if request.form.get("collect_data") == "YES":
        try:
            unsolved_board = load_board(session=session)
        except (KeyError, TypeError, ValueError):
            return "Error: No unsolved board found.", 400

        return render_template("collect_user_data.html", sudoku_grid=unsolved_board)

    return redirect("/")


@sudoku_bp.route("/correct-and-solve", methods=["POST"])
def correct_and_solve() -> str or Response:
    """
    Collects user-provided labels for the Sudoku puzzle, saves the labeled data,
    and attempts to solve the puzzle using the corrected input.

    Returns:
        str: Redirect to solution page or error message.
    """
    try:
        # Step 1: Get labels and convert to integers
        labels = request.form.getlist("cell")
        labels_81 = [
            int(label.strip()) if label.strip().isdigit() else 0 for label in labels
        ]
        labels_no_zeros = [num for num in labels_81 if num != 0]

        # Step 2: Load and validate digit images
        digit_images = load_digit_images(session)
        collector = UserDataCollector()
        collector.validate_labels(digit_images, labels_no_zeros)

        # Step 3: Save labeled data
        collector.save_labeled_data(digit_images, labels_no_zeros)

        # Step 4: Build corrected board from labels (9x9)
        converter = SudokuConverter()
        corrected_board = converter.labels_to_board(labels_81)

        # Optional: Save data for EWA test
        filename = load_filename(session=session)
        save_test_board(filename=filename, board=corrected_board, save=False)

        # Step 5: Convert to solver grid format
        sudoku_grid = converter.board_to_string(corrected_board)

        # Step 6: Solve the puzzle
        solver = NorvigSolver()
        solved_grid = solver.solve(grid=sudoku_grid)

        if not solved_grid:
            flash(
                "Sudoku still cannot be solved with the corrected input. Please try another puzzle.",
                "warning",
            )
            return redirect("/")

        # Step 7: Display solution
        display = SudokuBoardDisplay(image_collector=image_collector)
        solved_board = converter.dict_to_board(sudoku_dict=solved_grid)
        display.draw_unsolved_board(board=corrected_board)
        display.draw_solved_board(
            unsolved_board=corrected_board, solved_board=solved_board
        )

        # Step 8: Cleanup
        session.clear()

        return render_template("solution.html")

    except ValueError as ve:
        return str(ve), 400
    except Exception as e:
        return f"Unexpected error: {str(e)}", 500
