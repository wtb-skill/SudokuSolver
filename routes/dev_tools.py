# dev_tools.py
import os
import shutil
import json
import logging
from tqdm import tqdm
from flask import Blueprint
from modules.digit_recognition import SudokuDigitRecognizer
from modules.debug import ImageCollector
from modules.solving_algorithm.norvig_solver import NorvigSolver
from modules.solving_algorithm.sudoku_converter import SudokuConverter
from modules.sudoku_image_pipeline.sudoku_pipeline import SudokuPipeline
from modules.board_display import SudokuBoardDisplay


# Initialize Blueprint
dev_tools_bp = Blueprint('dev_tools', __name__, url_prefix='/dev')

image_collector = ImageCollector()

# Create a logger for this module
logger = logging.getLogger(__name__)

@dev_tools_bp.route('/process-all-sudoku-images', methods=['GET'])
def process_all_sudoku_images():
    """
    Automatically processes all Sudoku images in the 'uploads/' folder,
    attempting to solve each one and move the result to either 'solved/' or 'unsolved/' folder.
    """

    upload_dir = 'uploads'
    solved_dir = 'uploads/solved'
    unsolved_dir = 'uploads/unsolved'

    # Create the necessary directories if they don't exist
    os.makedirs(solved_dir, exist_ok=True)
    os.makedirs(unsolved_dir, exist_ok=True)

    # List all files in the 'uploads' folder
    files = [f for f in os.listdir(upload_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Process each image file
    for file_name in files:
        try:
            file_path = os.path.join(upload_dir, file_name)

            # Step 1: Process the image and extract the 2D list of digit images
            sudoku_pipeline = SudokuPipeline(image_file=file_path, image_collector=image_collector)
            preprocessed_digit_images = sudoku_pipeline.process_sudoku_image()

            # Step 2: Categorize digit images into actual numbers
            recognizer = SudokuDigitRecognizer(model_path="models/sudoku_digit_recognizer.keras")
            unsolved_board = recognizer.convert_cells_to_digits(extracted_cells=preprocessed_digit_images)

            # Convert the 2D digit board to a Sudoku grid string
            converter = SudokuConverter()
            sudoku_grid = converter.board_to_string(digit_board=unsolved_board)

            # Step 3: Solve the Sudoku puzzle
            solver = NorvigSolver()
            solved_grid = solver.solve(grid=sudoku_grid)

            if solved_grid:
                # If solved, move to the 'solved' folder
                solved_board = converter.dict_to_board(sudoku_dict=solved_grid)
                sudoku_board_display = SudokuBoardDisplay(image_collector=image_collector)
                sudoku_board_display.draw_solved_board(unsolved_board=unsolved_board, solved_board=solved_board)

                # Move the file to the 'solved' folder
                solved_file_path = os.path.join(solved_dir, file_name)
                shutil.move(file_path, solved_file_path)

            else:
                # If not solved, move to the 'unsolved' folder
                unsolved_file_path = os.path.join(unsolved_dir, file_name)
                shutil.move(file_path, unsolved_file_path)

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

    # Return a message when all images are processed
    return "All images have been processed!"

@dev_tools_bp.route('/process-test-dataset', methods=['GET'])
def process_test_dataset():
    """
    Processes all new Sudoku images in 'uploads/clean/', extracts the unsolved board for each,
    and saves the results in 'test.json' with format {filename: unsolved_board}.
    """

    import os
    import json

    clean_dir = 'uploads/clean'
    os.makedirs(clean_dir, exist_ok=True)

    # Load existing data if test.json exists
    test_data = {}
    if os.path.exists('test.json'):
        with open('test.json', 'r') as json_file:
            test_data = json.load(json_file)

    files = [f for f in os.listdir(clean_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for file_name in files:
        if file_name in test_data:
            # Skip files that are already processed
            continue

        try:
            file_path = os.path.join(clean_dir, file_name)

            # Step 1: Process the image and extract the 2D list of digit images
            sudoku_pipeline = SudokuPipeline(image_file=file_path, image_collector=image_collector)
            preprocessed_digit_images = sudoku_pipeline.process_sudoku_image()

            # Step 2: Categorize digit images into actual numbers
            recognizer = SudokuDigitRecognizer(model_path="models/sudoku_digit_recognizer.keras")
            unsolved_board = recognizer.convert_cells_to_digits(extracted_cells=preprocessed_digit_images)

            # Store the board as a list of lists (for JSON compatibility)
            test_data[file_name] = unsolved_board.tolist()

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

    # Save updated test_data to test.json
    with open('test.json', 'w') as json_file:
        json.dump(test_data, json_file, indent=4)

    return "Test dataset processed and saved to test.json!"

@dev_tools_bp.route('/move-skipped-files', methods=['POST'])
def move_skipped_files():
    """
    Moves images from 'uploads/test_folder' that are NOT in test.json
    to the parent 'uploads/' directory.
    """
    import os
    import json
    import shutil

    clean_dir = 'uploads/test_folder'
    uploads_dir = 'uploads'
    test_json_path = 'dev_tools/model_utils/test.json'

    if not os.path.exists(test_json_path):
        return "test.json not found. Please generate it first.", 400

    # Load ground truth
    with open(test_json_path, 'r') as f:
        ground_truth_data = json.load(f)

    files = [f for f in os.listdir(clean_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    moved_files = []

    for file_name in files:
        if file_name not in ground_truth_data:
            source_path = os.path.join(clean_dir, file_name)
            destination_path = os.path.join(uploads_dir, file_name)

            # Move the file
            shutil.move(source_path, destination_path)
            moved_files.append(file_name)

    if moved_files:
        return f"Moved {len(moved_files)} skipped files: {', '.join(moved_files)}"
    else:
        return "No skipped files found to move."

@dev_tools_bp.route('/evaluate-recognition-accuracy', methods=['GET'])
def evaluate_recognition_accuracy():
    """
    Evaluates the digit recognition accuracy by comparing predicted unsolved boards
    with the ground truth from test.json for images in 'uploads/test_folder/'.
    Only nonzero (non-empty) cells are considered.
    """
    # Disable logs
    logging.getLogger('modules.sudoku_image_pipeline.step_2_board_detector').setLevel(logging.WARNING)
    logging.getLogger('modules.sudoku_image_pipeline.sudoku_pipeline').setLevel(logging.WARNING)
    logging.getLogger('modules.digit_recognition').setLevel(logging.WARNING)
    logging.getLogger('modules.debug').setLevel(logging.WARNING)

    clean_dir = 'uploads/test_folder'
    test_json_path = 'dev_tools/model_utils/test.json'

    if not os.path.exists(test_json_path):
        return "test.json not found. Please generate it first.", 400

    # Load ground truth
    with open(test_json_path, 'r') as f:
        ground_truth_data = json.load(f)

    files = [f for f in os.listdir(clean_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    total_digits = 0
    correct_digits = 0

    results = []

    # Wrap with tqdm to add a progress bar
    for file_name in tqdm(files, desc="Processing files", unit="file"):
        if file_name not in ground_truth_data:
            logger.info(f"Skipping {file_name}: no ground truth available.")
            continue

        try:
            file_path = os.path.join(clean_dir, file_name)

            # Process the image
            sudoku_pipeline = SudokuPipeline(image_file=file_path, image_collector=image_collector)
            preprocessed_digit_images = sudoku_pipeline.process_sudoku_image()

            # Recognize digits
            recognizer = SudokuDigitRecognizer(model_path="models/sudoku_digit_recognizer.keras")
            predicted_board = recognizer.convert_cells_to_digits(extracted_cells=preprocessed_digit_images)
            predicted_board_list = predicted_board.tolist()

            # Ground truth
            true_board_list = ground_truth_data[file_name]

            # Compare per image
            image_total_digits = 0
            image_correct_digits = 0

            for i in range(9):
                for j in range(9):
                    true_digit = true_board_list[i][j]
                    pred_digit = predicted_board_list[i][j]
                    if true_digit != 0:
                        image_total_digits += 1
                        if true_digit == pred_digit:
                            image_correct_digits += 1

            # Update global stats
            total_digits += image_total_digits
            correct_digits += image_correct_digits

            # Per image stats
            image_accuracy = round(100 * image_correct_digits / image_total_digits, 2) if image_total_digits else 0
            results.append({
                'file': file_name,
                'accuracy': image_accuracy
            })

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

    # After all files
    overall_accuracy = round(100 * correct_digits / total_digits, 2) if total_digits else 0
    logger.info(f"\nOverall accuracy: {overall_accuracy}% based on {total_digits} digits.")

    return "Evaluation completed. Check server console output!"
