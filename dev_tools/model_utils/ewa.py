import os
import json
import logging
from tqdm import tqdm
from modules.digit_recognition import SudokuDigitRecognizer
from modules.sudoku_image_pipeline.sudoku_pipeline import SudokuPipeline
from modules.debug import ImageCollector
from datetime import datetime
import shutil

image_collector = ImageCollector()


class RecognitionAccuracyEvaluator:
    def __init__(self, clean_dir='uploads/test_folder', model_dir='models/currently_used/',
                 test_json_path='dev_tools/model_utils/test.json'):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.clean_dir = os.path.join(project_root, clean_dir)
        self.model_dir = os.path.join(project_root, model_dir)
        self.test_json_path = os.path.join(project_root, test_json_path)
        self.total_digits = 0
        self.correct_digits = 0
        self.results = []
        self.model_name = None
        self._setup_logging()

    def _setup_logging(self):
        logging.getLogger('modules.sudoku_image_pipeline.step_2_board_detector').setLevel(logging.WARNING)
        logging.getLogger('modules.sudoku_image_pipeline.sudoku_pipeline').setLevel(logging.WARNING)
        logging.getLogger('modules.digit_recognition').setLevel(logging.WARNING)
        logging.getLogger('modules.debug').setLevel(logging.WARNING)

    def load_ground_truth(self):
        if not os.path.exists(self.test_json_path):
            raise FileNotFoundError("test.json not found. Please generate it first.")
        with open(self.test_json_path, 'r') as f:
            return json.load(f)

    def log_result(self, message):
        with open('ewa_results.txt', 'a') as log_file:
            log_file.write(f"{datetime.now()}: {message}\n")

    def evaluate(self):
        ground_truth_data = self.load_ground_truth()
        files = [f for f in os.listdir(self.clean_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        with tqdm(total=len(files), desc="Evaluating Sudoku Images", unit="image") as pbar:
            for file_name in files:
                if file_name not in ground_truth_data:
                    logging.info(f"Skipping {file_name}: no ground truth available.")
                    pbar.update(1)
                    continue
                try:
                    file_path = os.path.join(self.clean_dir, file_name)
                    sudoku_pipeline = SudokuPipeline(image_file=file_path, image_collector=image_collector)
                    preprocessed_digit_images = sudoku_pipeline.process_sudoku_image()
                    recognizer = SudokuDigitRecognizer(model_path=self.model_dir)
                    predicted_board = recognizer.convert_cells_to_digits(extracted_cells=preprocessed_digit_images)
                    predicted_board_list = predicted_board.tolist()
                    true_board_list = ground_truth_data[file_name]

                    image_total_digits, image_correct_digits = 0, 0
                    for i in range(9):
                        for j in range(9):
                            true_digit = true_board_list[i][j]
                            pred_digit = predicted_board_list[i][j]
                            if true_digit != 0:
                                image_total_digits += 1
                                if true_digit == pred_digit:
                                    image_correct_digits += 1

                    self.total_digits += image_total_digits
                    self.correct_digits += image_correct_digits
                    image_accuracy = round(100 * image_correct_digits / image_total_digits, 2) if image_total_digits else 0
                    self.results.append({'file': file_name, 'accuracy': image_accuracy})
                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")
                pbar.update(1)

        overall_accuracy = round(100 * self.correct_digits / self.total_digits, 2) if self.total_digits else 0
        result_message = f"Overall accuracy for model {self.model_name}: {overall_accuracy}% based on {self.total_digits} digits."
        print(result_message)
        self.log_result(result_message)


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_path = os.path.join(project_root, 'models')
    current_model_path = os.path.join(models_path, 'currently_used')
    all_versions_path = os.path.join(models_path, 'all_versions')

    os.makedirs(current_model_path, exist_ok=True)
    os.makedirs(all_versions_path, exist_ok=True)

    model_files = [f for f in os.listdir(models_path) if f.endswith('.keras')]

    for model in model_files:
        shutil.move(os.path.join(models_path, model), os.path.join(current_model_path, model))
        evaluator = RecognitionAccuracyEvaluator()
        evaluator.model_name = model
        evaluator.evaluate()
        shutil.move(os.path.join(current_model_path, model), os.path.join(all_versions_path, model))

    print("[INFO] All models evaluated.")
