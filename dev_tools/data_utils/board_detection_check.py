import cv2
import os
import shutil
from pathlib import Path
from modules.debug import ImageCollector
from modules.sudoku_image_pipeline.step_1_image_preprocessor import ImagePreprocessor
from modules.sudoku_image_pipeline.step_2_board_detector import BoardDetector


class SudokuImageProcessor:
    def __init__(self, input_folder, ok_folder) -> None:
        self.input_folder = input_folder
        self.ok_folder = ok_folder
        self.image_collector = ImageCollector()

        os.makedirs(self.ok_folder, exist_ok=True)

    def process_images(self) -> None:
        for filename in os.listdir(self.input_folder):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                print(f"Processing {filename}...")
                img_path = os.path.join(self.input_folder, filename)

                try:
                    # Step 1: Image Preprocessing
                    original_image = cv2.imread(img_path)
                    preprocessor = ImagePreprocessor(original_image, self.image_collector)
                    thresholded = preprocessor.preprocess()

                    # Step 2: Board Detection
                    board_detector = BoardDetector(preprocessor.get_original(), thresholded, self.image_collector)
                    warped_board = board_detector.detect()

                    # If both steps succeed, move the file to the 'ok' folder
                    shutil.move(img_path, os.path.join(self.ok_folder, filename))
                    print(f"{filename} successfully processed and moved.")

                except Exception as e:
                    print(f"Failed to process {filename}: {e}")
                finally:
                    # Clear stored images for next run
                    self.image_collector.reset()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    input_folder = project_root / "uploads" / "very_noisy"
    ok_folder = input_folder / "ok"
    processor = SudokuImageProcessor(input_folder, ok_folder)
    processor.process_images()
