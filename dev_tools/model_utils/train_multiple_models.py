from pathlib import Path
from dev_tools.model_utils.sudokunet_trainer import SudokunetTrainer
import os

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent.parent
    data_utils_path = project_root / "dev_tools" / "data_utils"
    model_output_dir = project_root / "models"

    for folder in os.listdir(data_utils_path):
        if folder.startswith("digit_dataset_"):
            dataset_path = str(data_utils_path / folder)
            version = folder.split("v", 1)[-1]  # Split past the letter 'v'
            model_output_path = str(model_output_dir / f"v{version}.keras")

            trainer = SudokunetTrainer(
                dataset_path=dataset_path,
                model_output_path=model_output_path,
                image_size=32,
                init_lr=1e-3,
                epochs=10,
                batch_size=128
            )
            print(f"Training with dataset: {folder} -> Model: v{version}.keras")
            trainer.run()

    print("Training completed for all datasets.")
