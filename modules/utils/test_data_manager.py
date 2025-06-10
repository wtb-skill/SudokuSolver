# modules/test_data_manager.py

import json
from pathlib import Path
import numpy as np


def save_test_board(filename: str, board: np.ndarray, save=False):
    if save:
        project_root = Path(__file__).resolve().parent.parent
        test_json_path = project_root / "dev_tools" / "model_utils" / "test.json"
        board_list = board.tolist()
        test_json_path.parent.mkdir(parents=True, exist_ok=True)

        if test_json_path.exists():
            with open(test_json_path, "r") as f:
                test_data = json.load(f)
        else:
            test_data = {}

        if filename not in test_data:
            test_data[filename] = board_list
            with open(test_json_path, "w") as f:
                json.dump(test_data, f, indent=4)
