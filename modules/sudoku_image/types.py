# modules/sudoku_image/types.py
from typing import List, Optional
import numpy as np

DigitImage = Optional[np.ndarray]
DigitRow = List[DigitImage]
DigitGrid = List[DigitRow]
"""
DigitGrid represents a 9x9 Sudoku grid.
Each cell may either be:
- a NumPy array of shape (32, 32) representing a digit image
- or `None` for an empty cell.
"""

ProcessedDigitImage = Optional[np.ndarray]
ProcessedDigitRow = List[ProcessedDigitImage]
ProcessedDigitGrid = List[ProcessedDigitRow]
"""
ProcessedDigitGrid represents a 9x9 Sudoku grid of model-ready digit images.
Each cell may either be:
- a NumPy array of shape (1, 32, 32, 1) representing a preprocessed digit image
- or `None` for an empty cell.
"""
