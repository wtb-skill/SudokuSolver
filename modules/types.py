# modules/sudoku_image/types.py
from typing import List, Optional
import numpy as np

# Raw grayscale image of a digit: shape (32, 32), dtype 'uint8'
RawDigitImage = np.ndarray

# Model-ready image of a digit: shape (1, 32, 32, 1), dtype 'float32'
ModelReadyDigitImage = np.ndarray

DigitImage = Optional[RawDigitImage]
DigitRow = List[DigitImage]
DigitGrid = List[DigitRow]
"""
DigitGrid represents a 9x9 Sudoku grid of raw digit images.

Each cell may either be:
- a NumPy array of shape (32, 32) and dtype 'uint8' representing a grayscale digit image
- or `None` for an empty cell (i.e. no digit was detected).
"""

ProcessedDigitImage = Optional[ModelReadyDigitImage]
ProcessedDigitRow = List[ProcessedDigitImage]
ProcessedDigitGrid = List[ProcessedDigitRow]
"""
ProcessedDigitGrid represents a 9x9 Sudoku grid of preprocessed, model-ready digit images.

Each cell may either be:
- a NumPy array of shape (1, 32, 32, 1) and dtype 'float32', normalized to [0, 1]
  (format: batch, height, width, channels — suitable for TensorFlow/Keras)
- or `None` for an empty cell.
"""

__all__ = [
    "DigitImage", "DigitRow", "DigitGrid",
    "ProcessedDigitImage", "ProcessedDigitRow", "ProcessedDigitGrid",
]
