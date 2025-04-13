# modules/sudoku_image/step_1_image_preprocessor.py
import cv2
import imutils
import numpy as np
from typing import Optional
from modules.debug import ImageCollector


class ImagePreprocessor:
    """
    Preprocesses a Sudoku image for board detection and digit extraction.

    Steps:
    - Resize the image for consistent processing
    - Convert to grayscale
    - Apply Gaussian blur
    - Apply adaptive thresholding

    Attributes:
        original_image (np.ndarray): Resized original image (BGR).
        gray (Optional[np.ndarray]): Grayscale version of the image.
        thresholded (Optional[np.ndarray]): Thresholded (binary) version of the image.
        image_collector (ImageCollector): ImageCollector instance for storing images.
    """

    def __init__(self, image: np.ndarray, image_collector: ImageCollector):
        """
        Initialize with a BGR image and image collector (must always be provided).

        Args:
            image (np.ndarray): The input BGR image.
            image_collector (ImageCollector): ImageCollector object (used for storing images).
        """
        self.image_collector: ImageCollector = image_collector
        self.original_image: np.ndarray = imutils.resize(image, width=600)
        self.gray: Optional[np.ndarray] = None
        self.thresholded: Optional[np.ndarray] = None

        self.image_collector.add_image("Original_Image", self.original_image)

    def preprocess(self) -> np.ndarray:
        """
        Perform grayscale conversion, blurring, and adaptive thresholding.

        Returns:
            np.ndarray: The thresholded binary image.
        """
        self.gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(self.gray, (7, 7), 3)
        self.thresholded = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        self.image_collector.add_image("Thresholded_Edges", self.thresholded)

        return self.thresholded

    def get_original(self) -> np.ndarray:
        """
        Get the resized original image.

        Returns:
            np.ndarray: The resized BGR image.
        """
        return self.original_image

    def get_gray(self) -> Optional[np.ndarray]:
        """
        Get the grayscale image, if computed.

        Returns:
            Optional[np.ndarray]: The grayscale image, or None if not yet computed.
        """
        return self.gray

    def get_thresholded(self) -> Optional[np.ndarray]:
        """
        Get the thresholded image, if computed.

        Returns:
            Optional[np.ndarray]: The binary thresholded image, or None if not yet computed.
        """
        return self.thresholded

