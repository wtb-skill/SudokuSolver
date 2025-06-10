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
    - Apply optional brightness normalization
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

    def preprocess(self, normalize: bool = True) -> np.ndarray:
        """
        Perform grayscale conversion, optional brightness normalization,
        blurring, and adaptive thresholding.

        Args:
            normalize (bool): Whether to perform brightness normalization.

        Returns:
            np.ndarray: The thresholded binary image.
        """
        self.gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(self.gray, (7, 7), 3)

        # check if it's worth keeping normalize
        processed = blurred
        # if normalize and (self.needs_normalization(blurred) or self.is_low_contrast(blurred)):
        #     processed = self.normalize_brightness(gray=blurred)
        # else:
        #     processed = blurred

        self.thresholded = cv2.adaptiveThreshold(
            processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
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

    def normalize_brightness(self, gray: np.ndarray, dsize: int = 11) -> np.ndarray:
        close = cv2.morphologyEx(
            gray,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dsize, dsize)),
        )
        div = (np.uint16(gray) * 256) / np.uint16(close)
        normalized = np.uint8(cv2.normalize(div, None, 0, 255, cv2.NORM_MINMAX))
        self.image_collector.add_image("Normalized_Brightness", normalized)
        return normalized

    def needs_normalization(
        self, gray: np.ndarray, tile_size: int = 50, std_threshold: float = 20.0
    ) -> bool:
        """
        Determine if local brightness variation suggests uneven lighting.

        Args:
            gray (np.ndarray): Grayscale image.
            tile_size (int): Size of grid cells to compute local variation.
            std_threshold (float): Threshold for variation across tiles.

        Returns:
            bool: True if normalization might help.
        """
        h, w = gray.shape
        stds = []

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                tile = gray[y : min(y + tile_size, h), x : min(x + tile_size, w)]
                stds.append(np.std(tile))

        variation = np.std(stds)
        # self.image_collector.add_debug_info("Std_Variation_Across_Tiles", variation)
        return variation > std_threshold

    def is_low_contrast(self, gray: np.ndarray, threshold: float = 0.15) -> bool:
        """
        Estimate if the image has low contrast and might benefit from normalization.

        Args:
            gray (np.ndarray): Grayscale image.
            threshold (float): Minimum spread of histogram considered as good contrast.

        Returns:
            bool: True if the image contrast is low.
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        spread = np.sum(hist[10:245])
        # self.image_collector.add_debug_info("Histogram_Spread", float(spread))
        return spread < threshold
