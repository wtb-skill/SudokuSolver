# modules/board_detector.py
import cv2
import imutils
import numpy as np
from imutils.perspective import four_point_transform
from typing import Optional
from modules.debug import ImageCollector

class BoardDetector:
    """
    Detects the Sudoku board in an image, including identifying contours, detecting the Sudoku grid,
    and warping the board to a top-down perspective.

    Attributes:
        original_image (np.ndarray): The original input image (BGR).
        thresholded (np.ndarray): The thresholded (binary) image used for contour detection.
        warped (Optional[np.ndarray]): The warped top-down perspective of the Sudoku board (optional).
        image_collector (ImageCollector): An instance of ImageCollector for storing debug images.
    """

    def __init__(self, original_image: np.ndarray, thresholded: np.ndarray, image_collector: ImageCollector):
        """
        Initialize the BoardDetector with the original image, thresholded image, and image collector.

        Args:
            original_image (np.ndarray): The input BGR image of the Sudoku puzzle.
            thresholded (np.ndarray): The thresholded (binary) image used for contour detection.
            image_collector (ImageCollector): The image collector instance used to store debug images.
        """
        self.original_image: np.ndarray = original_image
        self.thresholded: np.ndarray = thresholded
        self.image_collector: ImageCollector = image_collector
        self.warped: Optional[np.ndarray] = None

    def detect(self) -> np.ndarray:
        """
        Detect the Sudoku board from the thresholded image by identifying contours,
        sorting and filtering them based on size, and then applying a perspective
        transform to obtain a top-down view of the board.

        Returns:
            np.ndarray: The warped top-down perspective of the Sudoku board.
        """
        contours = cv2.findContours(self.thresholded.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        height, width = self.original_image.shape[:2]
        min_area = 0.2 * height * width
        largest_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if not largest_contours:
            raise Exception("No large enough Sudoku grid found.")

        smallest_contour = min(largest_contours, key=cv2.contourArea)
        peri = cv2.arcLength(smallest_contour, True)
        approx = cv2.approxPolyDP(smallest_contour, 0.02 * peri, True)

        if len(approx) != 4:
            raise Exception("Detected contour is not quadrilateral.")

        # Drawing the detected contour on the original image for visualization
        outline_img = self.original_image.copy()
        cv2.drawContours(outline_img, [approx], -1, (0, 255, 0), 2)
        self.image_collector.add_image("Detected_Sudoku_Outline", outline_img)

        # Perform the perspective transformation
        warped = four_point_transform(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY), approx.reshape(4, 2))
        self.warped = warped

        # Storing the warped board image
        self.image_collector.add_image("Warped_Sudoku_Board", self.warped)

        return self.warped

