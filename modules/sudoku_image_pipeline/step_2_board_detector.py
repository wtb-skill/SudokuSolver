# modules/sudoku_image/step_2_board_detector.py
import cv2
import imutils
import numpy as np
import logging
from imutils.perspective import four_point_transform
from typing import Optional
from modules.debug import ImageCollector

# Create a logger for this module
logger = logging.getLogger(__name__)


class BoardDetector:
    def __init__(
        self,
        original_image: np.ndarray,
        thresholded: np.ndarray,
        image_collector: ImageCollector,
    ):
        self.original_image = original_image
        self.thresholded = thresholded
        self.image_collector = image_collector
        self.warped: Optional[np.ndarray] = None

    def detect(self) -> np.ndarray:
        try:
            corners = self._detect_contour_corners()
            self.warped = self.warp_board(corners, label="Warped_Sudoku_Board")
        except Exception as e:
            logger.info(f"[Primary Detection Failed] {e}")
            logger.info(f"[Falling back to grid-based detection method...]")
            corners = self.detect_fallback()
            self.warped = self.warp_board(corners, label="Fallback_Warped_Sudoku_Board")
        return self.warped

    def warp_board(
        self, board_corners: np.ndarray, label: str = "Warped_Sudoku_Board"
    ) -> np.ndarray:
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        warped = four_point_transform(gray, board_corners)
        self.image_collector.add_image(label, warped)
        return warped

    def _detect_contour_corners(self) -> np.ndarray:
        contours = cv2.findContours(
            self.thresholded.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        height, width = self.original_image.shape[:2]
        min_area = 0.25 * height * width
        largest_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if not largest_contours:
            raise Exception("No large enough Sudoku grid found.")

        smallest_contour = min(largest_contours, key=cv2.contourArea)
        peri = cv2.arcLength(smallest_contour, True)
        approx = cv2.approxPolyDP(smallest_contour, 0.02 * peri, True)

        if len(approx) != 4:
            raise Exception("Detected contour is not quadrilateral.")

        outline_img = self.original_image.copy()
        cv2.drawContours(outline_img, [approx], -1, (0, 255, 0), 2)
        self.image_collector.add_image("Detected_Sudoku_Outline", outline_img)

        return approx.reshape(4, 2).astype(np.float32)

    def is_valid_board(self, corners: np.ndarray) -> bool:
        hull = cv2.convexHull(corners)
        if len(hull) != 4:
            return False
        side_lengths = [
            np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)
        ]
        ratio = max(side_lengths) / min(side_lengths)
        if ratio > 1.5:
            return False

        def is_parallel(p1, p2, p3, p4):
            vec1 = p2 - p1
            vec2 = p4 - p3
            dot_product = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1) * np.linalg.norm(vec2)
            )
            return abs(dot_product) > 0.9

        if not (
            is_parallel(corners[0], corners[1], corners[3], corners[2])
            and is_parallel(corners[0], corners[3], corners[1], corners[2])
        ):
            return False
        return True

    def detect_fallback(self) -> np.ndarray:
        debug_img = self.original_image.copy()
        edges = cv2.Canny(self.thresholded, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
        )
        if lines is None:
            raise Exception("Fallback failed: No lines detected.")

        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            points.extend([(x1, y1), (x2, y2)])

        self.image_collector.add_image("Fallback_Lines", debug_img)

        top_left = min(points, key=lambda p: p[0] + p[1])
        top_right = max(points, key=lambda p: p[0] - p[1])
        bottom_left = min(points, key=lambda p: p[0] - p[1])
        bottom_right = max(points, key=lambda p: p[0] + p[1])

        fallback_corners = np.array(
            [top_left, top_right, bottom_right, bottom_left], dtype=np.float32
        )

        if not self.is_valid_board(fallback_corners):
            raise Exception(
                "Fallback failed: Detected structure is not a valid Sudoku board."
            )

        corner_debug = debug_img.copy()
        for idx, pt in enumerate(fallback_corners):
            cv2.circle(corner_debug, tuple(pt.astype(int)), 6, (0, 0, 255), -1)
            cv2.putText(
                corner_debug,
                f"P{idx}",
                tuple(pt.astype(int) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        self.image_collector.add_image("Fallback_Corners", corner_debug)
        return fallback_corners
