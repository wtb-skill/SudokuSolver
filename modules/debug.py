# modules/debug.py
import os
import cv2
import io
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging


# Create a logger for this module
logger = logging.getLogger(__name__)


class ImageCollector:
    """
    Handles storing, displaying, and saving intermediate images during the Sudoku image processing pipeline.
    Useful for visualizing processing stages such as preprocessing, digit detection, and final output.
    """

    def __init__(self, output_dir: str = "debug_images") -> None:
        """
        Initializes the image collector, creating an output directory if it doesn't exist.

        Args:
            output_dir (str): Directory where debug images will be saved.
            logging_enabled (bool): Whether to print debug messages during collection.
        """
        self.output_dir: str = output_dir
        self.images: Dict[str, np.ndarray] = {}
        self.grid_image: Optional[np.ndarray] = None
        self.digit_cells: List[np.ndarray] = []

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def add_image(self, step_name: str, image: np.ndarray) -> None:
        """
        Store a debug image under a descriptive step name.

        Args:
            step_name (str): Descriptive name for the image.
            image (np.ndarray): Image to store.
        """
        self.images[step_name] = image

    def save_images(self) -> None:
        """
        Save all collected images to the output directory.
        """
        for step, img in self.images.items():
            cv2.imwrite(os.path.join(self.output_dir, f"{step}.png"), img)

        if self.grid_image is not None:
            cv2.imwrite(
                os.path.join(self.output_dir, "grid_image.png"), self.grid_image
            )

    def show_images(self) -> None:
        """
        Display all collected images one at a time in OpenCV windows.
        """
        for step, img in self.images.items():
            cv2.imshow(step, img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def resize_images(self, width: int = 450, height: int = 450) -> None:
        """
        Resize all stored images to a consistent size.

        Args:
            width (int): Target width.
            height (int): Target height.
        """
        for step, img in self.images.items():
            self.images[step] = cv2.resize(img, (width, height))

    def display_images_in_grid(self, max_images_per_row: int = 4) -> None:
        """
        Display all stored debug images in a single grid image with labels.

        Args:
            max_images_per_row (int): Number of images per row in the grid.
        """
        self.resize_images(450, 450)

        image_list: List[Tuple[str, np.ndarray]] = [
            (name, img) for name, img in self.images.items()
        ]

        # Convert grayscale to BGR for consistent channel depth
        for i, (name, img) in enumerate(image_list):
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            image_list[i] = (name, img)

        num_images = len(image_list)
        num_rows = (num_images + max_images_per_row - 1) // max_images_per_row

        rows: List[np.ndarray] = []

        for i in range(num_rows):
            row_images = image_list[
                i * max_images_per_row : (i + 1) * max_images_per_row
            ]

            # Pad row if needed
            while len(row_images) < max_images_per_row:
                blank = np.zeros_like(row_images[0][1])
                row_images.append(("Empty", blank))

            labeled_images: List[np.ndarray] = []
            for name, img in row_images:
                label_area = np.zeros((50, img.shape[1], 3), dtype="uint8")
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale, thickness = 1, 2
                text_size = cv2.getTextSize(name, font, scale, thickness)[0]
                text_x = (img.shape[1] - text_size[0]) // 2
                text_y = (50 + text_size[1]) // 2
                cv2.putText(
                    label_area,
                    name,
                    (text_x, text_y),
                    font,
                    scale,
                    (255, 255, 255),
                    thickness,
                )
                labeled_images.append(np.vstack([label_area, img]))

            rows.append(np.hstack(labeled_images))

        self.grid_image = np.vstack(rows)

        cv2.imshow("Debug Images", self.grid_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_image_bytes(self, step_name: str) -> Optional[io.BytesIO]:
        """
        Retrieve an image as a byte stream for web serving (e.g., with Flask).

        Args:
            step_name (str): Step name used when the image was added.

        Returns:
            Optional[io.BytesIO]: JPEG-encoded image stream or None if not found.
        """
        img = self.images.get(step_name)
        if img is None:
            return None

        _, img_encoded = cv2.imencode(".jpg", img)
        return io.BytesIO(img_encoded.tobytes())

    def collect_digit_cells(self, digits: List[List[Optional[np.ndarray]]]) -> None:
        """
        Collect all non-empty digit cell images (e.g., 32x32 grayscale) for training or inspection.

        Args:
            digits (List[List[Optional[np.ndarray]]]): 2D list of digit images or None.
        """
        for row in digits:
            for cell in row:
                if cell is not None:
                    if cell.shape != (32, 32):
                        logger.info(
                            f"[Warning] Invalid image shape: expected (32, 32), got {cell.shape}"
                        )
                    else:
                        self.digit_cells.append(cell.copy())

        if self.digit_cells:
            logger.info(f"Collected {len(self.digit_cells)} digit cell(s).")
        else:
            logger.info("No valid digit cells were collected.")

    def reset(self) -> None:
        """
        Clear all collected digit cells.
        """
        self.digit_cells.clear()
        self.images.clear()
