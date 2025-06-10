# modules/user_data_collector.py
import os
import cv2
import numpy as np
from uuid import uuid4
from typing import List


class UserDataCollector:
    def __init__(self, label_folder: str = "collected_data"):
        """
        Initializes the UserDataCollector class with the given folder path for storing user data in case algorithm
        found no solution for Sudoku puzzle.

        Parameters:
            label_folder (str): The folder where the labeled images will be stored. Defaults to 'collected_data'.
        """
        self.label_folder = label_folder

    @staticmethod
    def validate_labels(digit_images: List[np.ndarray], labels: List[int]) -> None:
        """
        Validates that the number of digit images matches the number of labels.

        Parameters:
            digit_images (List[np.ndarray]): A list of images representing digits.
            labels (List[int]): A list of labels corresponding to the images.

        Raises:
            ValueError: If the number of images does not match the number of labels.
        """
        if len(digit_images) != len(labels):
            raise ValueError(
                f"Mismatch: {len(digit_images)} images but {len(labels)} labels"
            )

    def save_labeled_data(
        self, digit_images: List[np.ndarray], labels: List[int]
    ) -> None:
        """
        Saves labeled digit images to disk, organizing them into subfolders based on the labels.

        Parameters:
            digit_images (List[np.ndarray]): A list of images representing digits.
            labels (List[int]): A list of labels corresponding to the images.

        Creates:
            Directories under `self.label_folder` for each label, and saves each image as a `.png` file within
            the corresponding label's folder.
        """
        for img, label in zip(digit_images, labels):
            label_dir = os.path.join(
                self.label_folder, str(label)
            )  # Ensure label is a string for directory
            os.makedirs(label_dir, exist_ok=True)

            filename = f"{uuid4().hex}.png"
            filepath = os.path.join(label_dir, filename)

            cv2.imwrite(filepath, img)
