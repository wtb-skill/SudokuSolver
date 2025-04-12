# modules/user_data_collector.py
import os
import cv2
import pickle
import numpy as np
from uuid import uuid4
from typing import List
from flask.sessions import SessionMixin

class UserDataCollector:
    def __init__(self, label_folder: str = 'collected_data'):
        """
        Initializes the UserDataCollector class with the given folder path for storing user data in case algorithm
        found no solution for Sudoku puzzle.

        Parameters:
            label_folder (str): The folder where the labeled images will be stored. Defaults to 'collected_data'.
        """
        self.label_folder = label_folder

    @staticmethod
    def load_digit_images_from_session(session: SessionMixin) -> List[np.ndarray]:
        """
        Loads digit images from a session object.

        Parameters:
            session (dict): A dictionary containing session data, where 'digit_images' is expected to be pickled.

        Returns:
            List[np.ndarray]: A list of images (as numpy arrays) extracted from the session.

        Raises:
            ValueError: If 'digit_images' is not found in the session or the data is invalid.
        """
        pickled_data = session.pop('digit_images', None)
        if pickled_data is None:
            raise ValueError("Missing digit images in session")

        # Ensure that pickled_data is a valid bytes-like object
        if not isinstance(pickled_data, (bytes, bytearray)):
            raise ValueError("Invalid data format for 'digit_images', expected a bytes-like object")

        try:
            return pickle.loads(pickled_data)
        except pickle.UnpicklingError as e:
            raise ValueError("Failed to unpickle 'digit_images'") from e

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
            raise ValueError(f"Mismatch: {len(digit_images)} images but {len(labels)} labels")

    def save_labeled_data(self, digit_images: List[np.ndarray], labels: List[int]) -> None:
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
            label_dir = os.path.join(self.label_folder, str(label))  # Ensure label is a string for directory
            os.makedirs(label_dir, exist_ok=True)

            filename = f"{uuid4().hex}.png"
            filepath = os.path.join(label_dir, filename)

            cv2.imwrite(filepath, img)

