# modules/user_data_collector.py

import os
import cv2
import pickle
from uuid import uuid4

class UserDataCollector:
    def __init__(self, label_folder='collected_data'):
        self.label_folder = label_folder

    def load_digit_images_from_session(self, session):
        pickled_data = session.pop('digit_images', None)
        if pickled_data is None:
            raise ValueError("Missing digit images in session")
        return pickle.loads(pickled_data)

    def validate_labels(self, digit_images, labels):
        if len(digit_images) != len(labels):
            raise ValueError(f"Mismatch: {len(digit_images)} images but {len(labels)} labels")

    def save_labeled_data(self, digit_images, labels):
        for img, label in zip(digit_images, labels):
            label_dir = os.path.join(self.label_folder, label)
            os.makedirs(label_dir, exist_ok=True)

            filename = f"{uuid4().hex}.png"
            filepath = os.path.join(label_dir, filename)

            cv2.imwrite(filepath, img)
