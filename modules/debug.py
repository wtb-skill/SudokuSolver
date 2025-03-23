#modules/debug.py
import cv2
import os
import numpy as np

class DebugVisualizer:
    def __init__(self, output_dir: str = "debug_images"):
        """
        Handles saving and displaying debug images at different stages.
        """
        self.output_dir = output_dir
        self.images = {}  # Dictionary to store step-name -> image

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def add_image(self, step_name: str, image: np.ndarray):
        """
        Store an image under a specific step name.
        """
        self.images[step_name] = image

    def save_images(self):
        """
        Save all stored debug images to the output directory.
        """
        for step, img in self.images.items():
            cv2.imwrite(os.path.join(self.output_dir, f"{step}.png"), img)

    def show_images(self):
        """
        Display all debug images sequentially.
        """
        for step, img in self.images.items():
            cv2.imshow(step, img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
