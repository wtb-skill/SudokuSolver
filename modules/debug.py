#modules/debug.py
import os
import cv2
import io
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

    def resize_images(self, width: int = 450, height: int = 450):
        """
        Resize all stored images to the specified dimensions.

        Parameters:
            width (int): Target width for resizing.
            height (int): Target height for resizing.
        """
        for step, img in self.images.items():
            self.images[step] = cv2.resize(img, (width, height))

    def display_images_in_grid(self, max_images_per_row: int = 4):
        """
        Display all stored images in a single large image, arranged in a grid with labels.

        Parameters:
            max_images_per_row (int): The maximum number of images to display in one row.
        """
        # Resize all images to a consistent size
        self.resize_images(450, 450)

        image_list = [(name, img) for name, img in self.images.items()]

        # Ensure all images have the same number of channels (convert grayscale to 3 channels)
        for i in range(len(image_list)):
            name, img = image_list[i]
            if len(img.shape) == 2:  # Grayscale image (2D)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            image_list[i] = (name, img)

        num_images = len(image_list)
        num_rows = (num_images // max_images_per_row) + (1 if num_images % max_images_per_row != 0 else 0)

        rows = []

        for i in range(num_rows):
            row_images = image_list[i * max_images_per_row:(i + 1) * max_images_per_row]

            # Pad the row if necessary (to make all rows equally wide)
            while len(row_images) < max_images_per_row:
                row_images.append(("Empty", np.zeros_like(row_images[0][1])))  # Add empty images

            labeled_images = []
            for name, img in row_images:
                # Create a blank space for the text label (50 pixels high)
                label_area = np.zeros((50, img.shape[1], 3), dtype="uint8")

                # Put the text label in the center of the label area
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                text_size = cv2.getTextSize(name, font, font_scale, font_thickness)[0]
                text_x = (img.shape[1] - text_size[0]) // 2
                text_y = (50 + text_size[1]) // 2
                cv2.putText(label_area, name, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

                # Stack label area above the image
                labeled_img = np.vstack([label_area, img])
                labeled_images.append(labeled_img)

            # Concatenate images horizontally to form a row
            row = np.hstack(labeled_images)
            rows.append(row)

        # Concatenate all rows vertically to form the final image grid
        grid_image = np.vstack(rows)

        # Display the final image
        cv2.imshow("Debug Images", grid_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_image_bytes(self, step_name: str):
        """
        Retrieve an image as bytes for serving in Flask.

        Parameters:
            step_name (str): The name of the debug step.

        Returns:
            Flask response: Image file response or None if not found.
        """
        if step_name not in self.images:
            return None  # Return None if image not found

        _, img_encoded = cv2.imencode('.jpg', self.images[step_name])  # Encode image as JPEG
        return io.BytesIO(img_encoded.tobytes())  # Convert to BytesIO for Flask serving