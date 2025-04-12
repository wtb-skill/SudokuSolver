import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from typing import List, Optional


class DigitDatasetGenerator:
    def __init__(self, image_size: int = 32, output_dir: str = "digit_dataset", num_samples: int = 100,
                 blur_level: float = 2, shift_range: float = 1, rotation_range: int = 10,
                 noise_level: int = 10, fonts_dir: str = "fonts", clean_proportion: float = 0.5):
        """
        Initializes the dataset generator with configurable parameters.

        :param image_size: Size of the generated images (width and height in pixels).
        :param output_dir: Directory where generated images will be saved.
        :param num_samples: Number of images to generate per digit.
        :param blur_level: List of blur levels to apply (higher values increase blurring effect).
        :param shift_range: Maximum number of pixels a digit can shift randomly.
        :param rotation_range: Maximum rotation angle (in degrees) applied randomly.
        :param noise_level: Intensity of random noise added to the images.
        :param fonts_dir: Directory where font files are located.
        :param clean_proportion: The proportion of clean images to generate (between 0 and 1).
        """
        self.image_size: int = image_size
        self.output_dir: str = output_dir
        self.num_samples: int = num_samples
        self.blur_level: float = blur_level
        self.shift_range: float = shift_range
        self.rotation_range: int = rotation_range
        self.noise_level: int = noise_level
        self.fonts_dir: str = fonts_dir
        self.clean_proportion: float = clean_proportion

        # Automatically load font paths from the specified fonts directory
        self.font_paths: List[str] = self.load_fonts_from_directory(fonts_dir)

        if not self.font_paths:
            raise ValueError(f"No fonts found in {fonts_dir}! Please make sure the fonts are available.")

        print("Using the following fonts for digit generation:", self.font_paths)
        os.makedirs(self.output_dir, exist_ok=True)

    def load_fonts_from_directory(self, fonts_dir: str) -> List[str]:
        """
        Loads font files from the specified directory.

        :param fonts_dir: Directory where font files are located.
        :return: List of font file paths.
        """
        # Get the absolute path to the fonts directory relative to the current script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Path of the current script
        fonts_path = os.path.join(script_dir, fonts_dir)  # Correct path to 'fonts' folder

        if not os.path.exists(fonts_path):
            raise FileNotFoundError(f"Fonts directory not found: {fonts_path}")

        font_files = []
        for font_name in os.listdir(fonts_path):
            font_path = os.path.join(fonts_path, font_name)
            if font_name.endswith('.ttf') or font_name.endswith('.otf'):
                font_files.append(font_path)
        return font_files

    def create_digit_image(self, digit, font_path):
        font = ImageFont.truetype(font_path, size=28)
        img = Image.new("L", (self.image_size, self.image_size), 0)  # Black background
        draw = ImageDraw.Draw(img)

        # Get the bounding box of the text to be drawn (x0, y0, x1, y1)
        bbox = draw.textbbox((0, 0), str(digit), font=font)

        # Calculate the width and height of the text from the bounding box
        w = bbox[2] - bbox[0]  # width of the bounding box
        h = bbox[3] - bbox[1]  # height of the bounding box

        # Calculate x to center the digit horizontally
        x = (self.image_size - w) // 2 + random.uniform(-self.shift_range, self.shift_range)

        # Calculate y to start drawing from 3/4 of the height
        y = int(self.image_size - 2 * h) // 2 + random.uniform(-self.shift_range, self.shift_range)

        # Draw the text at the new (x, y) position
        draw.text((x, y), str(digit), font=font, fill=255) # Draw the text in white (255)

        return np.array(img)

    def apply_blur(self, img_np):
        # Randomly pick a blur strength as a float between 0 and 2 (or any other range you prefer)
        blur_strength = random.uniform(0.0, self.blur_level)

        # If blur_strength is greater than 0, apply blur using Gaussian kernel size based on the strength
        if blur_strength > 0:
            # Calculate the kernel size based on blur_strength
            kernel_size = int(blur_strength * 5) + 1  # Multiplier ensures larger blur for higher values
            # Make sure the kernel size is odd (Gaussian blur requires odd sizes)
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)

        return img_np

    def apply_distortion(self, img_np):
        rows, cols = img_np.shape
        src_pts = np.float32([[5, 5], [25, 5], [5, 25]])
        dst_pts = src_pts + np.random.randint(-2, 3, src_pts.shape).astype(np.float32)
        matrix = cv2.getAffineTransform(src_pts, dst_pts)
        return cv2.warpAffine(img_np, matrix, (cols, rows))

    def apply_rotation(self, img_np):
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        rows, cols = img_np.shape
        matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(img_np, matrix, (cols, rows))

    def apply_noise(self, img_np):
        noise = np.random.randint(-self.noise_level, self.noise_level, img_np.shape, dtype=np.int16)
        img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img_np

    def generate_images(self):
        """
        Generates a dataset of clean and distorted digit images.

        - Creates subdirectories for each digit (1-9).
        - Generates clean images using specified fonts.
        - Generates distorted images by applying various transformations.
        """
        print("[INFO] Starting dataset generation...")

        for digit in range(1, 10):
            digit_dir = os.path.join(self.output_dir, str(digit))
            os.makedirs(digit_dir, exist_ok=True)
            print(f"[INFO] Generating images for digit '{digit}' in {digit_dir}")

            # Generate clean images (based on clean_proportion)
            num_clean = int(self.num_samples * self.clean_proportion)
            print(f"[INFO] Generating {num_clean} clean images for digit '{digit}'")

            for i in range(num_clean):
                font_path = random.choice(self.font_paths)
                img_np = self.create_digit_image(digit, font_path)

                save_path = os.path.join(digit_dir, f"clean_{i}.png")
                cv2.imwrite(save_path, img_np)

            print(f"[INFO] {num_clean} clean images saved for digit '{digit}'")

            # Generate distorted images (remaining samples)
            num_distorted = self.num_samples - num_clean
            print(f"[INFO] Generating {num_distorted} distorted images for digit '{digit}'")

            for i in range(num_distorted):
                font_path = random.choice(self.font_paths)
                img_np = self.create_digit_image(digit, font_path)

                # Apply distortions
                img_np = self.apply_blur(img_np)
                img_np = self.apply_distortion(img_np)
                img_np = self.apply_rotation(img_np)
                img_np = self.apply_noise(img_np)

                save_path = os.path.join(digit_dir, f"distorted_{i}.png")
                cv2.imwrite(save_path, img_np)

            print(f"[INFO] {num_distorted} distorted images saved for digit '{digit}'")

        print("[INFO] Dataset generation complete!")


if __name__ == "__main__":
    generator = DigitDatasetGenerator(num_samples=10000, clean_proportion=0.3,
                                      output_dir="digit_dataset")  # Example: 40% clean, 60% distorted
                                                                # digit_dataset or test_dataset
    generator.generate_images()
