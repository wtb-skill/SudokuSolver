import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from typing import List


class DigitDatasetGenerator:
    def __init__(
        self,
        image_size: int = 32,
        output_dir: str = "digit_dataset",
        num_samples: int = 100,
        blur_level: float = 2,
        shift_range: float = 1,
        rotation_range: int = 10,
        noise_level: int = 10,
        fonts_dir: str = "fonts",
        clean_proportion: float = 0.5
    ):
        """
        Initializes the digit dataset generator.

        Parameters:
            image_size (int): Size of each generated image (square).
            output_dir (str): Directory where images will be saved.
            num_samples (int): Number of images per digit to generate.
            blur_level (float): Maximum Gaussian blur strength.
            shift_range (float): Max pixel shift in x/y direction.
            rotation_range (int): Max rotation angle in degrees.
            noise_level (int): Intensity of added random noise.
            fonts_dir (str): Directory containing .ttf/.otf font files.
            clean_proportion (float): Fraction of clean images (0 to 1).
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

        self.font_paths: List[str] = self.load_fonts_from_directory(fonts_dir)
        if not self.font_paths:
            raise ValueError(f"No fonts found in {fonts_dir}! Please make sure the fonts are available.")

        print("Using the following fonts for digit generation:", self.font_paths)
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def load_fonts_from_directory(fonts_dir: str) -> List[str]:
        """
        Loads all .ttf and .otf fonts from the specified directory.

        Parameters:
            fonts_dir (str): Relative path to the fonts' folder.

        Returns:
            List[str]: Paths to valid font files.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fonts_path = os.path.join(script_dir, fonts_dir)

        if not os.path.exists(fonts_path):
            raise FileNotFoundError(f"Fonts directory not found: {fonts_path}")

        return [
            os.path.join(fonts_path, font_name)
            for font_name in os.listdir(fonts_path)
            if font_name.endswith(('.ttf', '.otf'))
        ]

    def create_digit_image(self, digit: int, font_path: str) -> np.ndarray:
        """
        Renders a digit into a centered grayscale image using a given font.

        Parameters:
            digit (int): Digit (1–9) to render.
            font_path (str): Path to a valid .ttf/.otf font.

        Returns:
            np.ndarray: Grayscale image of the digit.
        """
        font = ImageFont.truetype(font_path, size=28)
        img = Image.new("L", (self.image_size, self.image_size), 0)
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), str(digit), font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        x = (self.image_size - w) // 2 + random.uniform(-self.shift_range, self.shift_range)
        y = (self.image_size - 2 * h) // 2 + random.uniform(-self.shift_range, self.shift_range)

        draw.text((x, y), str(digit), font=font, fill=255)
        return np.array(img)

    def apply_blur(self, img_np: np.ndarray) -> np.ndarray:
        """
        Applies Gaussian blur to the image.

        Parameters:
            img_np (np.ndarray): Input image.

        Returns:
            np.ndarray: Blurred image.
        """
        blur_strength = random.uniform(0.0, self.blur_level)
        if blur_strength > 0:
            kernel_size = int(blur_strength * 5) + 1
            kernel_size += 1 if kernel_size % 2 == 0 else 0
            img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
        return img_np

    @staticmethod
    def apply_distortion(img_np: np.ndarray) -> np.ndarray:
        """
        Applies affine transformation for minor distortions.

        Parameters:
            img_np (np.ndarray): Input image.

        Returns:
            np.ndarray: Distorted image.
        """
        rows, cols = img_np.shape
        src_pts = np.float32([[5, 5], [25, 5], [5, 25]])
        dst_pts = src_pts + np.random.randint(-2, 3, src_pts.shape).astype(np.float32)
        matrix = cv2.getAffineTransform(src_pts, dst_pts)
        return cv2.warpAffine(img_np, matrix, (cols, rows))

    def apply_rotation(self, img_np: np.ndarray) -> np.ndarray:
        """
        Applies random rotation to the image.

        Parameters:
            img_np (np.ndarray): Input image.

        Returns:
            np.ndarray: Rotated image.
        """
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        rows, cols = img_np.shape
        matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(img_np, matrix, (cols, rows))

    def apply_noise(self, img_np: np.ndarray) -> np.ndarray:
        """
        Adds random noise to the image.

        Parameters:
            img_np (np.ndarray): Input image.

        Returns:
            np.ndarray: Noisy image.
        """
        noise = np.random.randint(-self.noise_level, self.noise_level, img_np.shape, dtype=np.int16)
        img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img_np

    def generate_images(self) -> None:
        """
        Generates clean and distorted digit images for digits 1–9.

        Saves images in subdirectories under `output_dir` named after each digit.
        """
        print("[INFO] Starting dataset generation...")

        for digit in range(1, 10):
            digit_dir = os.path.join(self.output_dir, str(digit))
            os.makedirs(digit_dir, exist_ok=True)

            num_clean = int(self.num_samples * self.clean_proportion)
            num_distorted = self.num_samples - num_clean

            print(f"[INFO] Generating {num_clean} clean and {num_distorted} distorted images for '{digit}'")

            # Clean images
            for i in range(num_clean):
                font_path = random.choice(self.font_paths)
                img_np = self.create_digit_image(digit, font_path)
                save_path = os.path.join(digit_dir, f"clean_{i}.png")
                cv2.imwrite(save_path, img_np)

            # Distorted images
            for i in range(num_distorted):
                font_path = random.choice(self.font_paths)
                img_np = self.create_digit_image(digit, font_path)
                img_np = self.apply_blur(img_np)
                img_np = self.apply_distortion(img_np)
                img_np = self.apply_rotation(img_np)
                img_np = self.apply_noise(img_np)
                save_path = os.path.join(digit_dir, f"distorted_{i}.png")
                cv2.imwrite(save_path, img_np)

        print("[INFO] Dataset generation complete!")


if __name__ == "__main__":
    generator = DigitDatasetGenerator(
        num_samples=1000,
        clean_proportion=0.3,
        output_dir="test_dataset" # test_dataset to use for ModelEvaluator outside training or
                                # digit_dataset to use for training
    )
    generator.generate_images()

