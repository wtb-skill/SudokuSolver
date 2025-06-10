import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from typing import List
from dev_tools.data_utils.dataset_deduplicator import DuplicateImageReducer


class DeterministicDigitDatasetGenerator:
    def __init__(
        self,
        image_size: int = 32,
        output_dir: str = "digit_dataset_deterministic",
        fonts_dir: str = "fonts",
        font_sizes: List[int] = [20, 22, 24, 26, 28],
        blur_levels: List[int] = [1, 3, 5, 7],
        shift_ranges: List[int] = [1, 2, 3],
        rotations: List[int] = [-10, -5, 0, 5, 10],
        noise_levels: List[int] = [4, 6, 8],
    ):
        self.image_size = image_size
        self.output_dir = output_dir
        self.fonts_dir = fonts_dir
        self.font_sizes = font_sizes
        self.blur_levels = blur_levels
        self.shift_ranges = shift_ranges
        self.rotations = rotations
        self.noise_levels = noise_levels

        self.font_paths = self.load_fonts_from_directory(fonts_dir)
        if not self.font_paths:
            raise ValueError(
                f"No fonts found in {fonts_dir}! Please make sure the fonts are available."
            )

        print("Using the following fonts for digit generation:", self.font_paths)
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def load_fonts_from_directory(fonts_dir: str) -> List[str]:
        return [
            os.path.join(fonts_dir, font_name)
            for font_name in os.listdir(fonts_dir)
            if font_name.endswith((".ttf", ".otf"))
        ]

    def create_digit_image(
        self, digit: int, font_path: str, font_size: int, shift_x: int, shift_y: int
    ) -> np.ndarray:
        font = ImageFont.truetype(font_path, size=font_size)
        img = Image.new("L", (self.image_size, self.image_size), 0)
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), str(digit), font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        x = (self.image_size - w) // 2 + shift_x
        y = (self.image_size - h) // 2 + shift_y

        draw.text((x, y), str(digit), font=font, fill=255)
        return np.array(img)

    def apply_blur(self, img: np.ndarray, kernel_size: int) -> np.ndarray:
        if kernel_size > 1:
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        return img

    def apply_rotation(self, img: np.ndarray, angle: int) -> np.ndarray:
        matrix = cv2.getRotationMatrix2D(
            (self.image_size / 2, self.image_size / 2), angle, 1
        )
        return cv2.warpAffine(img, matrix, (self.image_size, self.image_size))

    def apply_noise(self, img: np.ndarray, noise_level: int) -> np.ndarray:
        noise = np.random.randint(-noise_level, noise_level, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def apply_distortion(img: np.ndarray) -> np.ndarray:
        rows, cols = img.shape
        src_pts = np.float32([[5, 5], [25, 5], [5, 25]])
        dst_pts = src_pts + np.random.randint(-2, 3, src_pts.shape).astype(np.float32)
        matrix = cv2.getAffineTransform(src_pts, dst_pts)
        return cv2.warpAffine(img, matrix, (cols, rows))

    def generate_images(self):
        print("[INFO] Starting deterministic dataset generation...")
        total_images = (
            len(self.font_paths)
            * len(self.font_sizes)
            * len(self.blur_levels)
            * len(self.shift_ranges)
            * len(self.rotations)
            * len(self.noise_levels)
            * 9
        )

        with tqdm(total=total_images, desc="Generating images", unit="img") as pbar:
            for digit in range(1, 10):
                digit_dir = os.path.join(self.output_dir, str(digit))
                os.makedirs(digit_dir, exist_ok=True)

                for font_path in self.font_paths:
                    for font_size in self.font_sizes:
                        for blur_level in self.blur_levels:
                            for shift_x in self.shift_ranges:
                                for shift_y in self.shift_ranges:
                                    for rotation in self.rotations:
                                        for noise_level in self.noise_levels:
                                            img = self.create_digit_image(
                                                digit,
                                                font_path,
                                                font_size,
                                                shift_x,
                                                shift_y,
                                            )
                                            img = self.apply_blur(img, blur_level)
                                            img = self.apply_rotation(img, rotation)
                                            img = self.apply_noise(img, noise_level)
                                            img = self.apply_distortion(img)

                                            filename = (
                                                f"{os.path.basename(font_path)}_{font_size}_"
                                                f"blur{blur_level}_shift{shift_x}_{shift_y}_rot{rotation}_"
                                                f"noise{noise_level}.png"
                                            )
                                            save_path = os.path.join(
                                                digit_dir, filename
                                            )
                                            cv2.imwrite(save_path, img)
                                            pbar.update(1)

        print("[INFO] Deduplicating images...")
        reducer = DuplicateImageReducer(self.output_dir)
        reducer.run()


if __name__ == "__main__":
    generator = DeterministicDigitDatasetGenerator(
        image_size=32,
        output_dir="digit_dataset_deterministic",
        fonts_dir="fonts",
        font_sizes=[20, 22, 24, 26, 28],
        blur_levels=[1, 3, 5, 7],
        shift_ranges=[1, 2, 3],
        rotations=[-10, -5, 0, 5, 10],
        noise_levels=[4, 6, 8],
    )
    generator.generate_images()
