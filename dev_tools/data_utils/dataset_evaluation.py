import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import random
import datetime
import imagehash
from PIL import Image
import concurrent.futures
from tqdm import tqdm
from typing import List, Dict, Union


class DigitDatasetEvaluator:
    """
    A comprehensive evaluation tool for digit image datasets.

    Provides analysis on:
    - Class distribution
    - Corrupt image detection
    - Digit centering
    - Pixel intensity distributions
    - Duplicate/near-duplicate detection
    - Partial (cut-off) digit detection
    """

    def __init__(self, dataset_path: str, image_size: int = 32, output_dir: str = "evaluation_reports") -> None:
        """
        Initializes the DigitDatasetEvaluator.

        Args:
            dataset_path (str): Path to the root dataset directory containing digit subfolders (1-9).
            image_size (int, optional): Size of the images (assumes square). Defaults to 32.
            output_dir (str, optional): Directory to save evaluation outputs. Defaults to "evaluation_reports".
        """
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.output_dir = output_dir
        self.digit_dirs = {str(i): os.path.join(dataset_path, str(i)) for i in range(1, 10)}
        os.makedirs(output_dir, exist_ok=True)
        self.report_lines: List[str] = []

        # Visual constants
        self.plot_params = {
            "figsize": (10, 10),            # Default figure size for histograms
            "font_size": 14,               # Font size for titles and labels
            "bar_color": 'skyblue',        # Default color for bars in histograms
            "bar_edgecolor": 'black',      # Bar edge color for histograms
            "color_map": 'hot',            # Default colormap for heatmaps
            "clean_color": "green",
            "distorted_color": "red",
            "grid_color": '#cccccc',       # Grid line color in plots
            "legend_font_size": 12,        # Font size for legend
            "legend_loc": 'best'           # Location of the legend in plots
        }

    def log(self, text: str) -> None:
        """Logs messages to console and saves to the evaluation report."""
        print(text)
        self.report_lines.append(text)

    def get_plot_params(self) -> dict:
        """Returns the plot parameters for consistent visualization."""
        return self.plot_params

    def set_plot_params(self, figsize: tuple = None, font_size: int = None,
                        bar_color: str = None, color_map: str = None,
                        grid_color: str = None, legend_font_size: int = None,
                        legend_loc: str = None) -> None:
        """Set custom plot parameters."""
        if figsize: self.plot_params["figsize"] = figsize
        if font_size: self.plot_params["font_size"] = font_size
        if bar_color: self.plot_params["bar_color"] = bar_color
        if color_map: self.plot_params["color_map"] = color_map
        if grid_color: self.plot_params["grid_color"] = grid_color
        if legend_font_size: self.plot_params["legend_font_size"] = legend_font_size
        if legend_loc: self.plot_params["legend_loc"] = legend_loc

    def step_1_class_distribution(self) -> Dict[str, Dict[str, int]]:
        """
        Analyzes and logs the distribution of clean vs distorted images per digit class.

        This step:
          - Counts total, clean, and distorted images for each digit (1–9).
          - Saves a grouped bar chart comparing these counts across digits.
          - Logs the results and returns a summary dictionary.

        Returns:
            Dict[str, Dict[str, int]]: A nested dictionary where keys are digit strings ('1'–'9'),
            and values are dictionaries containing counts for 'total', 'clean', and 'distorted' images.
        """
        self.log("\n📊 [Step 1] Starting class distribution analysis...")
        summary: Dict[str, Dict[str, int]] = {}

        # Retrieve the plot parameters (font size, colors, etc.)
        plot_params = self.get_plot_params()

        for digit, path in self.digit_dirs.items():
            if not os.path.exists(path):
                continue

            images = os.listdir(path)
            clean = len([img for img in images if "clean" in img])
            distorted = len(images) - clean

            summary[digit] = {
                "total": len(images),
                "clean": clean,
                "distorted": distorted
            }

        # 📊 Generate grouped bar chart
        digits = sorted(summary.keys(), key=int)
        totals = [summary[d]["total"] for d in digits]
        cleans = [summary[d]["clean"] for d in digits]
        distorteds = [summary[d]["distorted"] for d in digits]

        x = range(len(digits))
        width = 0.3

        plt.figure(figsize=plot_params["figsize"])
        plt.bar([i - width for i in x], totals, width=width, label="Total", color="gray",
                edgecolor=plot_params["bar_edgecolor"])
        plt.bar(x, cleans, width=width, label="Clean", color=plot_params["bar_color"],
                edgecolor=plot_params["bar_edgecolor"])
        plt.bar([i + width for i in x], distorteds, width=width, label="Distorted", color="red",
                edgecolor=plot_params["bar_edgecolor"])

        plt.xlabel("Digit Class", fontsize=plot_params["font_size"])
        plt.ylabel("Number of Images", fontsize=plot_params["font_size"])
        plt.title("Image Distribution per Digit (Total / Clean / Distorted)", fontsize=plot_params["font_size"])
        plt.xticks(x, digits, fontsize=plot_params["font_size"])
        plt.legend(fontsize=plot_params["legend_font_size"], loc=plot_params["legend_loc"])
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, "class_distribution_histogram.png")
        plt.savefig(save_path)
        plt.close()

        self.log(f"📊 [Step 1] Saved class distribution histogram → {save_path}")
        self.log("✅ [Step 1 Complete] Class distribution analysis done.\n")

        return summary

    def step_2_check_image_dimensions(self) -> str:
        """
        Analyzes all images in the dataset to identify unique image dimensions and generates a histogram.

        This step:
          - Checks and counts each unique (height x width) image dimension.
          - Logs the number of unique sizes and examples of each.
          - Saves a histogram plot visualizing the distribution of image dimensions.

        Returns:
            str: A string summarizing image size information for reporting purposes.
                 - If all images have the same dimensions, returns "HxW" (e.g., "32×32").
                 - If multiple dimensions are found, returns "Mixed".
        """
        self.log("\n📏 [Step 2] Checking image dimensions...")

        dim_counter: Dict[Tuple[int, int], int] = defaultdict(int)
        dim_examples: Dict[Tuple[int, int], List[str]] = defaultdict(list)

        all_images: List[Tuple[str, str]] = [
            (digit, os.path.join(path, img))
            for digit, path in self.digit_dirs.items()
            for img in os.listdir(path)
        ]

        with tqdm(total=len(all_images), desc="Analyzing image sizes", ncols=100) as bar:
            for digit, img_path in all_images:
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        h, w = img.shape
                        dim_counter[(h, w)] += 1
                        if len(dim_examples[(h, w)]) < 3:
                            dim_examples[(h, w)].append(img_path)
                except Exception:
                    pass
                bar.update(1)

        self.log(f"[Step 2] Found {len(dim_counter)} unique image sizes.")
        for dim, count in sorted(dim_counter.items(), key=lambda x: -x[1]):
            self.log(f"[Step 2] Size {dim}: {count} images. Examples: {dim_examples[dim]}")

        plot_params = self.get_plot_params()
        labels: List[str] = [f"{h}x{w}" for (h, w) in dim_counter.keys()]
        counts: List[int] = list(dim_counter.values())

        plt.figure(figsize=plot_params["figsize"])
        plt.bar(labels, counts, color=plot_params["bar_color"], edgecolor=plot_params["bar_edgecolor"])
        plt.title("Image Dimension Distribution", fontsize=plot_params["font_size"])
        plt.xlabel("Dimensions (HxW)", fontsize=plot_params["font_size"])
        plt.ylabel("Number of Images", fontsize=plot_params["font_size"])
        plt.xticks(rotation=45, ha='right', fontsize=plot_params["font_size"])
        plt.tight_layout()

        save_path: str = os.path.join(self.output_dir, "image_dimension_histogram.png")
        plt.savefig(save_path)
        plt.close()

        self.log(f"📊 [Step 2] Saved image dimension histogram → {save_path}")
        self.log("✅ [Step 2 Complete] Image dimension check done.\n")

        if len(dim_counter) == 1:
            (h, w), _ = next(iter(dim_counter.items()))
            return f"{h}×{w}"
        else:
            return "Mixed"

    def step_3_visualize_sample_grid(self, samples_per_digit: int = 10) -> None:
        """
        Saves a grid visualization of random samples for each digit.

        This step:
          - Randomly selects a specified number of images from each digit class.
          - Displays them in a grid format, with one row per digit (1-9).
          - Adds a title and saves the resulting image to a file.

        Args:
            samples_per_digit (int): The number of sample images to display per digit class. Default is 5.

        Returns:
            None
        """
        self.log("\n🖼️ [Step 3] Generating sample grid visualization...")

        # Get visual parameters from the class (e.g., fontsize, figsize, etc.)
        plot_params = self.get_plot_params()

        # Create a subplot grid for 9 rows (for digits 1-9) and the specified number of samples per digit
        fig, axs = plt.subplots(9, samples_per_digit, figsize=(samples_per_digit * 1.5, 13))

        # Iterate through each digit (1-9)
        for row, digit in enumerate(self.digit_dirs):
            imgs = os.listdir(self.digit_dirs[digit])  # Get images for the current digit
            samples = random.sample(imgs, min(samples_per_digit, len(imgs)))  # Randomly sample images

            for col, img_name in enumerate(samples):
                img_path = os.path.join(self.digit_dirs[digit], img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
                axs[row, col].imshow(img, cmap='gray')  # Display image in the grid
                axs[row, col].axis("off")  # Hide axes

        # Add a styled global title above the grid
        fig.suptitle("Sample Images per Digit", fontsize=plot_params["font_size"])

        # Adjust layout to leave space for the title, then save the output
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for title
        output_path = os.path.join(self.output_dir, "sample_grid.png")
        plt.savefig(output_path, bbox_inches="tight")  # Save the image
        plt.close()

        self.log(f"✅ [Step 3 Complete] Sample image grid saved as '{output_path}'\n")

    def step_4_intensity_histograms(self, sample_size: int = 1000) -> None:
        """
        Plots histograms of pixel intensity values for clean and distorted images.

        This step:
          - Loads a sample of images from each digit class.
          - Separates the pixels into clean and distorted categories based on the image filenames.
          - Plots the pixel intensity distributions for clean and distorted images and saves the plot.

        Args:
            sample_size (int): The number of images to sample from each digit class for the histogram. Default is 1000.

        Returns:
            None
        """
        self.log("\n📈 [Step 4] Generating pixel intensity histograms...")

        # Get visual parameters from the class (e.g., font size, color, etc.)
        plot_params = self.get_plot_params()

        clean_pixels, distorted_pixels = [], []
        total_images = sum(
            min(sample_size, len(os.listdir(path)))
            for path in self.digit_dirs.values()
            if os.path.exists(path)
        )

        # Load and process images
        with tqdm(total=total_images, desc="🔍 Loading images", unit="img") as pbar:
            for digit, path in self.digit_dirs.items():
                if not os.path.exists(path):
                    continue
                images = os.listdir(path)
                random.shuffle(images)
                for img_name in images[:sample_size]:
                    full_path = os.path.join(path, img_name)
                    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        if "clean" in img_name:
                            clean_pixels.extend(img.flatten())
                        else:
                            distorted_pixels.extend(img.flatten())
                    pbar.update(1)

        # Plot histograms for clean and distorted images
        with tqdm(total=1, desc="📊 Plotting & Saving", unit="task") as pbar:
            plt.figure(figsize=(plot_params["figsize"][0], plot_params["figsize"][1]))
            plt.hist(clean_pixels, bins=50, alpha=0.6, label="Clean", color=plot_params["clean_color"])
            plt.hist(distorted_pixels, bins=50, alpha=0.6, label="Distorted", color=plot_params["distorted_color"])

            plt.title("Pixel Intensity Histogram", fontsize=plot_params["font_size"])
            plt.xlabel("Pixel Value", fontsize=plot_params["font_size"])
            plt.ylabel("Frequency", fontsize=plot_params["font_size"])
            plt.legend(fontsize=plot_params["font_size"])

            output_path = os.path.join(self.output_dir, "pixel_histogram.png")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            pbar.update(1)

        self.log(f"✅ [Step 4 Complete] Histogram saved as '{output_path}'\n")

    def step_5_detect_corrupt_images(self) -> List[str]:
        """
        Detects images that are blank or nearly blank by checking if the pixel intensity range is too small.

        This step:
          - Scans each image in the dataset.
          - Detects images that are corrupt (i.e., images that are entirely blank or nearly blank).
          - Returns a list of potentially corrupt image paths.

        Args:
            None

        Returns:
            List[str]: A list of paths to images that are detected as corrupt.
        """
        self.log("\n🔍 [Step 5] Scanning for corrupt images...")

        corrupt: List[str] = []
        total_images = sum(len(os.listdir(path)) for path in self.digit_dirs.values())

        # Scan images for corruption
        with tqdm(total=total_images, desc="Scanning images", unit="img", ncols=100) as pbar:
            for digit, path in self.digit_dirs.items():
                if not os.path.exists(path):
                    self.log(f"   ⛔ Directory missing: {path}")
                    continue
                for img_name in os.listdir(path):
                    img_path = os.path.join(path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None or img.max() - img.min() < 10:  # Detect blank or nearly blank images
                        corrupt.append(img_path)
                    pbar.update(1)

        # Log potential corrupt images
        if corrupt:
            self.log(f"⚠️ [Step 5] Found {len(corrupt)} potentially corrupt images. Showing first 10:")
            for c in corrupt[:10]:
                self.log(f"    - {c}")
        else:
            self.log("✅ [Step 5] No corrupt images detected.")

        self.log("✅ [Step 5 Complete] Corrupt image scan finished.\n")
        return corrupt

    def step_6_digit_centering_heatmap(self, samples_per_digit: int = 100) -> None:
        """
        Creates a heatmap to visualize the average centering of digits across the dataset.

        This step:
          - Accumulates image data and averages it to visualize how centered the digits are.
          - Creates a heatmap that shows the normalized pixel intensity, which represents the centering of the digits.
          - Saves the resulting heatmap as an image.

        Args:
            samples_per_digit (int): The number of samples to use per digit to compute the centering heatmap. Default is 100.

        Returns:
            None
        """
        self.log("\n🔥 [Step 6] Generating digit centering heatmap...")

        accumulator = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        total = 0

        # Calculate the total number of images for progress tracking
        total_images = sum(min(samples_per_digit, len(os.listdir(path)))
                           for path in self.digit_dirs.values() if os.path.exists(path))

        with tqdm(total=total_images, desc="Processing images", unit="img", ncols=100) as pbar:
            for digit, path in self.digit_dirs.items():
                if not os.path.exists(path):
                    self.log(f"   ⛔ Skipping missing directory: {path}")
                    continue
                images = os.listdir(path)
                random.shuffle(images)
                for img_name in images[:samples_per_digit]:
                    img_path = os.path.join(path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        norm_img = img.astype(np.float32) / 255.0  # Normalize the image
                        accumulator += norm_img
                        total += 1
                    pbar.update(1)

        if total > 0:
            accumulator /= total  # Average the accumulated images

            # Get visualization parameters
            plot_params = self.get_plot_params()

            # Use the fixed figsize from plot_params
            plt.figure(figsize=plot_params["figsize"])

            # Generate heatmap
            plt.imshow(accumulator, cmap=plot_params["color_map"], interpolation="nearest")
            plt.colorbar(label="Normalized Intensity")
            plt.title("Digit Centering Heatmap (all digits)", fontsize=plot_params["font_size"])

            # Save plot
            output_path = os.path.join(self.output_dir, "centering_heatmap.png")
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()

            self.log(f"✅ [Step 6 Complete] Digit centering heatmap saved as '{output_path}'\n")
        else:
            self.log("⚠️ [Step 6 Complete] No images found for centering heatmap.")

    def step_7_detect_blurry_images(self, threshold: float = 100.0) -> List[Tuple[str, float]]:
        """
        Detects blurry images using the variance of the Laplacian method.

        This step:
          - Calculates the Laplacian variance for each image, which is a measure of image sharpness.
          - Flags images with a Laplacian variance below the specified threshold as blurry.
          - Logs the results and provides a list of detected blurry images.

        Args:
            threshold (float): The Laplacian variance threshold below which images will be considered blurry. Default is 100.0.

        Returns:
            List[Tuple[str, float]]: A list of tuples where each tuple contains the path to a blurry image and its Laplacian variance score.
        """
        self.log("\n🔍 [Step 7] Detecting blurry images...")

        blurry_images: List[Tuple[str, float]] = []
        all_images = [(digit, os.path.join(path, img))
                      for digit, path in self.digit_dirs.items()
                      for img in os.listdir(path)]

        with tqdm(total=len(all_images), desc="Checking for blur", ncols=100) as bar:
            for digit, img_path in all_images:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
                    if lap_var < threshold:
                        blurry_images.append((img_path, lap_var))
                bar.update(1)

        self.log(f"[Step 7] Found {len(blurry_images)} blurry images (Laplacian var < {threshold}).")
        for img_path, score in blurry_images[:10]:
            self.log(f" - {img_path} | Laplacian variance: {score:.2f}")

        self.log("✅ [Step 7 Complete] Blurry image detection done.\n")

        return blurry_images

    def step_8_estimate_dataset_diversity(self) -> List[Dict[str, Union[str, float, int]]]:
        """
        Estimates dataset diversity per digit via the average perceptual hash (phash) distance between images.

        Returns:
            List[Dict[str, Any]]: Diversity score and number of unique samples per digit.
        """
        self.log("\n🌈 [Step 8] Estimating dataset diversity (phash distance)...")

        digit_diversity_data = []
        digit_dirs = sorted(self.digit_dirs.items(), key=lambda x: int(x[0]))
        total_images = sum(len(os.listdir(path)) for _, path in digit_dirs)

        # Initialize total progress (hashing + pairwise comparisons for each digit)
        total_tasks = total_images + sum(
            len(os.listdir(path)) * (len(os.listdir(path)) - 1) // 2 for _, path in digit_dirs
        )

        with tqdm(total=total_tasks, desc="Overall Progress", ncols=100) as pbar:
            for digit, path in digit_dirs:
                image_paths = [os.path.join(path, f) for f in os.listdir(path)]
                hashes = []

                for p in image_paths:
                    try:
                        hashes.append(imagehash.phash(Image.open(p)))
                    except Exception:
                        continue
                    pbar.update(1)

                distances = []
                for i in range(len(hashes)):
                    for j in range(i + 1, len(hashes)):
                        distances.append(hashes[i] - hashes[j])
                        pbar.update(1)

                avg_dist = np.mean(distances) if distances else 0
                unique_hashes = len(set(str(h) for h in hashes))

                digit_diversity_data.append({
                    "Digit": digit,
                    "Diversity Score": round(avg_dist, 2),
                    "Unique Samples": unique_hashes
                })

            pbar.set_postfix({"Status": "Diversity Estimation Complete"})
            self.log("✅ [Step 8 Complete] Dataset diversity estimation done.\n")

        return digit_diversity_data

    def step_9_detect_partial_digits(self, margin: int = 0, samples_per_digit: int = 10) -> List[str]:
        """
        Detects images where digits touch the image frame (partial digits).

        This step:
          - Analyzes the bounding boxes of digits in each image.
          - Flags images where the digit is close to or touches the image edges (within the specified margin).

        Args:
            margin (int): Margin in pixels to consider near-edge proximity. Defaults to 0.
            samples_per_digit (int): Max number of flagged samples to display per digit. Default is 10.

        Returns:
            List[str]: List of paths to suspected partial digit images.
        """
        self.log("\n🔍 [Step 9] Detecting partial digits (those touching the image frame)...")

        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, "partial_digits_grid.png")
        partials: List[str] = []
        partials_by_digit: Dict[str, List[str]] = defaultdict(list)

        # Calculate total images for progress bar
        total_images = sum(len(os.listdir(path)) for path in self.digit_dirs.values())
        with tqdm(total=total_images, desc="Processing images", unit="img", ncols=100) as pbar:
            for digit, path in self.digit_dirs.items():
                images = os.listdir(path)
                for img_name in images:
                    img_path = os.path.join(path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    # Detect contours and check for partial digits
                    _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if not contours:
                        continue

                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    if (x <= margin or y <= margin or
                            x + w >= self.image_size - margin or
                            y + h >= self.image_size - margin):
                        partials.append(img_path)
                        partials_by_digit[digit].append(img_path)

                    pbar.update(1)  # Update progress bar

        if not partials_by_digit:
            self.log("✅ [Step 9 Complete] No partial digits detected based on edge proximity.")
            return []

        # Constructing visualization grid
        rows = len(partials_by_digit)
        cols = samples_per_digit
        grid_img = np.ones((rows * self.image_size, cols * self.image_size), dtype=np.uint8) * 255

        self.log("🖼️ [Step 9] Building composite image grid of partial digits...")
        for row_idx, digit in tqdm(enumerate(sorted(partials_by_digit.keys())), desc="Creating grid",
                                   total=len(partials_by_digit)):
            selected = partials_by_digit[digit][:samples_per_digit]
            for col_idx, img_path in enumerate(selected):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (self.image_size, self.image_size))
                    y1 = row_idx * self.image_size
                    y2 = y1 + self.image_size
                    x1 = col_idx * self.image_size
                    x2 = x1 + self.image_size
                    grid_img[y1:y2, x1:x2] = img_resized

        # Save grid image with a consistent and visually pleasing figsize
        plot_params = self.get_plot_params()
        fig, ax = plt.subplots(figsize=plot_params["figsize"])
        ax.imshow(grid_img, cmap="gray")
        ax.axis("off")
        ax.set_title("Partial Digit Detection", fontsize=plot_params["font_size"])

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        self.log(f"🟠 [Step 9] Found {len(partials)} potentially partial digit images.")
        self.log(f"✅ [Step 9 Complete] Composite image of partial digits saved as '{save_path}'\n")

        return partials

    def _compute_hash(self, img_path: str, hash_func=imagehash.phash) -> Optional[Tuple[imagehash.ImageHash, str]]:
        """Computes perceptual hash for a given image."""
        try:
            img = Image.open(img_path).convert("L").resize((32, 32))
            img_hash = hash_func(img)
            return img_hash, img_path
        except Exception:
            return None

    def step_10_detect_duplicate_images(self, hash_func=imagehash.phash, threshold: int = 0,
                                        max_groups: int = 5, samples_per_group: int = 5) -> np.ndarray:
        """
        Detects duplicate or near-duplicate images within each digit class using perceptual hashing.

        This step:
          - Computes perceptual hashes for all images.
          - Identifies duplicate or near-duplicate images based on a defined threshold.
          - Groups duplicate images and visualizes them in a grid.

        Args:
            hash_func (Callable): The hash function to use for image hashing. Default is `imagehash.phash`.
            threshold (int): The threshold for considering two images as duplicates. Default is 0.
            max_groups (int): Maximum number of duplicate groups to visualize per digit. Default is 5.
            samples_per_group (int): Number of duplicate images to show per group. Default is 5.

        Returns:
            np.ndarray: Final composite image of duplicate images.
        """
        self.log("\n🔍 [Step 10] Detecting duplicate or near-duplicate images within each digit class...")

        all_images: List[Tuple[imagehash.ImageHash, str, str]] = []

        # 1. Compute hashes
        image_paths_by_digit = {}
        for digit, path in self.digit_dirs.items():
            image_paths = [os.path.join(path, img_name) for img_name in os.listdir(path)]
            image_paths_by_digit[digit] = image_paths

        with tqdm(total=sum(len(v) for v in image_paths_by_digit.values()),
                  desc="Computing image hashes", ncols=100) as hash_pbar:

            for digit, image_paths in image_paths_by_digit.items():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = list(executor.map(self._compute_hash, image_paths, [hash_func] * len(image_paths)))

                for res in results:
                    if res is not None:
                        img_hash, img_path = res
                        all_images.append((img_hash, img_path, digit))
                    hash_pbar.update(1)

        # 2. Find duplicates
        total_images = len(all_images)
        duplicates: Dict[str, List[str]] = defaultdict(list)
        used = set()

        with tqdm(total=total_images, desc="Finding duplicate images", ncols=100) as find_bar:
            for i, (h1, path1, digit1) in enumerate(all_images):
                if path1 in used:
                    find_bar.update(1)
                    continue
                group = [path1]
                for j in range(i + 1, total_images):
                    h2, path2, digit2 = all_images[j]
                    if digit1 == digit2 and h1 - h2 <= threshold and path2 not in used:
                        group.append(path2)
                if len(group) > 1:
                    duplicates[str(h1)] = group
                    used.update(group)
                find_bar.update(1)

        if find_bar.n < find_bar.total:
            find_bar.update(find_bar.total - find_bar.n)

        # 3. Visualize duplicate groups
        digit_visuals = {}
        all_duplicates_by_digit = defaultdict(list)
        for group in duplicates.values():
            if group:
                digit = os.path.basename(os.path.dirname(group[0]))
                all_duplicates_by_digit[digit].append(group)

        for digit in sorted(self.digit_dirs.keys(), key=int):
            groups = all_duplicates_by_digit.get(digit, [])[:max_groups]
            if not groups:
                grid_img = np.ones((self.image_size, self.image_size), dtype=np.uint8) * 255
            else:
                rows = len(groups)
                cols = min(samples_per_group, max(len(group) for group in groups))
                grid_img = np.ones((rows * self.image_size, cols * self.image_size), dtype=np.uint8) * 255

                for row_idx, group in enumerate(groups):
                    for col_idx, img_path in enumerate(group[:samples_per_group]):
                        try:
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                img_resized = cv2.resize(img, (self.image_size, self.image_size))
                                y1 = row_idx * self.image_size
                                y2 = y1 + self.image_size
                                x1 = col_idx * self.image_size
                                x2 = x1 + self.image_size
                                grid_img[y1:y2, x1:x2] = img_resized
                        except Exception:
                            continue
            digit_visuals[int(digit) - 1] = grid_img

        # 4. Create composite image grid
        final_image = self._concatenate_3x3_grid(digit_visuals)

        # 5. Save grid using matplotlib with styled title
        grid_path = os.path.join(self.output_dir, "duplicate_images_grid.png")
        plot_params = self.get_plot_params()

        fig, ax = plt.subplots(figsize=plot_params["figsize"])
        ax.imshow(final_image, cmap="gray")
        ax.axis("off")
        ax.set_title("Duplicate Images Grid", fontsize=plot_params["font_size"])

        plt.tight_layout()
        plt.savefig(grid_path, bbox_inches="tight")
        plt.close()

        self.log(f"✅ [Step 10 Complete] Saved composite image of duplicate digits → {grid_path}")

        # 6. Plot histogram of duplicate group counts per digit
        duplicate_counts = [len(all_duplicates_by_digit.get(str(d), [])) for d in range(1, 10)]
        plt.figure(figsize=plot_params["figsize"])
        plt.bar(range(1, 10), duplicate_counts,
                color=plot_params["bar_color"],
                edgecolor=plot_params["bar_edgecolor"])
        plt.title("Duplicate Groups Per Digit", fontsize=plot_params["font_size"])
        plt.xlabel("Digit", fontsize=plot_params["font_size"])
        plt.ylabel("Duplicate Groups", fontsize=plot_params["font_size"])
        plt.grid(True, linestyle="--", color=plot_params["grid_color"])
        plt.xticks(range(1, 10))
        plt.tight_layout()

        hist_path = os.path.join(self.output_dir, "duplicate_group_histogram.png")
        plt.savefig(hist_path)
        plt.close()
        self.log(f"📊 Saved histogram of duplicate groups per digit → {hist_path}\n")

        return final_image, all_duplicates_by_digit

    def _concatenate_3x3_grid(self, digit_visuals: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Arranges digit visuals (0–8) into a 3x3 grid layout. If any are missing, fills with white.
        Returns the final concatenated image.
        """
        image_size = self.image_size
        cell_height = max(img.shape[0] for img in digit_visuals.values())
        cell_width = max(img.shape[1] for img in digit_visuals.values())

        rows = []
        for i in range(0, 9, 3):  # 0, 3, 6
            row_imgs = []
            for j in range(3):
                digit = i + j
                img = digit_visuals.get(digit, np.ones((cell_height, cell_width), dtype=np.uint8) * 255)
                padded_img = np.ones((cell_height, cell_width), dtype=np.uint8) * 255
                h, w = img.shape
                padded_img[:h, :w] = img
                row_imgs.append(padded_img)
            rows.append(cv2.hconcat(row_imgs))
        return cv2.vconcat(rows)

    def step_11_local_feature_consistency(self, samples_per_digit: int = 500) -> Tuple[
        List[Dict[str, Union[str, float, int]]], List[Dict[str, Union[str, float, int]]]]:
        """
        Analyzes local feature consistency across digits using both Sobel edges and ORB keypoints.
        Generates two separate 3x3 grid heatmaps (Sobel and ORB) and saves them as styled images.
        Returns relevant data for Table 5: Feature Consistency.

        Args:
            samples_per_digit (int): Number of samples to use per digit class. Default is 500.

        Returns:
            Tuple: Contains two lists of dictionaries with metrics for Sobel and ORB, respectively.
        """
        self.log(f"\n🔍 [Step 11] Checking local feature consistency using Sobel and ORB methods...")

        orb_detector = cv2.ORB_create(nfeatures=5000, edgeThreshold=5, fastThreshold=5)
        sobel_data = []
        orb_data = []

        os.makedirs(self.output_dir, exist_ok=True)
        total_images = samples_per_digit * len(self.digit_dirs)
        pbar = tqdm(total=total_images, desc="Processing images", ncols=100)

        # Setup 3x3 plot grids
        sobel_fig, sobel_axes = plt.subplots(3, 3, figsize=self.plot_params["figsize"])
        orb_fig, orb_axes = plt.subplots(3, 3, figsize=self.plot_params["figsize"])

        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        final_imshow_sobel = None
        final_imshow_orb = None

        for idx, (digit, path) in enumerate(sorted(self.digit_dirs.items())):
            images = os.listdir(path)
            random.shuffle(images)
            selected = images[:samples_per_digit]

            sobel_acc = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            orb_acc = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            sobel_count = 0
            orb_count = 0

            for img_name in selected:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    pbar.update(1)
                    continue

                # Sobel
                sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
                norm_mag = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
                sobel_acc += norm_mag
                sobel_count += 1

                # ORB
                keypoints = orb_detector.detect(img, None)
                kp_img = np.zeros_like(img, dtype=np.float32)
                for kp in keypoints:
                    x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
                    if 0 <= x < self.image_size and 0 <= y < self.image_size:
                        kp_img[y, x] = 1.0
                kp_img = cv2.GaussianBlur(kp_img, (5, 5), sigmaX=1, sigmaY=1)
                orb_acc += kp_img
                orb_count += 1

                pbar.update(1)

            # Store average metrics
            avg_sobel = np.sum(sobel_acc) / sobel_count if sobel_count else 0.0
            avg_orb = np.sum(orb_acc) / orb_count if orb_count else 0.0

            sobel_data.append({"Digit": digit, "Avg Sobel Intensity": avg_sobel})
            orb_data.append({"Digit": digit, "Avg ORB Keypoints": avg_orb})

            # Display Sobel heatmap
            if sobel_count > 0:
                vmin, vmax = np.percentile(sobel_acc, (1, 99))
                ax = sobel_axes[idx // 3, idx % 3]
                final_imshow_sobel = ax.imshow(sobel_acc, cmap=self.plot_params["color_map"], vmin=vmin, vmax=vmax)
                ax.set_title(f"Digit {digit} (Sobel)", fontsize=self.plot_params["font_size"])
                ax.axis("off")

            # Display ORB heatmap
            if orb_count > 0:
                vmin, vmax = np.percentile(orb_acc, (1, 99))
                ax = orb_axes[idx // 3, idx % 3]
                final_imshow_orb = ax.imshow(orb_acc, cmap=self.plot_params["color_map"], vmin=vmin, vmax=vmax)
                ax.set_title(f"Digit {digit} (ORB)", fontsize=self.plot_params["font_size"])
                ax.axis("off")

        pbar.close()

        # Format and save Sobel figure
        if final_imshow_sobel is not None:
            sobel_fig.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.05, wspace=0.3, hspace=0.3)
            cbar_ax = sobel_fig.add_axes([0.9, 0.15, 0.02, 0.7])
            sobel_fig.colorbar(final_imshow_sobel, cax=cbar_ax, label="Feature Intensity (Sobel)")
            sobel_fig.suptitle("Local Feature Consistency (Sobel)", fontsize=self.plot_params["font_size"] + 4)
            sobel_path = os.path.join(self.output_dir, "local_feature_consistency_sobel.png")
            sobel_fig.savefig(sobel_path, bbox_inches="tight")
            plt.close(sobel_fig)

        # Format and save ORB figure
        if final_imshow_orb is not None:
            orb_fig.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.05, wspace=0.3, hspace=0.3)
            cbar_ax = orb_fig.add_axes([0.9, 0.15, 0.02, 0.7])
            orb_fig.colorbar(final_imshow_orb, cax=cbar_ax, label="Feature Intensity (ORB)")
            orb_fig.suptitle("Local Feature Consistency (ORB)", fontsize=self.plot_params["font_size"] + 4)
            orb_path = os.path.join(self.output_dir, "local_feature_consistency_orb.png")
            orb_fig.savefig(orb_path, bbox_inches="tight")
            plt.close(orb_fig)

        self.log(f"✅ [Step 11 Complete] Sobel map saved: {sobel_path}")
        self.log(f"✅ [Step 11 Complete] ORB map saved: {orb_path}\n")

        return sobel_data, orb_data

    def step_12_digit_heatmap_grid(self) -> None:
        """
        Generates a heatmap for each digit (1–9) and plots them in a single composite 3×3 grid.

        This method calculates the average intensity heatmap for each digit by combining all available
        images for that digit. The heatmaps are then displayed in a 3x3 grid, and a colorbar is added
        to indicate the intensity values.

        The final grid is saved as an image in the output directory.

        Returns:
            None: This method does not return any value but saves the resulting heatmap grid to the output directory.
        """
        self.log("\n🔥 [Step 12] Generating per-digit heatmap grid (1–9)...")

        fig, axes = plt.subplots(3, 3, figsize=self.plot_params["figsize"])
        fig.suptitle("Digit Heatmaps (1–9)", fontsize=self.plot_params["font_size"] + 2)

        digits = list(map(str, range(1, 10)))
        reference_im = None

        with tqdm(total=len(digits), desc="Building heatmaps", ncols=100) as bar:
            for i, digit in enumerate(digits):
                row, col = divmod(i, 3)
                ax = axes[row, col]

                path = self.digit_dirs.get(digit)
                if not path or not os.path.exists(path):
                    ax.set_title(f"Digit {digit} (missing)", fontsize=self.plot_params["font_size"])
                    ax.axis("off")
                    bar.update(1)
                    continue

                image_paths = [
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                if not image_paths:
                    ax.set_title(f"Digit {digit} (empty)", fontsize=self.plot_params["font_size"])
                    ax.axis("off")
                    bar.update(1)
                    continue

                try:
                    acc = None
                    for p in image_paths:
                        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                        img = img.astype(np.float32) / 255.0
                        if acc is None:
                            acc = np.zeros_like(img)
                        acc += img
                    heatmap = acc / len(image_paths)

                    reference_im = ax.imshow(
                        heatmap,
                        cmap=self.plot_params["color_map"],
                        interpolation="nearest"
                    )
                    ax.set_title(f"Digit {digit}", fontsize=self.plot_params["font_size"])
                    ax.axis("off")
                except Exception as e:
                    self.log(f"❌ [Step 12] Failed for digit {digit}: {e}")
                    ax.axis("off")

                bar.update(1)

        # Manually adjust the layout
        plt.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.05, wspace=0.3, hspace=0.3)

        # Add colorbar on the right
        if reference_im is not None:
            cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
            fig.colorbar(reference_im, cax=cbar_ax, label="Normalized Intensity")

        # Save the figure
        output_path = os.path.join(self.output_dir, "digit_heatmap_grid.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        self.log(f"🖼️ [Step 12] Saved heatmap grid → {output_path}")
        self.log("✅ [Step 12 Complete] Digit heatmap grid generated.\n")

    def apply_alternating_row_colors(self, table) -> None:
        """
        Applies alternating row colors (light grey for even rows and white for odd rows) to the table.

        Args:
            table: The table object to which row colors will be applied.

        Returns:
            None: Modifies the table in place.
        """
        for (row, col), cell in table.get_celld().items():
            cell.set_fontsize(9)
            if row == 0:
                cell.set_text_props(weight="bold", color="black")
            else:
                if row % 2 == 0:
                    cell.set_facecolor("#e0e0e0")  # Medium light grey for even rows
                else:
                    cell.set_facecolor("white")  # White for odd rows

    def generate_table_1_dataset_summary(
            self,
            class_stats: Dict[str, Dict[str, int]],
            image_size: str
    ) -> pd.DataFrame:
        """
        Generates Table 1: Dataset Summary with improved formatting. Saves it as both a CSV and PNG image.

        Args:
            class_stats (Dict[str, Dict[str, int]]): Output from step_1_class_distribution().
            image_size (str): Output from step_2_check_image_dimensions(), e.g., "32×32".

        Returns:
            pd.DataFrame: The DataFrame containing the summary table.
        """
        self.log("📋 Generating Table 1: Dataset Summary...")

        summary_rows = []
        total_all = clean_all = distorted_all = other_all = 0

        for digit in sorted(class_stats.keys(), key=int):
            stats = class_stats[digit]
            total = stats["total"]
            clean = stats["clean"]
            distorted = stats["distorted"]
            other = total - clean - distorted

            clean_pct = 100 * clean / total if total > 0 else 0
            distorted_pct = 100 * distorted / total if total > 0 else 0
            other_pct = 100 * other / total if total > 0 else 0

            # Ensure 1 decimal place for percentages
            summary_rows.append({
                "Digit": digit,
                "Total": f"{total:,}",  # Format total with thousands separator
                "Clean %": f"{clean_pct:.1f}",
                "Distorted %": f"{distorted_pct:.1f}",
                "Other %": f"{other_pct:.1f}",
                "Image Size": image_size
            })

            total_all += total
            clean_all += clean
            distorted_all += distorted
            other_all += other

        # Totals row
        summary_rows.append({
            "Digit": "Total",
            "Total": f"{total_all:,}",  # Format total with thousands separator
            "Clean %": f"{100 * clean_all / total_all:.1f}" if total_all > 0 else "0.0",
            "Distorted %": f"{100 * distorted_all / total_all:.1f}" if total_all > 0 else "0.0",
            "Other %": f"{100 * other_all / total_all:.1f}" if total_all > 0 else "0.0",
            "Image Size": image_size
        })

        df = pd.DataFrame(summary_rows)

        # Save CSV
        csv_path = os.path.join(self.output_dir, "table_1_dataset_summary.csv")
        df.to_csv(csv_path, index=False)

        # Save image with enhanced formatting
        fig, ax = plt.subplots(figsize=(8, 0.1 * len(df) + 1.5))  # Adjust height based on row count
        ax.axis("off")

        # Add title above the table
        fig.text(0.5, 0.98, "Table 1. Dataset Summary", ha="center", va="top",
                 fontsize=13, weight="bold", color="black")

        # Create the table
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
            colColours=["#ADD8E6"] * len(df.columns),
        )

        self.apply_alternating_row_colors(table)

        # Style cells
        for (row, col), cell in table.get_celld().items():
            cell.set_fontsize(9)
            if row == 0:
                cell.set_text_props(weight="bold", color="black")

        # Tightly crop white space
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)

        # Save to file
        image_path = os.path.join(self.output_dir, "table_1_dataset_summary.png")
        plt.savefig(image_path, bbox_inches="tight", dpi=150, pad_inches=0.05)
        plt.close()

        # Log
        self.log(f"📄 Saved CSV → {csv_path}")
        self.log(f"📋 Saved table image → {image_path}")
        self.log("✅ [Table 1] Dataset Summary... generated successfully.\n")

        return df

    def generate_table_2_image_quality_issues(
            self,
            corrupt_paths: List[str],
            blurry_images: List[Tuple[str, float]],
            partial_paths: List[str]
    ) -> None:
        """
        Generates and saves Table 2: Image Quality Issues, combining image issue counts per digit.

        This table summarizes how many images per digit are corrupted, blurry, or partial.
        It also includes percentages for blurry and partial images.

        Args:
            corrupt_paths (List[str]): List of image paths flagged as corrupt (from Step 5).
            blurry_images (List[Tuple[str, float]]): List of blurry images and their sharpness score (from Step 7).
            partial_paths (List[str]): List of image paths where digits are partial (from Step 9).

        Returns:
            None: Saves a CSV file and table image, and appends a table header to `self.report_lines`.
        """
        self.log("📋 Generating Table 2: Image Quality Issues...")

        def extract_digit(path: str) -> str:
            return os.path.basename(os.path.dirname(path))

        corrupt_counts = defaultdict(int)
        for path in corrupt_paths:
            corrupt_counts[extract_digit(path)] += 1

        blurry_counts = defaultdict(int)
        for path, _ in blurry_images:
            blurry_counts[extract_digit(path)] += 1

        partial_counts = defaultdict(int)
        for path in partial_paths:
            partial_counts[extract_digit(path)] += 1

        total_per_digit = {
            digit: len(os.listdir(path)) for digit, path in self.digit_dirs.items()
        }

        digits = sorted(total_per_digit.keys(), key=int)
        rows = []
        total_corrupt = total_blurry = total_partial = total_images = 0

        for digit in digits:
            total = total_per_digit[digit]
            corrupt = corrupt_counts.get(digit, 0)
            blurry = blurry_counts.get(digit, 0)
            partial = partial_counts.get(digit, 0)
            blurry_pct = (blurry / total * 100) if total else 0
            partial_pct = (partial / total * 100) if total else 0

            rows.append([
                digit,
                f"{corrupt:,}",
                f"{blurry:,}",
                f"{partial:,}",
                f"{blurry_pct:.1f}%",
                f"{partial_pct:.1f}%"
            ])

            total_corrupt += corrupt
            total_blurry += blurry
            total_partial += partial
            total_images += total

        # Add total row
        rows.append([
            "Total",
            f"{total_corrupt:,}",
            f"{total_blurry:,}",
            f"{total_partial:,}",
            f"{(total_blurry / total_images * 100):.1f}%" if total_images else "0.0%",
            f"{(total_partial / total_images * 100):.1f}%" if total_images else "0.0%"
        ])

        df = pd.DataFrame(rows, columns=[
            "Digit", "Corrupt Count", "Blurry Count", "Partial Count", "% Blurry", "% Partial"
        ])

        # Save CSV
        csv_path = os.path.join(self.output_dir, "table_2_image_quality_issues.csv")
        df.to_csv(csv_path, index=False)

        # Save image
        fig, ax = plt.subplots(figsize=(8, 0.1 * len(df) + 1.5))
        ax.axis("off")

        fig.text(0.5, 0.98, "Table 2. Image Quality Issues Summary", ha="center", va="top",
                 fontsize=13, weight="bold", color="black")

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
            colColours=["#add8e6"] * len(df.columns),
        )

        # Apply alternating row colors
        self.apply_alternating_row_colors(table)

        # Style cells
        for (row, col), cell in table.get_celld().items():
            cell.set_fontsize(9)
            if row == 0:
                cell.set_text_props(weight="bold", color="black")

        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)
        image_path = os.path.join(self.output_dir, "table_2_image_quality_issues.png")
        plt.savefig(image_path, bbox_inches="tight", dpi=150, pad_inches=0.05)
        plt.close()

        self.log(f"📄 Saved CSV → {csv_path}")
        self.log(f"📋 Saved table image → {image_path}")
        self.log("✅ [Table 2] Image Quality Issues Summary generated successfully.\n")

    def generate_table_3_duplicate_summary(self, all_duplicates_by_digit: Dict[str, List[List[str]]]) -> None:
        """
        Generates and saves Table 3: Duplicate Image Summary.

        Args:
            all_duplicates_by_digit (Dict[str, List[List[str]]]): Grouped duplicate images per digit.

        Returns:
            None: Saves a CSV file and table image, and appends a table header to self.report_lines.
        """
        self.log("📋 Generating Table 3: Duplicate Image Summary...")

        digits = sorted(self.digit_dirs.keys(), key=int)
        rows = []

        total_groups = total_images = 0
        max_group_sizes = []

        for digit in digits:
            groups = all_duplicates_by_digit.get(digit, [])
            num_groups = len(groups)
            num_images = sum(len(g) for g in groups)
            max_group_size = max((len(g) for g in groups), default=0)

            total_groups += num_groups
            total_images += num_images
            max_group_sizes.append(max_group_size)

            rows.append([
                digit,
                f"{num_groups:,}",
                f"{num_images:,}",
                f"{max_group_size:,}"
            ])

        # Total row
        avg_max_group_size = sum(max_group_sizes) / len(max_group_sizes) if max_group_sizes else 0
        rows.append([
            "Total",
            f"{total_groups:,}",
            f"{total_images:,}",
            f"{avg_max_group_size:.1f}"
        ])

        df = pd.DataFrame(rows, columns=[
            "Digit", "Duplicate Groups", "Images in Duplicates", "Max Group Size"
        ])

        # Save CSV
        csv_path = os.path.join(self.output_dir, "table_3_duplicate_summary.csv")
        df.to_csv(csv_path, index=False)

        # Save table as image
        fig, ax = plt.subplots(figsize=(8, 0.1 * len(df) + 1.5))
        ax.axis("off")

        fig.text(0.5, 0.98, "Table 3. Duplicate Image Summary", ha="center", va="top",
                 fontsize=13, weight="bold", color="black")

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
            colColours=["#add8e6"] * len(df.columns),
        )

        self.apply_alternating_row_colors(table)

        for (row, col), cell in table.get_celld().items():
            cell.set_fontsize(9)
            if row == 0:
                cell.set_text_props(weight="bold", color="black")

        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)

        image_path = os.path.join(self.output_dir, "table_3_duplicate_summary.png")
        plt.savefig(image_path, bbox_inches="tight", dpi=150, pad_inches=0.05)
        plt.close()

        self.log(f"📄 Saved CSV → {csv_path}")
        self.log(f"📋 Saved table image → {image_path}")
        self.log("✅ [Table 3] Duplicate Image Summary generated successfully.\n")

    def generate_table_4_dataset_diversity(self, diversity_data: List[Dict[str, Union[str, float, int]]]) -> None:
        """
        Generates and saves Table 4: Dataset Diversity based on perceptual hash distance.

        Args:
            diversity_data (List[Dict[str, Union[str, float, int]]]): List of per-digit diversity metrics from Step 8.

        Returns:
            None: Saves a CSV file and table image, and appends a table header to self.report_lines.
        """
        self.log("📋 Generating Table 4: Dataset Diversity...")

        # Build DataFrame
        df = pd.DataFrame(diversity_data)
        df['Digit'] = df['Digit'].astype('int64')
        df['Unique Samples'] = df['Unique Samples'].astype('int64')
        df['Diversity Score'] = df['Diversity Score'].astype(float).round(2)
        df = df.sort_values(by="Digit").reset_index(drop=True)

        # Compute summary metrics
        avg_div_score = df['Diversity Score'].mean()
        combined_unique = df['Unique Samples'].sum()

        # Save raw values to CSV
        csv_path = os.path.join(self.output_dir, "table_4_dataset_diversity.csv")
        df.to_csv(csv_path, index=False)

        # Format values for display in the image
        df_display = df.copy()
        df_display['Digit'] = df_display['Digit'].astype(str)
        df_display['Unique Samples'] = df_display['Unique Samples'].apply(lambda x: f"{x:,}")
        df_display['Diversity Score'] = df_display['Diversity Score'].apply(lambda x: f"{x:.2f}")

        # Create compact figure
        fig_height = 0.08 * len(df_display) + 1.6
        fig, ax = plt.subplots(figsize=(8, fig_height))
        ax.axis("off")

        # Title
        fig.text(0.5, 0.97, "Table 4. Dataset Diversity", ha="center", va="top",
                 fontsize=13, weight="bold", color="black")

        # Table with alternating row colors
        table = ax.table(
            cellText=df_display.values,
            colLabels=df_display.columns,
            cellLoc="center",
            loc="center",
            colColours=["#add8e6"] * len(df_display.columns),
        )

        self.apply_alternating_row_colors(table)

        # Summary text with formatted combined_unique
        summary_y = 0.02
        fig.text(0.05, summary_y, f"Avg Diversity Score: {avg_div_score:.2f}", ha="left", fontsize=9.5)
        fig.text(0.05, summary_y - 0.06, f"Combined Unique Samples: {combined_unique:,}", ha="left", fontsize=9.5)

        # Layout and save
        plt.subplots_adjust(left=0.05, right=0.95, top=0.86, bottom=0.05)
        image_path = os.path.join(self.output_dir, "table_4_dataset_diversity.png")
        plt.savefig(image_path, bbox_inches="tight", dpi=150, pad_inches=0.01)
        plt.close()

        self.log(f"📄 Saved CSV → {csv_path}")
        self.log(f"📋 Saved table image → {image_path}")
        self.log("✅ [Table 4] Dataset Diversity generated successfully.\n")

    def generate_table_5_feature_consistency(self, sobel_data: List[Dict[str, Union[str, float, int]]],
                                             orb_data: List[Dict[str, Union[str, float, int]]]) -> None:
        """
        Generates and saves Table 5: Feature Consistency based on Sobel and ORB analysis.

        Args:
            sobel_data (List[Dict]): List of per-digit Sobel metrics.
            orb_data (List[Dict]): List of per-digit ORB metrics.

        Returns:
            None: Saves a CSV file and table image, and appends a table header to self.report_lines.
        """
        self.log("📋 Generating Table 5: Feature Consistency (Sobel & ORB)...")

        # Merge Sobel and ORB data based on 'Digit'
        feature_data = []
        for sobel_row in sobel_data:
            digit = sobel_row["Digit"]
            orb_row = next((orb for orb in orb_data if orb["Digit"] == digit), {})
            feature_data.append({
                "Digit": digit,
                "Avg Sobel Intensity": sobel_row.get("Avg Sobel Intensity", "—"),
                "Avg ORB Keypoints": orb_row.get("Avg ORB Keypoints", "—")
            })

        # Create DataFrame
        df = pd.DataFrame(feature_data)

        # Format 'Digit' as integers (if needed)
        df['Digit'] = df['Digit'].astype(int)

        # Ensure 'Avg Sobel Intensity' and 'Avg ORB Keypoints' are always rounded to 2 decimals
        df['Avg Sobel Intensity'] = df['Avg Sobel Intensity'].apply(
            lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

        df['Avg ORB Keypoints'] = df['Avg ORB Keypoints'].apply(
            lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

        # Calculate overall averages, excluding '—' values (non-numeric)
        sobel_numeric = pd.to_numeric(df['Avg Sobel Intensity'], errors='coerce')
        orb_numeric = pd.to_numeric(df['Avg ORB Keypoints'], errors='coerce')

        avg_sobel_intensity = sobel_numeric.mean()
        avg_orb_keypoints = orb_numeric.mean()

        # Add final row for overall averages, but do it after sorting
        overall_row = pd.DataFrame([{
            "Digit": "Overall",
            "Avg Sobel Intensity": f"{avg_sobel_intensity:.2f}",
            "Avg ORB Keypoints": f"{avg_orb_keypoints:.2f}"
        }])

        # Sort the DataFrame by 'Digit', ensuring 'Overall' comes last
        df = df.sort_values(by="Digit", key=lambda x: x.apply(lambda y: (y != 'Overall', y)))

        # Now add the "Overall" row at the end
        df = pd.concat([df, overall_row], ignore_index=True)

        # Save as CSV
        csv_path = os.path.join(self.output_dir, "table_5_feature_consistency.csv")
        df.to_csv(csv_path, index=False)

        # Save as styled image
        fig, ax = plt.subplots(figsize=(8, 0.1 * len(df) + 1.5))
        ax.axis("off")

        fig.text(0.5, 0.98, "Table 5. Feature Consistency (Sobel & ORB)", ha="center", va="top",
                 fontsize=13, weight="bold", color="black")

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
            colColours=["#add8e6"] * len(df.columns),
        )

        self.apply_alternating_row_colors(table)

        # Formatting table cells
        for (row, col), cell in table.get_celld().items():
            cell.set_fontsize(9)
            if row == 0:
                cell.set_text_props(weight="bold", color="black")

        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)
        image_path = os.path.join(self.output_dir, "table_5_feature_consistency.png")
        plt.savefig(image_path, bbox_inches="tight", dpi=150, pad_inches=0.05)
        plt.close()

        self.log(f"📄 Saved CSV → {csv_path}")
        self.log(f"📋 Saved table image → {image_path}")
        self.log("✅ [Table 5] Feature Consistency (Sobel & ORB) generated successfully.\n")

    def run_full_evaluation(self) -> None:
        """
        Runs the complete dataset evaluation pipeline and generates various diagnostic reports.

        This method sequentially performs multiple analyses on the dataset to assess:
            - Class distribution balance.
            - Image size consistency.
            - Visual inspection via sample grid.
            - Intensity histogram statistics.
            - Detection of corrupt, blurry, or partially visible digits.
            - Heatmap analysis for digit centering.
            - Feature consistency across digits using Sobel and ORB methods.
            - Duplicate image detection.
            - Dataset diversity analysis.
            - Aggregated metrics into summary tables.

        Artifacts generated include:
            - PNG visualizations (e.g., heatmaps, histograms, grids).
            - CSV tables with numerical results.
            - A compiled textual report saved in the output directory.

        The final report consolidates all findings and helps evaluate the dataset’s quality and reliability
        before use in model training or benchmarking.

        Returns:
            None
        """
        self.log("🧪 Running full dataset evaluation pipeline...")
        self.log(f"Dataset path: {self.dataset_path}")
        self.log(f"Evaluation time: {datetime.datetime.now()}\n")

        # Step-wise pipeline
        class_stats = self.step_1_class_distribution()
        image_size = self.step_2_check_image_dimensions()
        self.step_3_visualize_sample_grid()
        self.step_4_intensity_histograms()
        corrupt_paths = self.step_5_detect_corrupt_images()
        self.step_6_digit_centering_heatmap()
        blurry_images = self.step_7_detect_blurry_images()
        diversity_data = self.step_8_estimate_dataset_diversity()
        partial_paths = self.step_9_detect_partial_digits()
        _, all_duplicates_by_digit = self.step_10_detect_duplicate_images()
        sobel_data, orb_data = self.step_11_local_feature_consistency(samples_per_digit=10000)
        self.step_12_digit_heatmap_grid()

        # Table/report generation
        self.generate_table_1_dataset_summary(class_stats, image_size)
        self.generate_table_2_image_quality_issues(corrupt_paths, blurry_images, partial_paths)
        self.generate_table_3_duplicate_summary(all_duplicates_by_digit)
        self.generate_table_4_dataset_diversity(diversity_data)
        self.generate_table_5_feature_consistency(sobel_data, orb_data)

        # Save the final text report
        report_path = os.path.join(self.output_dir, "dataset_evaluation_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.report_lines))

        self.log(f"\n✅ Evaluation complete. Report saved to '{report_path}'.")


if __name__ == "__main__":
    evaluator = DigitDatasetEvaluator(dataset_path="digit_dataset")
    evaluator.run_full_evaluation()
    # diversity_data = evaluator.step_8_estimate_dataset_diversity()
    # evaluator.generate_table_4_dataset_diversity(diversity_data)
    # class_stats = evaluator.step_1_class_distribution()
    # image_size = evaluator.step_2_check_image_dimensions()
    # evaluator.generate_table_1_dataset_summary(class_stats, image_size)
