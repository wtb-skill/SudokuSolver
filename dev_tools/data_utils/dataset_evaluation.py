import os
import cv2
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

    def log(self, text: str) -> None:
        """Logs messages to console and saves to the evaluation report."""
        print(text)
        self.report_lines.append(text)

    def class_distribution(self) -> Dict[str, Dict[str, int]]:
        """Analyzes and logs the distribution of clean vs distorted images per digit."""
        self.log("\n📊 [Step 1] Starting class distribution analysis...")
        summary: Dict[str, Dict[str, int]] = {}

        for digit, path in self.digit_dirs.items():
            if not os.path.exists(path):
                self.log(f"   ⛔ Skipping missing directory: {path}")
                continue
            images = os.listdir(path)
            clean = len([img for img in images if "clean" in img])
            distorted = len(images) - clean
            summary[digit] = {"total": len(images), "clean": clean, "distorted": distorted}
            self.log(f"   - Digit {digit}: {summary[digit]}")

        self.log("✅ [Step 1 Complete] Class distribution analysis done.\n")
        return summary

    def visualize_sample_grid(self, samples_per_digit: int = 5) -> None:
        """Saves a grid visualization of random samples for each digit."""
        self.log("\n🖼️ [Step 2] Generating sample grid visualization...")

        fig, axs = plt.subplots(9, samples_per_digit, figsize=(samples_per_digit * 1.5, 13))
        for row, digit in enumerate(self.digit_dirs):
            imgs = os.listdir(self.digit_dirs[digit])
            samples = random.sample(imgs, min(samples_per_digit, len(imgs)))
            for col, img_name in enumerate(samples):
                img_path = os.path.join(self.digit_dirs[digit], img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                axs[row, col].imshow(img, cmap='gray')
                axs[row, col].axis("off")
                if col == 0:
                    axs[row, col].set_title(f"Digit {digit}")

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "sample_grid.png")
        plt.savefig(output_path)
        plt.close()
        self.log(f"✅ [Step 2 Complete] Sample image grid saved as '{output_path}'\n")

    def intensity_histograms(self, sample_size: int = 1000) -> None:
        """Plots histograms of pixel intensity values for clean and distorted images."""
        self.log("\n📈 [Step 3] Generating pixel intensity histograms...")

        clean_pixels, distorted_pixels = [], []
        total_images = sum(
            min(sample_size, len(os.listdir(path)))
            for path in self.digit_dirs.values()
            if os.path.exists(path)
        )

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
                        (clean_pixels if "clean" in img_name else distorted_pixels).extend(img.flatten())
                    pbar.update(1)

        with tqdm(total=1, desc="📊 Plotting & Saving", unit="task") as pbar:
            plt.hist(clean_pixels, bins=50, alpha=0.6, label="Clean", color="green")
            plt.hist(distorted_pixels, bins=50, alpha=0.6, label="Distorted", color="red")
            plt.title("Pixel Intensity Histogram")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.legend()
            output_path = os.path.join(self.output_dir, "pixel_histogram.png")
            plt.savefig(output_path)
            plt.close()
            pbar.update(1)

        self.log(f"✅ [Step 3 Complete] Histogram saved as '{output_path}'\n")

    def detect_corrupt_images(self) -> List[str]:
        """Detects images that are blank or nearly blank."""
        self.log("\n🔍 [Step 4] Scanning for corrupt images...")
        corrupt: List[str] = []

        for digit, path in self.digit_dirs.items():
            if not os.path.exists(path):
                self.log(f"   ⛔ Directory missing: {path}")
                continue
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None or img.max() - img.min() < 10:
                    corrupt.append(img_path)

        if corrupt:
            self.log(f"⚠️ Found {len(corrupt)} potentially corrupt images. Showing first 10:")
            for c in corrupt[:10]:
                self.log(f"    - {c}")
        else:
            self.log("✅ No corrupt images detected.")

        self.log("✅ [Step 4 Complete] Corrupt image scan finished.\n")
        return corrupt

    def digit_centering_heatmap(self, samples_per_digit: int = 100) -> None:
        """Creates a heatmap to visualize average digit centering across the dataset."""
        self.log("\n🔥 [Step 5] Generating digit centering heatmap...")

        accumulator = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        total = 0

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
                    norm_img = img.astype(np.float32) / 255.0
                    accumulator += norm_img
                    total += 1

        if total > 0:
            accumulator /= total
            plt.imshow(accumulator, cmap="hot")
            plt.colorbar()
            plt.title("Digit Centering Heatmap (all digits)")
            output_path = os.path.join(self.output_dir, "centering_heatmap.png")
            plt.savefig(output_path)
            plt.close()
            self.log(f"✅ [Step 5 Complete] Digit centering heatmap saved as '{output_path}'\n")
        else:
            self.log("⚠️ No images found for centering heatmap.")

    def detect_partial_digits(self, margin: int = 0, samples_per_digit: int = 10) -> List[str]:
        """
        Detects images where digits touch the image frame (partial digits).

        Args:
            margin (int): Margin in pixels to consider near-edge proximity. Defaults to 0.
            samples_per_digit (int): Max number of flagged samples to display per digit.

        Returns:
            List[str]: List of paths of suspected partial digit images.
        """
        self.log("\n🔍 [Step 6] Detecting partial digits (those touching the image frame)...")

        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, "partial_digits_grid.png")
        partials: List[str] = []
        partials_by_digit: Dict[str, List[str]] = defaultdict(list)

        # Calculate total images for progress bar
        total_images = sum(len(os.listdir(path)) for path in self.digit_dirs.values())
        with tqdm(total=total_images, desc="Processing all digits") as pbar:
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
            self.log("✅ [Step 6 Complete] No partial digits detected based on edge proximity.")
            return []

        # Constructing visualization grid
        rows = len(partials_by_digit)
        cols = samples_per_digit
        grid_img = np.ones((rows * self.image_size, cols * self.image_size), dtype=np.uint8) * 255

        self.log("🖼️ [Step 6] Building composite image grid of partial digits...")
        for row_idx, digit in tqdm(enumerate(sorted(partials_by_digit.keys())), desc="Creating visualization",
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

        cv2.imwrite(save_path, grid_img)
        self.log(f"✅ [Step 6 Complete] Composite image of partial digits saved as '{save_path}'")
        self.log(f"🟠 [Step 6 Complete] Found {len(partials)} potentially partial digit images.\n")

        return partials

    def _compute_hash(self, img_path: str, hash_func=imagehash.phash) -> Optional[Tuple[imagehash.ImageHash, str]]:
        """Computes perceptual hash for a given image."""
        try:
            img = Image.open(img_path).convert("L").resize((32, 32))
            img_hash = hash_func(img)
            return img_hash, img_path
        except Exception:
            return None

    def detect_duplicate_images(self, hash_func=imagehash.phash, threshold: int = 0,
                                max_groups: int = 5, samples_per_group: int = 5) -> np.ndarray:
        self.log("\n🔍 [Step 7] Detecting duplicate or near-duplicate images within each digit class...")

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

        # ✅ Force progress bar to 100% if it ended early
        if find_bar.n < find_bar.total:
            find_bar.update(find_bar.total - find_bar.n)

        # 3. Visualize duplicates per digit
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

        # 4. Create final grid image
        final_image = self._concatenate_3x3_grid(digit_visuals)

        save_path = os.path.join(self.output_dir, "duplicate_images_grid.png")
        cv2.imwrite(save_path, final_image)
        self.log(f"✅ [Step 7 Complete] Saved composite image of duplicate digits → {save_path}\n")

        return final_image

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

    def local_feature_consistency(self, method: str = "sobel", samples_per_digit: int = 500) -> None:
        """
        Analyzes local feature consistency across digits using either Sobel edges or ORB keypoints.
        All digit feature maps are combined into a single figure.

        Args:
            method (str): Feature extraction method, either "sobel" or "orb".
            samples_per_digit (int): Number of samples to use per digit class.
        """
        assert method in ["sobel", "orb"], "Method must be 'sobel' or 'orb'"
        self.log(f"\n🔍 [Step 8] Checking local feature consistency using '{method}' method...")

        orb_detector = None
        if method == "orb":
            orb_detector = cv2.ORB_create(
                nfeatures=5000,  # more keypoints
                edgeThreshold=5,  # less strict
                fastThreshold=5  # detect weaker corners
            )

        os.makedirs(self.output_dir, exist_ok=True)

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        total_images = samples_per_digit * len(self.digit_dirs)
        pbar = tqdm(total=total_images, desc="Processing images", ncols=100)

        for idx, (digit, path) in enumerate(self.digit_dirs.items()):
            images = os.listdir(path)
            random.shuffle(images)
            selected = images[:samples_per_digit]

            accumulator = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            count = 0

            for img_name in selected:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    pbar.update(1)
                    continue

                if method == "sobel":
                    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
                    norm_mag = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
                    accumulator += norm_mag

                elif method == "orb" and orb_detector is not None:
                    keypoints = orb_detector.detect(img, None)
                    kp_img = np.zeros_like(img, dtype=np.float32)
                    for kp in keypoints:
                        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
                        if 0 <= x < self.image_size and 0 <= y < self.image_size:
                            kp_img[y, x] = 1.0

                    kp_img = cv2.GaussianBlur(kp_img, (5, 5), sigmaX=1, sigmaY=1)
                    accumulator += kp_img

                count += 1
                pbar.update(1)  # ✅ update after every image

            if count > 0:
                vmin, vmax = np.percentile(accumulator, (1, 99))
                ax = axes[idx // 3, idx % 3]
                im = ax.imshow(accumulator, cmap="hot", vmin=vmin, vmax=vmax)
                ax.set_title(f"Digit {digit}", fontsize=14)
                ax.axis('off')
            else:
                self.log(f"⚠️ No images found for digit {digit} during local feature consistency check.")

        pbar.close()

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Colorbar
        fig.colorbar(im, cax=cbar_ax)

        save_path = os.path.join(self.output_dir, f"local_feature_consistency_{method}.png")
        plt.suptitle(f"Local Feature Consistency ({method.upper()})", fontsize=20)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        self.log(f"✅ [Step 8 Complete] Combined local feature map saved: {save_path}\n")

    def run_full_evaluation(self) -> None:
        """Runs a full evaluation pipeline and generates report artifacts."""
        self.log("🧪 Running full dataset evaluation pipeline...")
        self.log(f"Dataset path: {self.dataset_path}")
        self.log(f"Evaluation time: {datetime.datetime.now()}\n")

        self.class_distribution()
        self.visualize_sample_grid()
        self.intensity_histograms()
        self.detect_corrupt_images()
        self.digit_centering_heatmap()
        self.detect_partial_digits()
        self.detect_duplicate_images()
        self.local_feature_consistency(method="sobel")
        self.local_feature_consistency(method="orb")

        report_path = os.path.join(self.output_dir, "dataset_evaluation_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.report_lines))

        self.log(f"\n✅ Evaluation complete. Report saved to '{report_path}'.")

if __name__ == "__main__":
    evaluator = DigitDatasetEvaluator(dataset_path="digit_dataset")
    evaluator.run_full_evaluation()
    # evaluator.detect_partial_digits()