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

    def step_1_class_distribution(self) -> Dict[str, Dict[str, int]]:
        """Analyzes class distribution, imbalance, and generates histogram."""
        self.log("\n📊 [Step 1] Starting class distribution & balance analysis...")
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

        # ⚖️ Class imbalance calculation
        class_totals = {d: summary[d]["total"] for d in summary}
        min_class = min(class_totals, key=class_totals.get)
        max_class = max(class_totals, key=class_totals.get)
        imbalance_ratio = class_totals[max_class] / max(1, class_totals[min_class])

        self.log(f"[Step 1] Class sample counts: {class_totals}")
        self.log(f"[Step 1] Imbalance ratio (max/min): {imbalance_ratio:.2f}")
        if imbalance_ratio > 3:
            self.log("⚠️ [Step 1] Warning: High class imbalance detected.")

        # 📊 Generate grouped histogram
        try:
            digits = sorted(summary.keys(), key=int)
            totals = [summary[d]["total"] for d in digits]
            cleans = [summary[d]["clean"] for d in digits]
            distorteds = [summary[d]["distorted"] for d in digits]

            x = range(len(digits))
            width = 0.3

            plt.figure(figsize=(10, 6))
            plt.bar([i - width for i in x], totals, width=width, label="Total", color="skyblue")
            plt.bar(x, cleans, width=width, label="Clean", color="green")
            plt.bar([i + width for i in x], distorteds, width=width, label="Distorted", color="red")

            plt.xlabel("Digit Class")
            plt.ylabel("Number of Images")
            plt.title("Image Distribution per Digit (Total / Clean / Distorted)")
            plt.xticks(x, digits)
            plt.legend()
            plt.tight_layout()

            save_path = os.path.join(self.output_dir, "class_distribution_histogram.png")
            plt.savefig(save_path)
            plt.close()

            self.log(f"📊 [Step 1] Saved class distribution histogram → {save_path}")
        except Exception as e:
            self.log(f"[Step 1] ❌ Failed to generate histogram: {e}")

        self.log("✅ [Step 1 Complete] Class distribution & balance analysis done.\n")
        return summary

    def step_2_check_image_dimensions(self) -> None:
        """Check for inconsistent image dimensions across dataset."""
        self.log("\n📏 [Step 2] Checking image dimensions...")

        dim_counter = defaultdict(int)
        dim_examples = defaultdict(list)

        all_images = [(digit, os.path.join(path, img))
                      for digit, path in self.digit_dirs.items()
                      for img in os.listdir(path)]

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

        # 📊 Generate histogram
        labels = [f"{h}x{w}" for (h, w) in dim_counter.keys()]
        counts = list(dim_counter.values())

        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts, color='skyblue', edgecolor='black')
        plt.title("Image Dimension Distribution")
        plt.xlabel("Dimensions (HxW)")
        plt.ylabel("Number of Images")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, "image_dimension_histogram.png")
        plt.savefig(save_path)
        plt.close()

        self.log(f"📊 [Step 2] Saved image dimension histogram → {save_path}")
        self.log("✅ [Step 2 Complete] Image dimension check done.\n")

    def step_3_visualize_sample_grid(self, samples_per_digit: int = 5) -> None:
        """Saves a grid visualization of random samples for each digit."""
        self.log("\n🖼️ [Step 3] Generating sample grid visualization...")

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
        self.log(f"✅ [Step 3 Complete] Sample image grid saved as '{output_path}'\n")

    def step_4_intensity_histograms(self, sample_size: int = 1000) -> None:
        """Plots histograms of pixel intensity values for clean and distorted images."""
        self.log("\n📈 [Step 4] Generating pixel intensity histograms...")

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

        self.log(f"✅ [Step 4 Complete] Histogram saved as '{output_path}'\n")

    def step_5_detect_corrupt_images(self) -> List[str]:
        """Detects images that are blank or nearly blank."""
        self.log("\n🔍 [Step 5] Scanning for corrupt images...")
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
            self.log(f"⚠️ [Step 5] Found {len(corrupt)} potentially corrupt images. Showing first 10:")
            for c in corrupt[:10]:
                self.log(f"    - {c}")
        else:
            self.log("✅ [Step 5] No corrupt images detected.")

        self.log("✅ [Step 5 Complete] Corrupt image scan finished.\n")
        return corrupt

    def step_6_digit_centering_heatmap(self, samples_per_digit: int = 100) -> None:
        """Creates a heatmap to visualize average digit centering across the dataset."""
        self.log("\n🔥 [Step 6] Generating digit centering heatmap...")

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
            self.log(f"✅ [Step 6 Complete] Digit centering heatmap saved as '{output_path}'\n")
        else:
            self.log("⚠️ [Step 6 Complete] No images found for centering heatmap.")

    def step_7_detect_blurry_images(self, threshold: float = 100.0) -> None:
        """Detects blurry images using variance of Laplacian."""
        self.log("\n🔍 [Step 7] Detecting blurry images...")

        blurry_images = []
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

    def step_8_estimate_dataset_diversity(self, sample_size: int = 200) -> None:
        """Estimates dataset diversity via average perceptual hash distance."""
        self.log("\n🌈 [Step 8] Estimating dataset diversity (phash distance)...")

        all_paths = []
        for digit, path in self.digit_dirs.items():
            all_paths.extend([os.path.join(path, f) for f in os.listdir(path)])
        sampled_paths = random.sample(all_paths, min(sample_size, len(all_paths)))

        hashes = []
        with tqdm(total=len(sampled_paths), desc="Computing image hashes", ncols=100) as bar:
            for p in sampled_paths:
                try:
                    hashes.append(imagehash.phash(Image.open(p)))
                except Exception:
                    continue
                bar.update(1)

        distances = []
        total_comparisons = len(hashes) * (len(hashes) - 1) // 2
        with tqdm(total=total_comparisons, desc="Calculating hash distances", ncols=100) as bar:
            for i in range(len(hashes)):
                for j in range(i + 1, len(hashes)):
                    distances.append(hashes[i] - hashes[j])
                    bar.update(1)

        if distances:
            avg_dist = np.mean(distances)
            std_dist = np.std(distances)
            self.log(f"[Step 8] Average perceptual hash distance: {avg_dist:.2f} ± {std_dist:.2f}")
            if avg_dist < 5:
                self.log("⚠️ [Step 8] Low average hash distance — dataset may lack visual diversity.")
        else:
            self.log("[Step 8] Not enough valid images to estimate diversity.")

        self.log("✅ [Step 8 Complete] Dataset diversity estimation done.\n")

    def step_9_detect_partial_digits(self, margin: int = 0, samples_per_digit: int = 10) -> List[str]:
        """
        Detects images where digits touch the image frame (partial digits).

        Args:
            margin (int): Margin in pixels to consider near-edge proximity. Defaults to 0.
            samples_per_digit (int): Max number of flagged samples to display per digit.

        Returns:
            List[str]: List of paths of suspected partial digit images.
        """
        self.log("\n🔍 [Step 9] Detecting partial digits (those touching the image frame)...")

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

        self.log("🖼️ [Step 9] Building composite image grid of partial digits...")
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
        self.log(f"✅ [Step 10 Complete] Saved composite image of duplicate digits → {save_path}\n")

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

    def step_11_local_feature_consistency(self, method: str = "sobel", samples_per_digit: int = 500) -> None:
        """
        Analyzes local feature consistency across digits using either Sobel edges or ORB keypoints.
        All digit feature maps are combined into a single figure.

        Args:
            method (str): Feature extraction method, either "sobel" or "orb".
            samples_per_digit (int): Number of samples to use per digit class.
        """
        assert method in ["sobel", "orb"], "Method must be 'sobel' or 'orb'"
        self.log(f"\n🔍 [Step 11] Checking local feature consistency using '{method}' method...")

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

        # Adjust layout to ensure proper space
        plt.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.05, wspace=0.3, hspace=0.3)

        # Add colorbar on the right side
        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax, label="Feature Intensity")

        # Title and save figure
        plt.suptitle(f"Local Feature Consistency ({method.upper()})", fontsize=20)
        save_path = os.path.join(self.output_dir, f"local_feature_consistency_{method}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        self.log(f"✅ [Step 11 Complete] Combined local feature map saved: {save_path}\n")

    def step_12_digit_heatmap_grid(self) -> None:
        """Generates a heatmap for each digit (1–9) and plots them in a single composite 3×3 grid."""
        self.log("\n🔥 [Step 12] Generating per-digit heatmap grid (1–9)...")

        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle("Digit Heatmaps (1–9)", fontsize=16)

        digits = list(map(str, range(1, 10)))
        with tqdm(total=len(digits), desc="Building heatmaps", ncols=100) as bar:
            for i, digit in enumerate(digits):
                row, col = divmod(i, 3)
                ax = axes[row, col]

                path = self.digit_dirs.get(digit)
                if not path or not os.path.exists(path):
                    ax.set_title(f"Digit {digit} (missing)")
                    ax.axis("off")
                    bar.update(1)
                    continue

                image_paths = [os.path.join(path, f) for f in os.listdir(path)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not image_paths:
                    ax.set_title(f"Digit {digit} (empty)")
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

                    reference_im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
                    ax.set_title(f"Digit {digit}")
                    ax.axis("off")
                except Exception as e:
                    self.log(f"❌ [Step 12] Failed for digit {digit}: {e}")
                    ax.axis("off")

                bar.update(1)

        # Manually adjust the layout to avoid tight_layout issues
        plt.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.05, wspace=0.3, hspace=0.3)

        # Add colorbar on the right side
        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(reference_im, cax=cbar_ax, label="Normalized Intensity")

        # Save the figure
        output_path = os.path.join(self.output_dir, "digit_heatmap_grid.png")
        plt.savefig(output_path)
        plt.close()

        self.log(f"🖼️ [Step 12] Saved heatmap grid → {output_path}")
        self.log("✅ [Step 12 Complete] Digit heatmap grid generated.\n")

    def run_full_evaluation(self) -> None:
        """Runs a full evaluation pipeline and generates report artifacts."""
        self.log("🧪 Running full dataset evaluation pipeline...")
        self.log(f"Dataset path: {self.dataset_path}")
        self.log(f"Evaluation time: {datetime.datetime.now()}\n")

        self.step_1_class_distribution()
        self.step_2_check_image_dimensions()
        self.step_3_visualize_sample_grid()
        self.step_4_intensity_histograms()
        self.step_5_detect_corrupt_images()
        self.step_6_digit_centering_heatmap()
        self.step_7_detect_blurry_images()
        self.step_8_estimate_dataset_diversity()
        self.step_9_detect_partial_digits()
        self.step_10_detect_duplicate_images()
        self.step_11_local_feature_consistency(method="sobel")
        self.step_11_local_feature_consistency(method="orb")

        report_path = os.path.join(self.output_dir, "dataset_evaluation_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.report_lines))

        self.log(f"\n✅ Evaluation complete. Report saved to '{report_path}'.")

if __name__ == "__main__":
    evaluator = DigitDatasetEvaluator(dataset_path="digit_dataset")
    # evaluator.run_full_evaluation()
    evaluator.step_11_local_feature_consistency()