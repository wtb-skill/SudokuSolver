import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from typing import List, Dict
import random
import datetime
import imagehash
from PIL import Image
import concurrent.futures


class DigitDatasetEvaluator:
    def __init__(self, dataset_path: str, image_size: int = 32, output_dir: str = "evaluation_reports"):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.output_dir = output_dir
        self.digit_dirs = {str(i): os.path.join(dataset_path, str(i)) for i in range(1, 10)}
        os.makedirs(output_dir, exist_ok=True)
        self.report_lines = []

    def log(self, text: str):
        print(text)
        self.report_lines.append(text)

    def class_distribution(self) -> Dict[str, Dict[str, int]]:
        summary = {}
        self.log("📊 Class Distribution Summary:")
        for digit, path in self.digit_dirs.items():
            if not os.path.exists(path): continue
            images = os.listdir(path)
            clean = len([img for img in images if "clean" in img])
            distorted = len(images) - clean
            summary[digit] = {"total": len(images), "clean": clean, "distorted": distorted}
            self.log(f"Digit {digit}: {summary[digit]}")
        return summary

    def visualize_sample_grid(self, samples_per_digit=5):
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
        plt.savefig(os.path.join(self.output_dir, "sample_grid.png"))
        plt.close()
        self.log("🖼️ Sample image grid saved as 'sample_grid.png'")

    def intensity_histograms(self, sample_size=1000):
        clean_pixels, distorted_pixels = [], []

        for digit, path in self.digit_dirs.items():
            if not os.path.exists(path): continue
            images = os.listdir(path)
            random.shuffle(images)
            for img_name in images[:sample_size]:
                full_path = os.path.join(path, img_name)
                img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    (clean_pixels if "clean" in img_name else distorted_pixels).extend(img.flatten())

        plt.hist(clean_pixels, bins=50, alpha=0.6, label="Clean", color="green")
        plt.hist(distorted_pixels, bins=50, alpha=0.6, label="Distorted", color="red")
        plt.title("Pixel Intensity Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "pixel_histogram.png"))
        plt.close()
        self.log("📉 Pixel intensity histogram saved as 'pixel_histogram.png'")

    def detect_corrupt_images(self) -> List[str]:
        corrupt = []
        for digit, path in self.digit_dirs.items():
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None or img.max() - img.min() < 10:
                    corrupt.append(img_path)
        if corrupt:
            self.log(f"⚠️ Found {len(corrupt)} potentially corrupt images:")
            for c in corrupt[:10]:
                self.log(f"    - {c}")
        else:
            self.log("✅ No corrupt images detected.")
        return corrupt

    def digit_centering_heatmap(self, samples_per_digit=100):
        accumulator = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        total = 0

        for digit, path in self.digit_dirs.items():
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
            plt.savefig(os.path.join(self.output_dir, "centering_heatmap.png"))
            plt.close()
            self.log("🔥 Digit centering heatmap saved as 'centering_heatmap.png'")
        else:
            self.log("⚠️ No images found for centering heatmap.")

    def detect_partial_digits(self, margin: int = 3, samples_per_digit: int = 10) -> List[str]:
        """
        Detects images where the digit touches the edge of the image frame.
        Also saves a single composite PNG with up to `samples_per_digit` flagged images per digit.

        Parameters:
            margin (int): Pixel distance from the edge to consider as "touching".
            samples_per_digit (int): Number of sample images per digit to visualize.

        Returns:
            List[str]: Paths of suspected partial digit images.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, "partial_digits_grid.png")
        partials = []
        partials_by_digit = defaultdict(list)

        # Detect partials and organize by digit
        for digit, path in self.digit_dirs.items():
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

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

        # Create single composite PNG
        digits = sorted(partials_by_digit.keys())
        if not digits:
            self.log("✅ No partial digits detected based on edge proximity.")
            return []

        grid_rows = len(digits)
        grid_cols = samples_per_digit
        grid_img = np.ones((grid_rows * self.image_size, grid_cols * self.image_size),
                           dtype=np.uint8) * 255  # white background

        for row_idx, digit in enumerate(digits):
            selected = partials_by_digit[digit][:samples_per_digit]
            for col_idx, img_path in enumerate(selected):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img_resized = cv2.resize(img, (self.image_size, self.image_size))
                y1 = row_idx * self.image_size
                y2 = y1 + self.image_size
                x1 = col_idx * self.image_size
                x2 = x1 + self.image_size
                grid_img[y1:y2, x1:x2] = img_resized

        cv2.imwrite(save_path, grid_img)
        self.log(f"🖼️ Saved composite image of partial digits → {save_path}")

        self.log(f"🟠 Found {len(partials)} potentially partial digit images (touching image edge):")
        for p in partials[:10]:
            self.log(f"    - {p}")

        return partials

    def compute_hash(self, img_path, hash_func=imagehash.phash):
        try:
            img = Image.open(img_path).convert("L").resize((32, 32))
            img_hash = hash_func(img)
            return (img_hash, img_path)
        except Exception as e:
            return None

    def detect_duplicate_images(self, hash_func=imagehash.phash, threshold=1,
                                max_groups=10, samples_per_group=5):
        """
        Detects and visualizes duplicate or near-duplicate images **within the same digit class**.
        """
        self.log("🔍 Detecting duplicate or near-duplicate images within each digit class...")
        hash_to_files = defaultdict(list)
        all_images = []

        # Iterate through each digit class and gather image paths
        for digit, path in self.digit_dirs.items():
            self.log(f"🔄 Processing digit {digit}...")
            image_paths = [os.path.join(path, img_name) for img_name in os.listdir(path)]

            # Parallelize the image hash computation
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(self.compute_hash, image_paths, [hash_func] * len(image_paths)))

            # Filter out None results from failed image processing
            results = [res for res in results if res is not None]

            for img_hash, img_path in results:
                all_images.append((img_hash, img_path, digit))  # Add digit class to the tuple

        # Progress log after image hash computation
        self.log("🔍 Image hash computation complete, now detecting duplicates...")

        # Find duplicates **within the same digit class** by comparing hashes
        duplicates = defaultdict(list)
        used = set()

        # Parallel comparison of images for duplicates
        total_comparisons = len(all_images)
        comparisons_done = 0

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i, (h1, path1, digit1) in enumerate(all_images):
                if path1 in used:
                    continue
                group = [path1]
                # Compare hashes in parallel
                futures = []
                for j in range(i + 1, len(all_images)):
                    h2, path2, digit2 = all_images[j]
                    if digit1 == digit2 and h1 - h2 <= threshold and path2 not in used:  # Compare only within the same digit class
                        futures.append(executor.submit(lambda x: group.append(x), path2))
                concurrent.futures.wait(futures)
                if len(group) > 1:
                    duplicates[str(h1)].extend(group)
                    used.update(group)

                comparisons_done += 1
                if comparisons_done % 500 == 0:  # Log progress every 500 comparisons
                    self.log(f"🔄 Progress: {comparisons_done}/{total_comparisons} comparisons done.")

        # Log summary after comparison
        if duplicates:
            total_dupes = sum(len(v) for v in duplicates.values())
            self.log(f"🟠 Found {total_dupes} images in {len(duplicates)} duplicate groups.")
            for h, paths in list(duplicates.items())[:3]:
                self.log(f"  🔗 Group {h[:8]} → {len(paths)} images")
                for p in paths[:3]:
                    self.log(f"     - {p}")
        else:
            self.log("✅ No duplicate images found.")
            return {}

        # Build sample visualization
        groups = list(duplicates.values())[:max_groups]
        rows = len(groups)
        cols = samples_per_group
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

        # Save visual
        output_path = os.path.join(self.output_dir, "duplicate_samples.png")
        cv2.imwrite(output_path, grid_img)
        self.log(f"🖼️ Sample duplicate groups visual saved → {output_path}")

        return duplicates

    def run_full_evaluation(self):
        self.log("🧪 Running full dataset evaluation pipeline...")
        self.log(f"Dataset path: {self.dataset_path}")
        self.log(f"Evaluation time: {datetime.datetime.now()}\n")

        self.class_distribution()
        self.visualize_sample_grid()
        self.intensity_histograms()
        self.detect_corrupt_images()
        self.digit_centering_heatmap()

        report_path = os.path.join(self.output_dir, "dataset_evaluation_report.txt")
        with open(report_path, "w") as f:
            f.write("\n".join(self.report_lines))

        self.log(f"\n✅ Evaluation complete. Report saved to '{report_path}'.")

if __name__ == "__main__":
    evaluator = DigitDatasetEvaluator(dataset_path="digit_dataset")
    # evaluator.run_full_evaluation()
    # evaluator.detect_partial_digits(margin=0)
    evaluator.detect_duplicate_images()