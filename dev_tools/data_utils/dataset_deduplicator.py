# dev_tools/data_utils/dataset_deduplicator.py
import os
import imagehash
import cv2
import networkx as nx
from tqdm import tqdm
from collections import defaultdict


class DuplicateImageReducer:
    def __init__(self, dataset_dir, hash_func=imagehash.phash, threshold=0):
        self.dataset_dir = dataset_dir
        self.hash_func = hash_func
        self.threshold = threshold
        self.removed_count = defaultdict(int)
        self.graph = nx.Graph()

    def _compute_hash(self, image_path):
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, (64, 64))
                return self.hash_func(imagehash.Image.fromarray(image)), image_path
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
        return None

    def load_duplicates(self):
        """
        Loads images, computes hashes, and builds a graph of connected components.
        """
        for digit in sorted(os.listdir(self.dataset_dir)):
            digit_dir = os.path.join(self.dataset_dir, digit)
            if not os.path.isdir(digit_dir):
                continue

            image_paths = [os.path.join(digit_dir, img) for img in os.listdir(digit_dir)]

            with tqdm(total=len(image_paths), desc=f"Processing digit {digit}", ncols=100) as pbar:
                hashes = []
                for image_path in image_paths:
                    result = self._compute_hash(image_path)
                    if result:
                        img_hash, path = result
                        hashes.append((img_hash, path))
                    pbar.update(1)

                # Build graph from hashes and set 'digit' attribute
                for i in range(len(hashes)):
                    for j in range(i + 1, len(hashes)):
                        h1, path1 = hashes[i]
                        h2, path2 = hashes[j]
                        if h1 - h2 <= self.threshold:
                            # Add nodes and set 'digit' attribute
                            if path1 not in self.graph:
                                self.graph.add_node(path1, digit=digit)
                            if path2 not in self.graph:
                                self.graph.add_node(path2, digit=digit)
                            self.graph.add_edge(path1, path2, digit=digit)

    def reduce_duplicates(self):
        """
        Reduces duplicate images, keeping one per cluster.
        """
        components = list(nx.connected_components(self.graph))
        processed = set()
        files_to_delete = []

        with tqdm(total=len(components), desc="Reducing duplicates", ncols=100) as pbar:
            for cluster in components:
                cluster = list(cluster)
                representative = cluster[0]  # Keep the first appearing image
                for img in cluster[1:]:
                    if img not in processed:
                        files_to_delete.append(img)
                        digit = self.graph.nodes[img]['digit']
                        self.removed_count[digit] += 1
                    processed.add(img)
                pbar.update(1)

        # Delete files after processing
        for file in files_to_delete:
            os.remove(file)

    def generate_report(self):
        """
        Generates a visually appealing report of the removed images per digit.
        This will produce a nicely formatted .txt file.
        """
        # Create the report content
        report_lines = ["Duplicate Reduction Report\n"]
        report_lines.append("-" * 30 + "\n")
        report_lines.append(f"{'Digit':<10}{'Removed Images':>20}\n")
        report_lines.append("-" * 30 + "\n")

        for digit in sorted(self.removed_count.keys()):
            report_lines.append(f"{digit:<10}{self.removed_count[digit]:>20}\n")

        report_lines.append("-" * 30 + "\n")

        # Save the report as a text file in the script's directory
        report_path = os.path.join(os.getcwd(), 'duplicate_reduction_report.txt')
        with open(report_path, 'w') as f:
            f.writelines(report_lines)

        # Print the result to the console as well
        print("✅ Report saved to:", report_path)
        for line in report_lines:
            print(line, end="")  # Print each line of the report to console


if __name__ == '__main__':
    dataset_dir = './test_dataset'  # Path to the dataset folder (input and output are the same)
    reducer = DuplicateImageReducer(dataset_dir)
    reducer.load_duplicates()
    reducer.reduce_duplicates()
    reducer.generate_report()

