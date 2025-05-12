# dev_tools/data_utils/dataset_deduplicator.py
import os
import imagehash
import cv2
import networkx as nx
import gc
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

            with tqdm(total=len(image_paths), desc=f"Deduplicator: Processing digit {digit}", ncols=100) as pbar:
                hashes = []
                for image_path in image_paths:
                    result = self._compute_hash(image_path)
                    if result:
                        img_hash, path = result
                        hashes.append((img_hash, path))
                    pbar.update(1)

            # Building graph with a progress bar
            total_comparisons = len(hashes) * (len(hashes) - 1) // 2
            with tqdm(total=total_comparisons, desc=f"Deduplicator: Building graph for digit {digit}", ncols=100) as pbar_graph:
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
                        pbar_graph.update(1)

            # After processing each digit, clear memory for the current hashes
            del hashes
            gc.collect()

    def reduce_duplicates(self):
        components = list(nx.connected_components(self.graph))
        processed = set()
        files_to_delete = []

        with tqdm(total=len(components), desc="Deduplicator: Reducing duplicates", ncols=100) as pbar:
            for cluster in components:
                cluster = list(cluster)
                representative = cluster[0]
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

        # Clear memory after reducing duplicates
        del components, files_to_delete
        gc.collect()

    def generate_report(self):
        report_lines = ["Duplicate Reduction Report\n", "-" * 30 + "\n", f"{'Digit':<10}{'Removed Images':>20}\n", "-" * 30 + "\n"]

        for digit in sorted(self.removed_count.keys()):
            report_lines.append(f"{digit:<10}{self.removed_count[digit]:>20}\n")

        report_lines.append("-" * 30 + "\n")
        report_path = os.path.join(os.getcwd(), 'duplicate_reduction_report.txt')
        with open(report_path, 'w') as f:
            f.writelines(report_lines)
        print("✅ Report saved to:", report_path)
        for line in report_lines:
            print(line, end="")

    def run(self):
        self.load_duplicates()
        self.reduce_duplicates()
        self.generate_report()

        # Clear memory
        del self.graph
        del self.removed_count
        gc.collect()

if __name__ == '__main__':
    dataset_dir = './digit_dataset_v2b'
    reducer = DuplicateImageReducer(dataset_dir)
    reducer.run()
