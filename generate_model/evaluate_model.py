import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from keras.api.models import load_model
from keras.api.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from keras.api.preprocessing.image import img_to_array


class ModelEvaluator:
    def __init__(self, model_path, test_data, test_labels, class_names=None):
        """
        Initializes the ModelEvaluator class.

        :param model_path: Path to the trained model.
        :param test_data: Numpy array of test images.
        :param test_labels: Numpy array of one-hot encoded test labels.
        :param class_names: List of class labels (e.g., ['0', '1', ..., '9']).
        """
        print("[INFO] Loading model...")
        self.model = load_model(model_path)
        self.test_data = test_data
        self.test_labels = test_labels
        self.class_names = class_names if class_names else [str(i) for i in range(10)]

        # Convert one-hot labels to integers
        self.true_classes = self.test_labels.argmax(axis=1)
        self.predictions = self.model.predict(self.test_data)
        self.predicted_classes = self.predictions.argmax(axis=1)

    def evaluate_model(self):
        """Evaluates the model and prints accuracy and loss."""
        loss, accuracy = self.model.evaluate(self.test_data, self.test_labels, verbose=1)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        return loss, accuracy

    def classification_report(self):
        """Prints the classification report with precision, recall, and F1-score."""
        report = classification_report(self.true_classes, self.predicted_classes, target_names=self.class_names)
        print("\nClassification Report:\n", report)
        return report

    def confusion_matrix(self):
        """Displays a confusion matrix to visualize misclassifications."""
        cm = confusion_matrix(self.true_classes, self.predicted_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()
        return cm

    def analyze_misclassifications(self):
        """Finds and prints the most common misclassifications."""
        cm = confusion_matrix(self.true_classes, self.predicted_classes)
        np.fill_diagonal(cm, 0)  # Ignore correct classifications
        misclassified_pairs = np.unravel_index(np.argmax(cm), cm.shape)
        most_misclassified_from = self.class_names[misclassified_pairs[0]]
        most_misclassified_to = self.class_names[misclassified_pairs[1]]
        print(f"Most common misclassification: {most_misclassified_from} -> {most_misclassified_to}")
        return most_misclassified_from, most_misclassified_to

    def view_misclassified_images(self, num_images=5):
        """Displays a few misclassified images with predicted and true labels."""
        misclassified_idxs = np.where(self.predicted_classes != self.true_classes)[0]
        selected_idxs = np.random.choice(misclassified_idxs, min(num_images, len(misclassified_idxs)), replace=False)

        plt.figure(figsize=(10, 5))
        for i, idx in enumerate(selected_idxs):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(self.test_data[idx].squeeze(), cmap="gray")
            plt.title(
                f"Pred: {self.class_names[self.predicted_classes[idx]]}\nTrue: {self.class_names[self.true_classes[idx]]}")
            plt.axis("off")
        plt.show()

    def save_evaluation_results(self, output_path="evaluation_results.json"):
        """Saves evaluation results to a JSON file."""
        metrics = {
            "classification_report": classification_report(self.true_classes, self.predicted_classes,
                                                           target_names=self.class_names, output_dict=True)
        }
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"[INFO] Evaluation results saved to {output_path}")


def load_test_data(dataset_path, image_size=(32, 32)):
    """
    Loads the test dataset from the given folder structure.

    :param dataset_path: Path to the dataset folder (containing digit folders).
    :param image_size: Target size of images (default 32x32).
    :return: Tuple (testData, testLabels, class_names)
    """
    print(f"[INFO] Loading test data from {dataset_path}...")
    test_data = []
    test_labels = []
    class_names = sorted(os.listdir(dataset_path))  # Ensure classes are sorted in order

    for label, digit in enumerate(class_names):
        digit_folder = os.path.join(dataset_path, digit)
        if not os.path.isdir(digit_folder):
            continue  # Skip non-folder items

        print(f"[INFO] Processing digit '{digit}' ({label})...")
        for file_name in os.listdir(digit_folder):
            file_path = os.path.join(digit_folder, file_name)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # Skip unreadable images

            img = cv2.resize(img, image_size)  # Resize to match model input
            img = img.astype("float32") / 255.0  # Normalize
            img = img_to_array(img)

            test_data.append(img)
            test_labels.append(label)

    # Convert to numpy arrays
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    test_labels = to_categorical(test_labels, num_classes=len(class_names))  # One-hot encode labels

    print(f"[INFO] Loaded {len(test_data)} images from {len(class_names)} classes.")
    return test_data, test_labels, class_names


if __name__ == '__main__':
    # Define paths
    MODEL_PATH = "../models/sudoku_digit_recognizer.keras"
    TEST_DATASET_PATH = "test_dataset"

    # Load dataset
    testData, testLabels, classNames = load_test_data(TEST_DATASET_PATH)

    # Evaluate model
    evaluator = ModelEvaluator(MODEL_PATH, testData, testLabels, class_names=classNames)
    evaluator.evaluate_model()
    evaluator.classification_report()
    evaluator.confusion_matrix()
    evaluator.analyze_misclassifications()
    evaluator.view_misclassified_images()
    evaluator.save_evaluation_results("evaluation_results.json")
