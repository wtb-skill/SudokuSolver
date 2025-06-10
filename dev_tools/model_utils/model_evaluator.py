# generate_model/model_evaluator.py
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from keras.api.models import load_model
from keras.api.utils import to_categorical
from sklearn.metrics import (
    classification_report as skl_classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from keras.api.preprocessing.image import img_to_array
from sklearn.preprocessing import label_binarize
from typing import List, Optional, Tuple


class ModelEvaluator:
    """
    Evaluates a trained Keras image classification model using various metrics,
    including accuracy, confusion matrix, ROC curves, and misclassification analysis.
    """

    def __init__(
        self,
        model_path: str,
        training_tests: bool = True,
        test_data: Optional[np.ndarray] = None,
        test_labels: Optional[np.ndarray] = None,
        dataset_path: Optional[str] = None,
        image_size: Tuple[int, int] = (32, 32),
        class_names: Optional[List[str]] = None,
    ):
        """
        Initializes the evaluator, loads model and test data.
        """
        print("[INFO] Loading model...")
        self.model = self._load_model_from_directory(model_path)

        if training_tests:
            self.test_data = test_data
            self.test_labels = test_labels
            self.class_names = (
                class_names if class_names else [str(i) for i in range(1, 10)]
            )
        else:
            print(f"[INFO] Loading external test data from: {dataset_path}")
            self.test_data, self.test_labels, self.class_names = self.load_test_data(
                dataset_path, image_size
            )

        self.true_classes = self.test_labels.argmax(axis=1)
        self.predictions = self.model.predict(self.test_data)
        self.predicted_classes = self.predictions.argmax(axis=1)

    @staticmethod
    def _load_model_from_directory(model_path: str):
        """
        Loads a model from the given path, handling both file-based and directory-based models.

        Parameters:
            model_path (str): Path to the model file or directory.

        Returns:
            keras.Model: The loaded model.
        """
        if os.path.isfile(model_path):
            # Directly load model file (.keras or .h5)
            print(f"[INFO] Loading model from file: {model_path}")
            return load_model(model_path)
        elif os.path.isdir(model_path):
            # Find the most recent model file in the directory
            valid_extensions = [".keras", ".h5"]
            model_files = [
                f
                for f in os.listdir(model_path)
                if any(f.endswith(ext) for ext in valid_extensions)
            ]

            if len(model_files) == 0:
                raise ValueError(
                    "No valid model files found in the specified directory."
                )

            # Get the most recent model file
            latest_model_file = max(
                model_files, key=lambda f: os.path.getmtime(os.path.join(model_path, f))
            )
            model_file_path = os.path.join(model_path, latest_model_file)

            print(f"[INFO] Loading model from directory: {model_file_path}")
            return load_model(model_file_path)
        else:
            raise ValueError(
                f"Invalid model path: {model_path}. Must be a file or directory."
            )

    @staticmethod
    def ensure_output_folder(path: str) -> None:
        """Ensures the directory for saving results exists."""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def load_test_data(
        dataset_path: str, image_size: Tuple[int, int] = (32, 32)
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Loads test images and labels from a directory structure.
        """
        print(f"[INFO] Loading test data from {dataset_path}...")
        test_data = []
        test_labels = []
        class_names = sorted(os.listdir(dataset_path))

        for label, digit in enumerate(class_names):
            digit_folder = os.path.join(dataset_path, digit)
            if not os.path.isdir(digit_folder):
                continue

            print(f"[INFO] Processing digit '{digit}' ({label})...")
            for file_name in os.listdir(digit_folder):
                file_path = os.path.join(digit_folder, file_name)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                img = cv2.resize(img, image_size)
                img = img.astype("float32") / 255.0
                img = img_to_array(img)
                test_data.append(img)
                test_labels.append(label)

        test_data = np.array(test_data)
        test_labels = to_categorical(
            np.array(test_labels), num_classes=len(class_names)
        )

        print(f"[INFO] Loaded {len(test_data)} images from {len(class_names)} classes.")
        return test_data, test_labels, class_names

    def show_test_distribution(self, save_path: Optional[str] = None) -> None:
        """Plots and optionally saves class distribution of test set."""
        counts = np.sum(self.test_labels, axis=0)
        plt.figure()
        plt.bar(self.class_names, counts)
        plt.title("Test Set Distribution per Class")
        plt.xlabel("Digit")
        plt.ylabel("Count")
        if save_path:
            plt.savefig(save_path)
            print(f"[INFO] Test distribution saved to {save_path}")
        plt.close()

    def evaluate_model(self, save_path: Optional[str] = None) -> Tuple[float, float]:
        """Evaluates model on the test dataset and optionally saves a metrics table plot.

        Args:
            save_path (Optional[str]): Path to save the evaluation table plot.

        Returns:
            Tuple[float, float]: The loss and accuracy of the model on test data.
        """
        loss, accuracy = self.model.evaluate(
            self.test_data, self.test_labels, verbose=1
        )
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        if save_path:
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.axis("tight")
            ax.axis("off")
            table_data = [
                ["Metric", "Value"],
                ["Loss", f"{loss:.4f}"],
                ["Accuracy", f"{accuracy * 100:.2f}%"],
            ]
            table = ax.table(cellText=table_data, colLabels=None, loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"[INFO] Evaluation table saved to {save_path}")
            plt.close()

        return loss, accuracy

    def per_class_accuracy(self, save_path: Optional[str] = None) -> np.ndarray:
        """Calculates and displays per-class accuracy with optional bar plot.

        Args:
            save_path (Optional[str]): Path to save the per-class accuracy bar chart.

        Returns:
            np.ndarray: Array of per-class accuracies.
        """
        cm = confusion_matrix(self.true_classes, self.predicted_classes)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)

        for cls, acc in zip(self.class_names, per_class_acc):
            print(f"Accuracy for class {cls}: {acc:.2%}")

        if save_path:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=self.class_names, y=per_class_acc, palette="viridis")

            plt.title("Per-Class Accuracy", fontsize=14, weight="bold", pad=20)
            plt.xlabel("Class", fontsize=12)
            plt.ylabel("Accuracy", fontsize=12)
            plt.ylim(0, 1.05)

            for i, acc in enumerate(per_class_acc):
                plt.text(i, acc + 0.02, f"{acc:.2%}", ha="center", fontsize=10)

            plt.tight_layout()
            plt.savefig(save_path)
            print(f"[INFO] Per-class accuracy plot saved to {save_path}")
            plt.close()

        return per_class_acc

    def classification_report(self, save_path: Optional[str] = None) -> str:
        """Generates a classification report and optionally saves it as an image."""
        report = skl_classification_report(
            self.true_classes, self.predicted_classes, target_names=self.class_names
        )
        print("\nClassification Report:\n", report)

        if save_path:
            plt.figure(figsize=(10, 6))
            plt.text(0.01, 1, report, {"fontsize": 12}, fontfamily="monospace")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"[INFO] Classification report saved to {save_path}")
            plt.close()

        return report

    def roc_curve(self, save_path: Optional[str] = None) -> None:
        """Plots ROC curve per class and optionally saves the plot."""
        y_true_bin = label_binarize(
            self.true_classes, classes=np.arange(len(self.class_names))
        )
        y_pred_bin = self.predictions
        plt.figure(figsize=(10, 6))

        for i in range(len(self.class_names)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr, label=f"Class {self.class_names[i]} (AUC = {roc_auc:.2f})"
            )

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"[INFO] ROC curve saved to {save_path}")
        plt.close()

    def confusion_matrix(self, save_path: Optional[str] = None) -> np.ndarray:
        """Plots and optionally saves confusion matrix."""
        cm = confusion_matrix(self.true_classes, self.predicted_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        if save_path:
            plt.savefig(save_path)
            print(f"[INFO] Confusion matrix saved to {save_path}")
        plt.close()
        return cm

    def analyze_misclassifications(
        self, save_path: Optional[str] = None, top_n: int = 10
    ) -> List[Tuple[Tuple[str, str], int]]:
        """Analyzes and plots top misclassified label pairs."""
        cm = confusion_matrix(self.true_classes, self.predicted_classes)
        np.fill_diagonal(cm, 0)
        misclassifications = [
            ((self.class_names[i], self.class_names[j]), cm[i, j])
            for i in range(len(self.class_names))
            for j in range(len(self.class_names))
            if cm[i, j] > 0
        ]
        misclassifications.sort(key=lambda x: x[1], reverse=True)
        top_misclassifications = misclassifications[:top_n]

        print("[INFO] Top misclassifications:")
        for (true_label, pred_label), count in top_misclassifications:
            print(f"{true_label} → {pred_label}: {count} times")

        labels = [f"{true} → {pred}" for (true, pred), _ in top_misclassifications]
        counts = [count for _, count in top_misclassifications]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=counts, y=labels, palette="Reds_r")
        plt.xlabel("Count")
        plt.ylabel("Misclassification (True → Predicted)")
        plt.title(f"Top {top_n} Most Common Misclassifications")

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"[INFO] Top misclassifications chart saved to {save_path}")
        plt.close()

        return top_misclassifications

    def analyze_confused_classes(
        self, top_n: int = 5, save_path: Optional[str] = None
    ) -> List[Tuple[str, str, int]]:
        """Finds and plots the most confused class pairs."""
        cm = confusion_matrix(self.true_classes, self.predicted_classes)
        np.fill_diagonal(cm, 0)

        misclassifications = [
            (self.class_names[i], self.class_names[j], cm[i, j])
            for i in range(len(self.class_names))
            for j in range(len(self.class_names))
            if cm[i, j] > 0
        ]
        misclassifications.sort(key=lambda x: x[2], reverse=True)
        most_confused = misclassifications[:top_n]

        print("[INFO] Most confused class pairs:")
        for true_label, pred_label, count in most_confused:
            print(f"{true_label} → {pred_label}: {count} times")

        labels = [f"{true} → {pred}" for true, pred, _ in most_confused]
        counts = [count for _, _, count in most_confused]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=counts, y=labels, palette="magma")
        plt.xlabel("Count")
        plt.ylabel("Confused Class Pair (True → Predicted)")
        plt.title(f"Top {top_n} Most Confused Class Pairs")

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"[INFO] Most confused class pairs plot saved to {save_path}")
        plt.close()

        return most_confused

    def view_misclassified_images(
        self, num_images: int = 10, save_path: Optional[str] = None
    ) -> None:
        """
        Displays and optionally saves a grid of randomly selected misclassified images.

        Args:
            num_images (int): Number of misclassified images to display.
            save_path (Optional[str]): If provided, saves the plot to this path.
        """
        misclassified_idxs = np.where(self.predicted_classes != self.true_classes)[0]
        if len(misclassified_idxs) == 0:
            print("[INFO] No misclassified samples to display.")
            return

        selected_idxs = np.random.choice(
            misclassified_idxs, min(num_images, len(misclassified_idxs)), replace=False
        )

        max_per_row = 10
        num_cols = min(max_per_row, len(selected_idxs))
        num_rows = int(np.ceil(len(selected_idxs) / max_per_row))

        plt.figure(figsize=(num_cols * 2, num_rows * 2.5))

        for i, idx in enumerate(selected_idxs):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(self.test_data[idx].squeeze(), cmap="gray")
            plt.title(
                f"Pred: {self.class_names[self.predicted_classes[idx]]}\nTrue: {self.class_names[self.true_classes[idx]]}",
                fontsize=8,
            )
            plt.axis("off")

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"[INFO] Misclassified images saved to {save_path}")
        plt.close()

    def run(
        self,
        num_misclassified_images: int = 50,
        results_output_path: str = "evaluation_results/evaluation_results.json",
    ) -> None:
        """
        Runs the full evaluation pipeline, saving results to the output directory.
        """
        output_dir = os.path.dirname(results_output_path)
        self.ensure_output_folder(output_dir)

        self.evaluate_model(save_path=os.path.join(output_dir, "model_evaluation.jpg"))
        self.per_class_accuracy(
            save_path=os.path.join(output_dir, "per_class_accuracy.jpg")
        )
        self.classification_report(
            save_path=os.path.join(output_dir, "classification_report.jpg")
        )
        self.roc_curve(save_path=os.path.join(output_dir, "roc_curve.jpg"))
        self.confusion_matrix(
            save_path=os.path.join(output_dir, "confusion_matrix.jpg")
        )
        self.show_test_distribution(
            save_path=os.path.join(output_dir, "test_distribution.jpg")
        )
        self.analyze_misclassifications(
            save_path=os.path.join(output_dir, "top_misclassifications.jpg")
        )
        self.analyze_confused_classes(
            save_path=os.path.join(output_dir, "analyze_confused_classes.jpg")
        )
        self.view_misclassified_images(
            num_misclassified_images,
            save_path=os.path.join(output_dir, "misclassified_samples.jpg"),
        )


if __name__ == "__main__":
    MODEL_PATH = "../../models/currently_used"
    TEST_DATASET_PATH = "../data_utils/test_dataset"

    evaluator = ModelEvaluator(
        model_path=MODEL_PATH, training_tests=False, dataset_path=TEST_DATASET_PATH
    )
    evaluator.run()
