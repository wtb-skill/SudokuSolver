# generate_model/sudokunet_trainer.py
from sudokunet import SudokuNet
from model_evaluator import ModelEvaluator
from keras.api.optimizers import Adam
from keras.api.utils import to_categorical
from keras.api.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from pathlib import Path

class SudokunetTrainer:
    def __init__(self, dataset_path: str, model_output_path: str, image_size: int = 32,
                 init_lr: float = 1e-3, epochs: int = 10, batch_size: int = 128):
        """
        Initializes the SudokuDigitTrainer.

        :param dataset_path: Path to the dataset directory.
        :param model_output_path: Path to save the trained model.
        :param image_size: Size (height/width) to resize input images.
        :param init_lr: Initial learning rate for training.
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        """
        self.dataset_path = dataset_path
        self.model_output_path = model_output_path
        self.image_size = image_size
        self.init_lr = init_lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = None
        self.trainData = None
        self.trainLabels = None
        self.testData = None
        self.testLabels = None

    def load_dataset(self):
        """
        Loads and preprocesses the dataset from self.dataset_path.
        """
        print("[INFO] Accessing the custom dataset...")

        data, labels = [], []

        for digit in range(1, 10):
            digit_path = os.path.join(self.dataset_path, str(digit))
            if not os.path.exists(digit_path):
                print(f"[WARNING] Skipping {digit_path} (Folder not found)")
                continue

            print(f"[INFO] Processing digit '{digit}'...")
            digit_images = os.listdir(digit_path)
            print(f"   - Found {len(digit_images)} images for digit {digit}")

            for i, image_name in enumerate(digit_images):
                img_path = os.path.join(digit_path, image_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (self.image_size, self.image_size))
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=-1)

                data.append(img)
                labels.append(digit - 1)

                if i % 1000 == 0 and i > 0:
                    print(f"     - Processed {i} images for digit {digit}...")

        data = np.array(data)
        labels = to_categorical(np.array(labels), num_classes=9)
        print(f"[INFO] Total dataset size: {len(data)} images")

        indices = np.arange(len(data))
        np.random.shuffle(indices)
        data, labels = data[indices], labels[indices]

        split = int(0.8 * len(data))
        self.trainData, self.testData = data[:split], data[split:]
        self.trainLabels, self.testLabels = labels[:split], labels[split:]
        print(f"[INFO] Dataset split into {len(self.trainData)} training and {len(self.testData)} testing samples.")

    def load_mnist_dataset(self):
        """
        Loads and preprocesses the MNIST dataset (digits 1 through 9 only),
        structured to behave like the custom dataset loading logic.
        """
        print("[INFO] Loading MNIST dataset...")

        (data, labels), _ = mnist.load_data()

        # Filter out digit '0' to match your custom dataset (1–9 only)
        print("[INFO] Filtering digits 1–9 only...")
        mask = labels != 0
        data = data[mask]
        labels = labels[mask] - 1  # Convert labels from 1–9 to 0–8

        # Normalize and reshape
        data = data.astype("float32") / 255.0
        data = np.expand_dims(data, axis=-1)

        # Resize if needed
        if self.image_size != 28:
            print(f"[INFO] Resizing images to {self.image_size}x{self.image_size}...")
            data_resized = np.array([cv2.resize(img, (self.image_size, self.image_size)) for img in data])
            data = np.expand_dims(data_resized, axis=-1)

        # One-hot encode labels
        labels = to_categorical(labels, num_classes=9)

        print(f"[INFO] Total dataset size: {len(data)} images")

        # Shuffle and split just like in load_dataset()
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        data, labels = data[indices], labels[indices]

        split = int(0.8 * len(data))
        self.trainData, self.testData = data[:split], data[split:]
        self.trainLabels, self.testLabels = labels[:split], labels[split:]
        print(f"[INFO] Dataset split into {len(self.trainData)} training and {len(self.testData)} testing samples.")

    def compile_model(self):
        """
        Compiles the SudokuNet model.
        """
        print("[INFO] Compiling model...")
        opt = Adam(learning_rate=self.init_lr)
        self.model = SudokuNet.build(
            width=self.image_size, height=self.image_size, depth=1, classes=9
        )
        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    def save_model(self):
        """
        Saves the trained model to disk.
        """
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        print(f"[INFO] Saving model to {self.model_output_path}...")
        self.model.save(self.model_output_path)

    @staticmethod
    def plot_training_history(history, save_path=None):
        """
        Plots and saves the training history (accuracy and loss).

        :param history: The history object returned by model.fit().
        :param save_path: Path to save the plot (optional).
        """
        plt.figure(figsize=(12, 5))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"[INFO] Training history plot saved to {save_path}")
        plt.close()

    def train(self):
        """
        Trains the model using loaded dataset and saves the training history plot.
        """
        if self.model is None:
            raise RuntimeError("Model has not been compiled. Call compile_model() first.")

        print("[INFO] Training network...")
        history = self.model.fit(
            self.trainData, self.trainLabels,
            validation_data=(self.testData, self.testLabels),
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1
        )

        # Save the training history plot
        self.plot_training_history(history, save_path="training_history.jpg")

    def run(self):
        """
        Runs the complete training pipeline.
        """
        self.load_dataset() # load_dataset or load_mnist_dataset
        self.compile_model()
        self.train()
        self.save_model()

        # Use ModelEvaluator for evaluation during training
        evaluator = ModelEvaluator(
            model_path=self.model_output_path,
            training_tests=True,
            test_data=self.testData,
            test_labels=self.testLabels,
            class_names=[str(i) for i in range(1, 10)]
        )
        evaluator.run()


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_path = str(project_root / "dev_tools" / "data_utils" / "digit_dataset")
    model_output_path = str(project_root / "models" / "v2.keras")

    trainer = SudokunetTrainer(
        dataset_path=dataset_path,
        model_output_path=model_output_path,
        image_size=32,
        init_lr=1e-3,
        epochs=10,
        batch_size=128
    )
    trainer.run()