# train_sudokunet.py
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations
from Sudokunet import SudokuNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import argparse
import numpy as np
from tensorflow.keras.models import Model


def parse_arguments() -> dict:
    """
    Parses command-line arguments.

    Returns:
        dict: A dictionary containing the parsed arguments.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Path to output model after training")
    return vars(ap.parse_args())


def preprocess_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads and preprocesses the MNIST dataset.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        - trainData: Training images, reshaped and normalized.
        - trainLabels: One-hot encoded training labels.
        - testData: Test images, reshaped and normalized.
        - testLabels: One-hot encoded test labels.
    """
    print("[INFO] Accessing MNIST dataset...")
    (trainData, trainLabels), (testData, testLabels) = mnist.load_data()

    # Add a channel dimension (grayscale)
    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))

    # Normalize data to the range [0, 1]
    trainData = trainData.astype("float32") / 255.0
    testData = testData.astype("float32") / 255.0

    # Convert labels to one-hot encoded vectors
    trainLabels = to_categorical(trainLabels, num_classes=10)
    testLabels = to_categorical(testLabels, num_classes=10)
    # le = LabelBinarizer()
    # trainLabels = le.fit_transform(trainLabels)
    # testLabels = le.transform(testLabels)

    return trainData, trainLabels, testData, testLabels


def compile_model(init_lr: float, width: int, height: int, depth: int, classes: int) -> Model:
    """
    Builds and compiles the SudokuNet CNN model.

    Parameters:
        init_lr (float): Initial learning rate.
        width (int): Width of input images.
        height (int): Height of input images.
        depth (int): Number of channels in input images.
        classes (int): Number of output classes.

    Returns:
        Model: A compiled Keras model.
    """
    print("[INFO] Compiling model...")
    opt = Adam(learning_rate=init_lr)
    model = SudokuNet.build(width=width, height=height, depth=depth, classes=classes)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


def train_model(model: Model, trainData: np.ndarray, trainLabels: np.ndarray,
                testData: np.ndarray, testLabels: np.ndarray, batch_size: int, epochs: int):
    """
    Trains the SudokuNet CNN model.

    Parameters:
        model (Model): The compiled Keras model.
        trainData (np.ndarray): Training image data.
        trainLabels (np.ndarray): One-hot encoded training labels.
        testData (np.ndarray): Test image data.
        testLabels (np.ndarray): One-hot encoded test labels.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.

    Returns:
        Model: The trained model.
    """
    print("[INFO] Training network...")
    model.fit(
        trainData, trainLabels,
        validation_data=(testData, testLabels),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
    return model


def evaluate_model(model: Model, testData: np.ndarray, testLabels: np.ndarray) -> None:
    """
    Evaluates the trained model and prints classification report.

    Parameters:
        model (Model): The trained Keras model.
        testData (np.ndarray): Test image data.
        testLabels (np.ndarray): One-hot encoded test labels.
    """
    print("[INFO] Evaluating network...")
    predictions = model.predict(testData)
    print(classification_report(
        testLabels.argmax(axis=1),  # Convert one-hot encoded labels back to integers
        predictions.argmax(axis=1),  # Convert predictions to integer labels
        target_names=[str(x) for x in range(10)]  # Label names ('0' to '9')
    ))


def save_model(model: Model, model_path: str) -> None:
    """
    Saves the trained model to disk.

    Parameters:
        model (Model): The trained Keras model.
        model_path (str): Path where the model should be saved.
    """
    print(f"[INFO] Serializing digit model to {model_path}...")
    model.save(model_path)


if __name__ == "__main__":
    args = parse_arguments()

    INIT_LR = 1e-3  # Initial learning rate
    EPOCHS = 10  # Number of epochs
    BS = 128  # Batch size

    # Load and preprocess data
    trainData, trainLabels, testData, testLabels = preprocess_data()

    # Compile model
    model = compile_model(INIT_LR, width=28, height=28, depth=1, classes=10)

    # Train model
    model = train_model(model, trainData, trainLabels, testData, testLabels, batch_size=BS, epochs=EPOCHS)

    # Evaluate model
    evaluate_model(model, testData, testLabels)

    # Save model
    save_model(model, args["model"])

    # TO RUN:
    # In Bash:
    # python train_sudokunet.py --model models/sudoku_digit_recognizer.keras
