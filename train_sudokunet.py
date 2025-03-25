# train_sudokunet.py
from Sudokunet import SudokuNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report
import argparse
import numpy as np
import cv2
import os

def parse_arguments() -> dict:
    """
    Parses command-line arguments.

    Returns:
        dict: A dictionary containing the parsed arguments.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Path to output model after training")
    return vars(ap.parse_args())

def load_custom_dataset(dataset_path: str, image_size: int = 32):
    """
    Loads the custom dataset from the specified directory with progress messages.

    :param dataset_path: Path to the dataset directory.
    :param image_size: The target size of images (assumes square images).
    :return: (trainData, trainLabels, testData, testLabels)
    """
    print("[INFO] Accessing the custom dataset...")

    data = []
    labels = []

    # Loop through digit folders (1-9)
    for digit in range(1, 10):  # Your dataset uses 1-9
        digit_path = os.path.join(dataset_path, str(digit))

        if not os.path.exists(digit_path):
            print(f"[WARNING] Skipping {digit_path} (Folder not found)")
            continue

        print(f"[INFO] Processing digit '{digit}'...")  # Show progress

        digit_images = os.listdir(digit_path)
        print(f"   - Found {len(digit_images)} images for digit {digit}")

        for i, image_name in enumerate(digit_images):
            img_path = os.path.join(digit_path, image_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            img = cv2.resize(img, (image_size, image_size))  # Resize to match training size
            img = img.astype("float32") / 255.0  # Normalize to [0, 1]
            img = np.expand_dims(img, axis=-1)  # Add channel dimension

            data.append(img)
            labels.append(digit - 1)  # labels (1-9)

            # Print progress every 1000 images (optional)
            if i % 1000 == 0 and i > 0:
                print(f"     - Processed {i} images for digit {digit}...")

    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # One-hot encode labels
    labels = to_categorical(labels, num_classes=9)  # 9 classes
    print(f"[INFO] Total dataset size: {len(data)} images")

    # Shuffle data
    print("[INFO] Shuffling dataset...")
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data, labels = data[indices], labels[indices]

    # Split into train/test sets (e.g., 80% train, 20% test)
    split_index = int(0.8 * len(data))
    trainData, testData = data[:split_index], data[split_index:]
    trainLabels, testLabels = labels[:split_index], labels[split_index:]

    print(f"[INFO] Dataset split into {len(trainData)} training and {len(testData)} testing images")

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

    # Convert one-hot encoded labels back to integers
    test_labels = testLabels.argmax(axis=1)
    predictions = predictions.argmax(axis=1)

    # The labels are from 1 to 9, so we want target_names to reflect that range.
    print(classification_report(
        test_labels,  # Ground truth labels
        predictions,  # Predicted labels
        target_names=[str(x) for x in range(1, 10)]  # Label names ('1' to '9')
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
    # trainData, trainLabels, testData, testLabels = preprocess_data()
    DATASET_PATH = "generate_model/digit_dataset"  # Path to your generated dataset
    trainData, trainLabels, testData, testLabels = load_custom_dataset(DATASET_PATH, image_size=32)

    # Compile model
    model = compile_model(INIT_LR, width=32, height=32, depth=1, classes=9)

    # Train model
    model = train_model(model, trainData, trainLabels, testData, testLabels, batch_size=BS, epochs=EPOCHS)

    # Evaluate model
    evaluate_model(model, testData, testLabels)

    # Save model
    save_model(model, args["model"])

    # TO RUN:
    # In Bash:
    # python train_sudokunet.py --model models/sudoku_digit_recognizer.keras
