from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model


class SudokuNet:
    """
    A convolutional neural network (CNN) architecture for recognizing handwritten digits
    in Sudoku puzzles. The network consists of convolutional layers, activation functions,
    pooling layers, fully connected (dense) layers, and dropout for regularization.
    """

    @staticmethod
    def build(width: int, height: int, depth: int, classes: int) -> Model:
        """
        Builds and returns the SudokuNet CNN model.

        Parameters:
        - width (int): The width of the input image.
        - height (int): The height of the input image.
        - depth (int): The number of channels in the input image (e.g., 1 for grayscale, 3 for RGB).
        - classes (int): The number of output classes (digits 0-9, typically 10 classes).

        Returns:
        - Model: A compiled Keras Sequential model.
        """
        # Initialize the model with an explicit Input layer
        model = Sequential([
            Input(shape=(height, width, depth)),

            # First set of CONV => RELU => POOL layers
            Conv2D(32, (5, 5), padding="same"),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),

            # Second set of CONV => RELU => POOL layers
            Conv2D(32, (3, 3), padding="same"),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),

            # First set of FC => RELU layers
            Flatten(),
            Dense(64),
            Activation("relu"),
            Dropout(0.5),

            # Second set of FC => RELU layers
            Dense(64),
            Activation("relu"),
            Dropout(0.5),

            # Softmax classifier
            Dense(classes),
            Activation("softmax")
        ])

        return model
