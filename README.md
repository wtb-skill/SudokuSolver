# Sudoku Solver

A computer vision-based Sudoku solver that can automatically detect, interpret, and solve Sudoku puzzles from images. 
This project combines convolutional neural networks (CNNs), image processing, and algorithmic problem-solving to extract 
Sudoku grids from photos, recognize digits, and solve the puzzle efficiently. It also features an 
interactive web interface for users to upload images and view results in real time.

### Features

- **Sudoku Image Processing**  
  Automatically detects, extracts, and corrects perspective distortion and noise in Sudoku grids using OpenCV. Robust to imperfect lighting, angles, and camera input.

- **Digit Recognition with CNN**  
  Classifies digits in the Sudoku grid using a custom-trained convolutional neural network.

- **Norvig Solver Integration**  
  Efficiently solves Sudoku puzzles using Peter Norvig’s constraint propagation and recursive backtracking algorithm.

- **Interactive Web Interface**  
  A user-friendly Flask-based frontend that allows users to upload Sudoku images and visualize the original, recognized, and solved grids side-by-side.

- **Model Evaluation Suite**  
  Includes tools for evaluating CNN performance with:
  - Confusion matrix visualization  
  - Per-class accuracy bar charts  
  - ROC curve generation  
  - Misclassified image display  
  - Exportable metric summaries

- **User Data Collection and Feedback Loop**  
  Allows users to provide manual labels for misclassified digits, creating a dataset for future model retraining and iterative improvement.

- **Modular and Extensible Architecture**  
  Clear separation between image processing, digit recognition, solving logic, evaluation, and web presentation. Designed to support easy debugging, scaling, and unit testing.

- **Synthetic Dataset Generator**  
  Scripted pipeline for generating digit datasets using a variety of fonts and augmentations. Facilitates model training without requiring large labeled datasets.

- **Model Training Pipeline**  
  Includes utilities for training and evaluating CNN models (e.g., Sudokunet) with visualization and performance tracking.

- **Session and Upload Management**  
  Secure and isolated handling of uploaded images and session data, with automatic cleanup on restart to maintain a clean working environment.

- **Image Debugging Visualizer**  
  Step-by-step image visualization for each stage of the processing pipeline, aiding in error analysis and debugging.

- **Deployment-Ready Design**  
  Lightweight Flask app architecture, easily deployable locally or to cloud environments (e.g., Render, Heroku, or Docker-based platforms).

- **Comprehensive Documentation and Code Comments**  
  Well-documented codebase with detailed inline docstrings, descriptive naming conventions, and a clean project structure.


## Project Structure

```
├── app.py                           # Flask app entry point
│
├── routes/
│   └── main.py                      # Blueprint for handling web routes and application logic
│
├── modules/
│   ├── board_display.py             # Generates visual representations of Sudoku boards
│   ├── digit_recognition.py         # CNN-based digit recognition from image patches
│   ├── debug.py                     # Tools for image collection and debugging
│   ├── user_data_collector.py       # Handles collection and validation of user-labeled data
│   └── sudoku_image/
│       ├── step_1_image_preprocessor.py   # Preprocesses raw image (grayscale, blur, thresholding)
│       ├── step_2_board_detector.py       # Detects and warps the Sudoku board from the image
│       ├── step_3_digit_extractor.py      # Extracts digit cells from the board
│       ├── step_4_digit_preprocessor.py   # Normalizes digits for model prediction
│       └── sudoku_pipeline.py             # Orchestrates full Sudoku image processing pipeline
│
├── modules/solving_algorithm/
│   ├── norvig_solver.py             # Norvig-style backtracking algorithm to solve Sudoku
│   └── sudoku_converter.py          # Converts between board matrix and string/dictionary formats
│
├── generate_model/
│   ├── digit_dataset/               # Generated synthetic digits for training
│   ├── evaluation_results/          # Evaluation results and graphs
│   ├── fonts/                       # Fonts used for synthetic digit generation
│   ├── test_dataset/                # Dataset for model testing
│   ├── generate_digit_dataset.py    # Script to synthesize digit images using fonts
│   ├── model_evaluator.py           # Evaluation logic for trained models
│   ├── sudokunet.py                 # CNN model architecture definition
│   └── sudokunet_trainer.py         # Training pipeline for digit recognition CNN
│
├── models/
│   └── sudoku_digit_recognizer.keras  # Trained Keras model for digit classification
│
├── templates/
│   ├── index.html                   # Homepage for uploading images
│   ├── solution.html                # Displays the solved Sudoku puzzle
│   ├── no_solution.html             # Informs user when puzzle can't be solved
│   └── collect_user_data.html       # UI for manually labeling digits for retraining
│
├── static/
│   ├── styles.css                   # CSS for styling HTML templates
│   └── logo_1.jpg                   # Logo image for the app
│
├── debug_images/                    # Stores debug images during development
├── collected_data/                  # Stores user-labeled digit samples
├── uploads/                         # Temporary storage for uploaded Sudoku images
├── flask_session_data/              # Stores session files across requests
│
├── requirements.txt                 # Python dependency list
└── README.md                        # You’re here!
```

## Usage

### 1. Install Project Dependencies

Start by installing the required Python dependencies. You can install them using `pip` from the `requirements.txt` file.

```
pip install -r requirements.txt
```
### 2. Run the Application

To start the Flask application, run the following command:
```
python app.py
```
The app will launch and be accessible at:
http://127.0.0.1:5000/
### 3. Using the Interface

Once the app is running, follow these steps:

    Upload a Sudoku Image
    On the homepage, you can upload a photo of a Sudoku puzzle. For best results, ensure the image is well-lit, the Sudoku grid is centered, and the digits are large and clear.

    Image Processing & Puzzle Recognition
    After uploading the image, the app will automatically:

        Detect the Sudoku grid: The app uses image processing techniques to find and extract the Sudoku grid.

        Recognize the digits: A Convolutional Neural Network (CNN) will classify the digits in the grid, identifying each digit from the image.

        Attempt to solve the puzzle: Using Peter Norvig’s efficient algorithm, the app will attempt to solve the Sudoku puzzle.

    View Unsolved and Solved Grids
    Once the image is processed and the puzzle is solved, the app will display:

        The original, unsolved Sudoku grid (as recognized by the app).

        The solved Sudoku grid, with the solution computed by the algorithm.

    Handling Unsolvable Puzzles
    If the app is unable to solve the puzzle (for example, due to poor image quality or unreadable digits), it will prompt you to manually label the misclassified digits.

        Label Misclassified Digits: You will be asked to identify any incorrectly recognized digits and assign the correct values. This helps improve the model for future puzzle-solving tasks.

        Once you’ve labeled the digits, the model will learn from your input, and you can use the labeled data to retrain the model, improving its performance over time.

## 4. (Optional) Model Generation and Evaluation

### Dataset Generation

Before training the model, you'll need a custom dataset of digit images. The `DigitDatasetGenerator` class generates 
images of digits (1-9) in different styles and levels of distortion. These images are used to train the neural network 
to recognize digits in Sudoku puzzles.

#### Steps to Generate a Custom Dataset

1. **Configure the Dataset Generator**
   
   The `DigitDatasetGenerator` class allows you to customize the generation of images. You can specify:
   - `image_size`: The size (width and height) of each image (default is 32x32).
   - `output_dir`: The directory where the generated images will be saved.
   - `num_samples`: The number of images to generate for each digit (default is 100).
   - `clean_proportion`: The fraction of clean, unaltered images (0 to 1).
   - `blur_level`, `shift_range`, `rotation_range`, `noise_level`: Parameters that control the distortion applied to images (random Gaussian blur, pixel shifts, rotations, and noise).
   - `fonts_dir`: The directory containing `.ttf` or `.otf` font files used to render the digits.

2. **Run Dataset Generation**
   
   After setting the parameters, instantiate the `DigitDatasetGenerator` and call `generate_images()` to create the dataset.

   ```python
   generator = DigitDatasetGenerator(
       num_samples=1000,
       clean_proportion=0.3,
       output_dir="test_dataset"  # Choose between 'digit_dataset' for training or 'test_dataset' for evaluation
   )
   generator.generate_images()

### Model Training

Once your dataset is ready, you can train a Convolutional Neural Network (CNN) to recognize the digits. The `SudokuNet` class is a simple CNN architecture built using Keras, consisting of convolutional layers, activation functions, max-pooling layers, and dense layers.

1. **Set Up the Trainer**

    The `SudokunetTrainer` class handles dataset loading, model compilation, training, and evaluation. Configure the trainer by specifying the following parameters:
   - `dataset_path`: The directory containing the dataset (either the one you just generated or a pre-existing one).
   - `model_output_path`: The path where the trained model will be saved.
   - `image_size`: The input size for images (default is 32x32).
   - `init_lr`: The learning rate for training (default is 1e-3).
   - `epochs`: The number of training epochs (default is 10).
   - `batch_size`: The batch size for training (default is 128).
   
2. **Customizing the Model Layers**

    Before running the trainer, you can customize the model architecture by modifying the `SudokuNet` class. This class defines the structure of the CNN, including convolutional layers, activation functions, pooling layers.
   
3. **Run the Trainer**

    After setting the parameters, instantiate the `SudokunetTrainer` and call `run()` to start the training process and save the trained model to model_output_path.

    ```python
    trainer = SudokunetTrainer(
        dataset_path="generate_model/digit_dataset",  
        model_output_path="models/sudoku_digit_recognizer.keras",  
        image_size=32,  
        init_lr=1e-3,  
        epochs=10,  
        batch_size=128  
    )    
    trainer.run()

## Authors

    Adam Bałdyga

## License

This project is licensed under the MIT License.
