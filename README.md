# Sudoku Solver

A computer vision-based Sudoku solver that can automatically detect, interpret, and solve Sudoku puzzles from images. 
This project combines **convolutional neural networks (CNNs)**, **image processing**, and **algorithmic problem-solving** to extract 
Sudoku grids from photos, recognize digits, and solve the puzzle efficiently. It also features an 
interactive web interface for users to upload images and view results in real time.

## Features

- 📷 **Image Processing** – Detects and extracts Sudoku grids from real-world photos using OpenCV.
- 🔢 **Digit Recognition (CNN)** – Recognizes digits using a custom-trained convolutional neural network.
- 🧩 **Puzzle Solver** – Uses Peter Norvig’s constraint propagation and backtracking algorithm.
- 🌐 **Web Interface** – Flask-based web UI for uploading, solving, and visualizing Sudoku puzzles.
- 📈 **Model Evaluation Tools** – Confusion matrix, per-class accuracy, misclassified images, and more.
- 📝 **User Feedback Loop** – Allows manual correction of misclassified digits to improve the model.
- 🧱 **Modular Architecture** – Cleanly separated pipeline for processing, solving, evaluation, and UI.
- 🔧 **Dataset & Model Tools** – Includes optional tools for generating synthetic and real-world digit datasets, plus CNN training utilities.

## Project Structure

```
├── app.py                           # Flask app entry point
│
├── routes/
│   ├── sudoku_solver.py                      # Blueprint for handling web routes and main pipeline logic
│   └── dev_tools.py                          # Blueprint for handling web routes and dev_tools logic
│
├── modules/
│   ├── board_display.py             # Generates visual representations of Sudoku boards
│   ├── debug.py                     # Tools for image collection and debugging
│   ├── digit_recognition.py         # CNN-based digit recognition from image patches
│   ├── sudoku_image_pipeline/
│   │    ├── step_1_image_preprocessor.py   # Preprocesses raw image (grayscale, blur, thresholding)
│   │    ├── step_2_board_detector.py       # Detects and warps the Sudoku board from the image
│   │    ├── step_3_digit_extractor.py      # Extracts digit cells from the board
│   │    ├── step_4_digit_preprocessor.py   # Normalizes digits for model prediction
│   │    └── sudoku_pipeline.py             # Orchestrates full Sudoku image processing pipeline
│   │
│   ├── solving_algorithm/
│   │   ├── norvig_solver.py             # Norvig-style backtracking algorithm to solve Sudoku
│   │   └── sudoku_converter.py          # Converts between board matrix and string/dictionary formats
│   │
│   └── utils/                           # utility scripts for main pipeline
│
├── dev_tools/
│   ├── data_utils/
│   │     ├── digit_dataset/                          # Synthetic digits generated for training
│   │     ├── fonts/                                  # Fonts used for synthetic digit generation
│   │     ├── sudoku_digit_image_extractor/
│   │     │    ├── extracted_digit_images/            # Output: extracted individual digit images
│   │     │    ├── sudoku_images/                     # Input: real-world Sudoku puzzles
│   │     │    └── sudoku_digit_image_extractor.py    # Script to extract digits from Sudoku puzzles
│   │     │
│   │     ├── test_dataset/                           # Dataset for model testing
│   │     ├── dataset_evaluation.py                   # Evaluate dataset quality (e.g., label distribution, missing digits)
│   │     └── generate_digit_dataset.py               # Script to generate synthetic digit dataset using fonts
│   │
│   └── model_utils/
│      ├── evaluation_results/             # Saved evaluation metrics, plots, and reports
│      ├── model_evaluator.py              # Logic for evaluating trained models on test datasets
│      ├── sudokunet.py                    # CNN model architecture definition
│      ├── sudokunet_trainer.py            # Model training pipeline and tools
│      └── test.json                       # JSON file with labeled test data for recognition evaluation
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
├── uploads/                         # Storage for uploaded Sudoku images
├── flask_session_data/              # Stores session files across requests
│
├── requirements.txt                 # Python dependency list
└── README.md                        # You’re here!
```

## Usage


### 1. Install Python 3.12.6

### 2. Install Project Dependencies

Start by installing the required Python dependencies. You can install them using `pip` from the `requirements.txt` file.

```
pip install -r requirements.txt
```

### 3. Run the Application

To start the Flask application, run the following command:
```
python app.py
```
The app will launch and be accessible at:
http://127.0.0.1:5000/

### 4. Upload & Solve Sudoku Puzzles

1. **Upload an Image** <br>
    Upload a photo of a Sudoku grid — centered, well-lit, and clearly visible digits work best.
2. **Automatic Processing** <br>
    The app will:
   - Detect and extract the Sudoku board
   - Recognize digits using the CNN
   - Solve the puzzle using Norvig’s algorithm
3. **View Results** <br>
You’ll see:
   - The detected grid
   - The recognized digits
   - The solved puzzle (if solvable)
4. **Handle Unsolvable or Misread Grids** <br>
If a puzzle cannot be solved due to digit recognition errors, you can:
    - Manually label incorrect digits
    - Automatically solve the puzzle using the corrected input
    - Help improve the model for future usage
    - Optionally use this feedback to retrain the digit recognizer

### ⚙️ Optional: Train Your Own Digit Recognizer ###
### Dataset Generation

Before training the model, you'll need a custom dataset of digit images. The `DigitDatasetGenerator` class generates 
images of digits (1-9) in different styles and levels of distortion. Alternatively you can use 
`SudokuDigitImageExtractor` to prepare the more real-world based sudoku digits dataset. These images are used to train 
the neural network to recognize digits in Sudoku puzzles.


#### **Option 1**: Steps to Generate a Custom Synthetic Dataset

1. **Configure the Dataset Generator**
   
   The `DigitDatasetGenerator` class allows you to customize the generation of images. You can specify:
   - `image_size`: The size (width and height) of each image (default is 32x32).
   - `output_dir`: The directory where the generated images will be saved.
   - `num_samples`: The number of images to generate for each digit (default is 10000).
   - `clean_proportion`: The fraction of clean, unaltered images (0 to 1).
   - `blur_level`, `shift_range`, `rotation_range`, `noise_level`: Parameters that control the distortion applied to images (random Gaussian blur, pixel shifts, rotations, and noise).
   - `fonts_dir`: The directory containing `.ttf` or `.otf` font files used to render the digits.
   - `font_sizes`: A list of font point sizes to randomly choose from when rendering digits (e.g. [24, 28, 32]), allowing variation in digit appearance.

2. **Run Dataset Generation**
   
   After setting the parameters, instantiate the `DigitDatasetGenerator` and call `generate_images()` to create the dataset.

   ```python
   generator = DigitDatasetGenerator(
       num_samples=1000,
       clean_proportion=0.3,
       output_dir="digit_dataset"
   )
   generator.generate_images()

3. **Collect digit images from `generate_model/digit_dataset` folder.**

#### **Option 2**: Steps to Generate a Real-World Sudoku Digit Dataset

1. **Place sudoku images into `generate_model/sudoku_digit_image_extractor/sudoku_images` folder.**

2. **Run Dataset Generation**
   
   After setting the parameters, instantiate the `SudokuDigitImageExtractor` and call `process_images()` to create the dataset.

   ```python
    input_folder = "sudoku_images"
    output_folder = "extracted_digit_images"

    extractor = SudokuDigitImageExtractor(input_folder, output_folder)
    extractor.process_images()

3. **Collect digit images from `generate_model/sudoku_digit_image_extractor/extracted_digit_images` folder and label them into correct subfolders (1 to 9)**

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

## 🔌 API Route Map

### Main Sudoku Solver (/)

| Route | Method | Description                                                                                                        |
|-------|--------|--------------------------------------------------------------------------------------------------------------------|
| `/` | `GET` | Clears the session, resets the image collector, and renders the homepage (`index.html`).                           |
| `/process-sudoku-image` | `POST` | Accepts uploaded Sudoku image, processes it, recognizes digits, solves it, and renders the solution page or error. |
| `/debug-image/<step_name>` | `GET` | Serves a debug image for a specific step (used for debugging image processing pipeline).                           |
| `/handle-collect-decision` | `POST` | Handles whether the user wants to label misclassified digits or return to homepage.                                |
| `/correct-and-solve` | `POST` | Solves Sudoku with user input and stores corrected data for training.                                              |
| `/uploads/<filename>` | `GET` | Serves an uploaded file from the uploads/ directory.                                                       |

### Developer Tools (/dev)

| Route | Method | Description                                                                                                        |
|-------|--------|--------------------------------------------------------------------------------------------------------------------|
| `/dev/process-all-sudoku-images` | `GET` | Automatically processes all Sudoku images in uploads/ folder, solves them, and moves to solved/ or unsolved/ folders. |
| `/dev/process-test-dataset` | `GET` | Extracts unsolved boards from images in uploads/clean/ and saves them into test.json.                         |
| `/dev/move-skipped-files` | `POST` | Moves images from uploads/test_folder/ that were not included in test.json back to uploads/.                              |
| `/dev/evaluate-recognition-accuracy` | `GET` | Evaluates digit recognition accuracy by comparing predicted boards to ground-truth labels in test.json.                                              |

## 🙋 FAQ ##

**Do I need to train a model to use the app?**<br>
No! A pre-trained model is included and used by default.

**What if the app misreads a digit?**<br>
You can manually correct it using the feedback interface. These corrections can be saved and later used to improve the model.

**Where do I find the saved data to improve the model?**<br>
They're saved inside collected_data/ folder. Each image already labeled and model ready.

## Authors

    Adam Bałdyga

## License

This project is licensed under the MIT License.
