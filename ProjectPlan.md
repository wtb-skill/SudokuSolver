# **Project Plan: Sudoku Solver from an Image**

## **1. Image Preprocessing (Computer Vision)**

**Objective:** Extract the Sudoku grid from the image and prepare it for digit recognition.

- **Steps:**
    
    **1️⃣ Convert an image to grayscale (simplifies data).**
    
    Convert the image to grayscale to remove color information and reduce complexity.
    
    **2️⃣ Apply Gaussian blur (reduces noise).**
    
    Smooth the image using Gaussian Blur to reduce noise and improve edge detection.
    
    **3️⃣ Apply adaptive thresholding (improves contrast between grid and background).**
    
    Use adaptive thresholding to dynamically adjust brightness differences and enhance the grid.
    
    Invert the colors so that the grid appears white on a black background.
    
    **4️⃣ Find contours and sort them by size (detects potential Sudoku grid).**
    
    Detect external contours in the thresholded image and sort them from largest to smallest.
    
    **5️⃣ Approximate the largest contour to a four-sided polygon (identifies the Sudoku board).**
    
    Approximate the shape of the largest contour and check if it has four sides to confirm it’s a Sudoku grid.
    
    **6️⃣ Apply a four-point perspective transform (warps Sudoku grid to a top-down view).**
    
    Warp the detected grid to a straightened, top-down view for better processing.
    
    **7️⃣ Extract individual Sudoku cells for digit recognition.**
    
    Divide the warped grid into individual cells and extract each one for digit recognition.
    

**Tools/Libraries:** OpenCV (Python)

---

## **2. Digit Recognition (Machine Learning Approach - CNN)**

**Objective:** Recognize the numbers in each cell of the Sudoku grid using ML.

### **Approach:**

✅ **CNN-based Recognition (Deep Learning)**

- Train a **Convolutional Neural Network (CNN)** on a dataset of Sudoku digits.
- Handles printed, handwritten, and distorted digits with high accuracy.
- Dataset: MNIST and/or a custom dataset of Sudoku images.

**Tools/Libraries:** OpenCV, TensorFlow/Keras (CNN)

---

## **3. Sudoku Solver (Algorithmic Approach)**

**Objective:** Solve the extracted Sudoku puzzle.

### **Approach:**

✅ **Backtracking Algorithm** – A simple but effective brute-force search for a valid solution.

✅ **Constraint Propagation** – Uses logical rules to reduce the search space before backtracking.

🔹 *(Optional ML, but less efficient: ML-based solvers exist, but traditional algorithms outperform them for Sudoku solving.)*

**Tools/Libraries:** NumPy, Python

---

## **4. Solution Display**

**Objective:** Present the solved Sudoku puzzle in a structured and visually clear format.

- **Steps:**
    
    **1️⃣ Display the solved Sudoku grid as an image or table.**
    
    Present the solution in a structured and easy-to-read format.
    
    **2️⃣ Differentiate between original and solved digits.**
    
    Use color or styling to highlight the newly filled numbers.
    
- **Approach:**
    - Use **Streamlit/Flask** for a web-based interactive display.
    - Or implement **Tkinter/PyQt** for a local GUI.
    - Ensure a **clean, responsive, and user-friendly interface**.

**Tools/Libraries:** Python, NumPy, OpenCV, Streamlit, Flask, Tkinter, PyQt

---

## **5. User Interface for Input/Output**

**Objective:** Allow users to upload an image and get the solved Sudoku as output.

### **Variants:**

- Command-line script
- Web app using Flask/Django
- GUI using Tkinter or Streamlit

---

## **6. (Optional) Handling Different Sudoku Sizes**

**Objective:** Extend the solver to recognize and solve Sudoku puzzles of various sizes using ML or OpenCV-based techniques.

### **Steps:**

1. **Detect Grid Size (ML-Based Classification or OpenCV-Based Detection)**
    - Train an ML model (CNN/Random Forest) to classify Sudoku grid sizes (4×4, 6×6, 9×9, etc.).
    - Alternatively, use OpenCV to detect grid lines and count intersections to determine the grid size.
2. **Adjust Digit Recognition Model**
    - Extend CNN-based digit recognition to handle numbers beyond 1–9 (e.g., hexadecimal for 16×16 puzzles).
3. **Adapt Solver Algorithm**
    - Modify backtracking and constraint propagation to work for different grid sizes.
4. **Solution Overlay**
    - Scale and position numbers appropriately for each Sudoku variant.

**Tools/Libraries:** OpenCV, TensorFlow/Keras (CNN), NumPy

---

## **7. Solution Verification (Automatic Checking)**

**Objective:** Automatically validate that the Sudoku solution satisfies all constraints (no duplicates in rows, columns, or subgrids).

### **Steps:**

1. **Check Row Constraints:**
    - Ensure that each row contains all digits from 1 to 9 without duplicates.
2. **Check Column Constraints:**
    - Ensure that each column contains all digits from 1 to 9 without duplicates.
3. **Check 3x3 Subgrid (Box) Constraints:**
    - Ensure that each of the nine 3x3 subgrids (boxes) contains all digits from 1 to 9 without duplicates.
4. **Overall Validation:**
    - Combine all checks (rows, columns, and subgrids) and return whether the solution is valid or not.

---

## **Technologies to Use**

- **Python** (Main language)
- **OpenCV** (Image processing)
- **TensorFlow/Keras** (CNN for digit recognition & grid classification)
- **NumPy** (Data handling)
- **Flask/Streamlit** (UI)

---

## **Next Steps**

1️⃣ Start with a small dataset of Sudoku images and preprocess them.

2️⃣ Train and test a CNN for digit recognition.

3️⃣ Implement the solver and validate results.

4️⃣ Integrate everything into a working pipeline.

[Different approaches to this problem](https://www.notion.so/Different-approaches-to-this-problem-1b4b9fbe714180dfa62aeb9e422fe45c?pvs=21)

[Solver variants](https://www.notion.so/Solver-variants-1b4b9fbe714180d4af9aedcd5efd0b8d?pvs=21)

[Optional functionality to consider (if time and skill allows)](https://www.notion.so/Optional-functionality-to-consider-if-time-and-skill-allows-1b4b9fbe71418025ab70f55c565a1369?pvs=21)

[Assignment week 1](https://www.notion.so/Assignment-week-1-1b4b9fbe7141803fac6dcdec31abba48?pvs=21)