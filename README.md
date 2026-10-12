# Handwritten Digit Recognition using CNN

A Deep Learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Features

- Predict digits from 0 to 9.
- Upload digit images through a Gradio web interface.
- Display prediction confidence scores for all 10 digits.
- Preview the processed image given to the CNN model.
- Train the CNN model using TensorFlow and Keras.
- Generate training curves and a classification report.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Gradio
- Pillow
- Matplotlib
- MNIST Dataset

## Project Structure

```text
Handwritten-Digit-Recognition/
├── app.py
├── train.py
├── models/
│   └── handwritten_digit_cnn.keras
├── outputs/
│   ├── classification_report.txt
│   └── training_curves.png
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Pooja06095/Handwritten-Digit-Recognition.git
cd Handwritten-Digit-Recognition
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows Git Bash:

```bash
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install tensorflow gradio numpy matplotlib scikit-learn pillow
```

### 4. Run the application

```bash
python app.py
```

Open the local URL displayed in the terminal to use the application.

## Train the Model

To train the CNN model again, run:

```bash
python train.py
```

The trained model and evaluation outputs are saved in the `models/` and `outputs/` folders.

## Limitations

The model is trained on the MNIST dataset. Predictions may be incorrect for unusual handwriting, photographs, decorative fonts, or images with complex backgrounds.

## Future Improvements

- Improve image preprocessing and prediction accuracy.
- Experiment with data augmentation.
- Compare different CNN architectures.
- Deploy the application online.

## Author

**Pooja Goswami**

GitHub: [Pooja06095](https://github.com/Pooja06095)

---

*Built as a Deep Learning project to explore CNN-based image classification.*