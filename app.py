
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ImageFilter

MODEL_PATH = "models/handwritten_digit_cnn.keras"
model = tf.keras.models.load_model(MODEL_PATH)


def predict_digit(image):
    if image is None:
        return "Please upload an image.", {}, None

    # Convert uploaded image to grayscale
    image = ImageOps.exif_transpose(image).convert("L")
    arr = np.array(image).astype(np.uint8)

    # Improve contrast for photos and downloaded images
    image = Image.fromarray(arr)
    image = ImageOps.autocontrast(image)
    arr = np.array(image)

    # Convert dark digits on light backgrounds to
    # white digits on a black background
    if np.percentile(arr, 75) > 150:
        arr = 255 - arr

    # Remove very faint noise
    arr[arr < 35] = 0

    # Find the main digit area
    coords = np.argwhere(arr > 35)

    if len(coords) == 0:
        return "No digit detected. Try a clearer image.", {}, None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = arr[y_min:y_max + 1, x_min:x_max + 1]
    digit = Image.fromarray(cropped)

    # Keep the original aspect ratio
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Center digit on a 28x28 black canvas
    canvas = Image.new("L", (28, 28), 0)
    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2
    canvas.paste(digit, (x, y))

    # Prepare model input
    processed = np.array(canvas).astype("float32") / 255.0
    model_input = processed.reshape(1, 28, 28, 1)

    # Predict
    probabilities = model.predict(model_input, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit]) * 100

    result = (
        f"Predicted Digit: {predicted_digit}\n"
        f"Confidence: {confidence:.2f}%"
    )

    scores = {str(i): float(probabilities[i]) for i in range(10)}

    # Display the exact 28x28 image given to the CNN
    preview = Image.fromarray(processed * 255).convert("L")

    return result, scores, preview


with gr.Blocks(title="Handwritten Digit Recognition") as demo:
    gr.Markdown(
        """
        # Handwritten Digit Recognition
        ### CNN-based digit classification

        Upload a clear image containing one digit from 0 to 9.
        The app will display its prediction and processed input.
        """
    )

    with gr.Row():
        image_input = gr.Image(
            label="Upload a digit image",
            type="pil",
            sources=["upload", "webcam"],
            height=280
        )

        with gr.Column():
            prediction_output = gr.Textbox(
                label="Prediction",
                interactive=False
            )
            confidence_output = gr.Label(
                label="Confidence Scores",
                num_top_classes=10
            )
            preview_output = gr.Image(
                label="Processed image given to CNN",
                interactive=False,
                image_mode="L"
            )

    with gr.Row():
        predict_button = gr.Button("Predict Digit", variant="primary")
        clear_button = gr.Button("Clear")

    predict_button.click(
        fn=predict_digit,
        inputs=image_input,
        outputs=[
            prediction_output,
            confidence_output,
            preview_output
        ]
    )

    clear_button.click(
        fn=lambda: (None, "", {}, None),
        inputs=[],
        outputs=[
            image_input,
            prediction_output,
            confidence_output,
            preview_output
        ]
    )

    gr.Markdown(
        """
        **Note:** This model is trained on MNIST.
        Unusual handwriting, photographs, decorative fonts,
        and complex backgrounds may still cause incorrect predictions.
        """
    )

if __name__ == "__main__":
    demo.launch()
