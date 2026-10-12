
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import classification_report

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create output folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# 1. Load MNIST dataset
print("Loading MNIST dataset...")

(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# 2. Data preprocessing
# Normalize pixel values from 0-255 to 0-1
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Add channel dimension for CNN
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# 3. Build CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(
        32, kernel_size=(3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(
        64, kernel_size=(3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

# 4. Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 5. Train model
print("\nTraining started...")

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# 6. Evaluate model
print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# 7. Save trained model
model.save("models/handwritten_digit_cnn.keras")
print("\nModel saved successfully!")

# 8. Plot accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/training_curves.png")
plt.show()

# 9. Generate classification report
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

report = classification_report(y_test, y_pred)
with open("outputs/classification_report.txt", "w") as file:
    file.write(report)

print("\nClassification report saved!")
print("\nTraining completed successfully!")
