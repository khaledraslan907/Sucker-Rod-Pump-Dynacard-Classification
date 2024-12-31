import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# === Define Paths ===
REFERENCE_FOLDER = r"E:\Data analysis\Projects\Petroleum\SRP\Dynacard\Lufkin Cards"  # Replace with your reference folder path
TEST_IMAGE_PATH = r"E:\Data analysis\Projects\Petroleum\SRP\Dynacard\Test 1.jpg"  # Replace with the test image path

# === Image Parameters ===
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# === Data Augmentation (Light Only) ===
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize images
    shear_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    rotation_range=5,  # Very small rotation
    horizontal_flip=False,
    validation_split=0.2  # 80% training, 20% validation
)

# === Load Training and Validation Data ===
train_generator = datagen.flow_from_directory(
    REFERENCE_FOLDER,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    REFERENCE_FOLDER,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# === Model Architecture ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# === Compile the Model ===
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Train the Model ===
EPOCHS = 20
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS
)

# === Plot Training History ===
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# === Save the Model ===
model.save("sucker_rod_pump_model.h5")
print("Model saved as sucker_rod_pump_model.h5")

# === Test the Model with a New Image ===
def predict_image(model_path, image_path):
    model = tf.keras.models.load_model(model_path)
    img = image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    
    predictions = model.predict(img_array)
    class_indices = {v: k for k, v in train_generator.class_indices.items()}
    predicted_class = class_indices[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%")

# Test with your image
predict_image("sucker_rod_pump_model.h5", TEST_IMAGE_PATH)
