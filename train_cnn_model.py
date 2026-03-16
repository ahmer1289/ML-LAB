"""
Train CNN model for gender classification and save it
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parent
train_dir = BASE_DIR / 'data' / 'gender' / 'Train'
test_dir = BASE_DIR / 'data' / 'gender' / 'Test'
model_save_path_pkl = BASE_DIR / 'models' / 'cnn_model.pkl'
model_save_path_keras = BASE_DIR / 'models' / 'cnn_model.keras'

# Check if data exists
if not train_dir.exists():
    print(f"ERROR: Training data not found at {train_dir}")
    exit(1)

if not test_dir.exists():
    print(f"WARNING: Test data not found at {test_dir}, using train data for validation")
    test_dir = train_dir

print(f"Training data: {train_dir}")
print(f"Test data: {test_dir}")

# Load datasets
batch_size = 32
img_size = (128, 128)

print("\nLoading training dataset...")
train_dataset = image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="int",
)

print("Loading test dataset...")
test_dataset = image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="int",
)

# Normalize pixel values to [0, 1]
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

print("\nBuilding model...")
# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining model...")
# Train the model
history = model.fit(train_dataset, epochs=20, validation_data=test_dataset)

print("\nEvaluating model...")
# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
print(f"Test accuracy: {test_acc:.2f}")

print(f"\nSaving model to:")
print(f"  - Keras format: {model_save_path_keras}")
print(f"  - Pickle format: {model_save_path_pkl}")

model_save_path_keras.parent.mkdir(parents=True, exist_ok=True)
model_save_path_pkl.parent.mkdir(parents=True, exist_ok=True)

# Save as Keras format
model.save(str(model_save_path_keras))
print(f"✓ Keras model saved")

# Save as Pickle format
with open(str(model_save_path_pkl), 'wb') as f:
    pickle.dump(model, f)
print(f"✓ Pickle model saved")

print("\nDone! Model training complete.")
