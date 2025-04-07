import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping



# Load images from the dataset
dataset_path = "dataset_resized"
batch_size = 32
img_size = (256, 256)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),  # Flip images randomly
    layers.RandomRotation(0.1),  # Slightly rotate images
    layers.RandomBrightness(0.2),  # Adjust brightness
    layers.RandomContrast(0.2)  # Adjust contrast
])

train_ds = image_dataset_from_directory(
    dataset_path,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=123
)

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

val_ds = image_dataset_from_directory(
    dataset_path,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=123
)


model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stopping])


loss, acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {acc:.2%}")

model.save("screen_photo_detector.keras")
