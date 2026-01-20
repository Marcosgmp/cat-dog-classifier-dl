import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set image size, batch size, and number of epochs
img_size = (224, 224)
batch_size = 32
epochs = 5

train_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "dataset/archive/training_set",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary"
)

test_data = test_gen.flow_from_directory(
    "dataset/archive/test_set",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary"
)

# Define the CNN architecture
model = tf.keras.Sequential([
    # First convolutional layer + max pooling
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    # Second convolutional
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten: convert feature maps into a 1D vector
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(128, activation="relu"),
    
    # Output layer with sigmoid activation for binary classification
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=test_data,
    epochs=epochs
)

# Saves the trained model to an .h5 file this allows anyone to load the model later without retraining
model.save("model/cat_dog_model.h5")

print("Modelo treinado e salvo com sucesso!")