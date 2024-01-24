# import numpy as np
# import tensorflow as tf
# import keras
# from keras.layers import Rescaling
# from keras import layers
#
# # Images that are input are of size (None, None, 3) as they are rgb images (even though they are black and white)
# # and they are of variable height and width
#
# # creating the training dataset from the pre-processed cut up letters
# dataset = keras.utils.image_dataset_from_directory('individual_letters', batch_size=64, image_size=(None, None))
#
# # Let's say we expect our inputs to be RGB images of arbitrary size
# inputs = keras.Input(shape=(None, None, 3))
#
# # Rescale images to [0, 1]
# x = Rescaling(scale=1.0 / 255)(inputs)
#
# # Apply some convolution and pooling layers
# x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=(3, 3))(x)
# x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=(3, 3))(x)
# x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
#
# # Apply global average pooling to get flat feature vectors
# x = layers.GlobalAveragePooling2D()(x)
#
# # Add a dense classifier on top
# num_classes = 10
# outputs = layers.Dense(num_classes, activation="softmax")(x)
#
# model = keras.Model(inputs=inputs, outputs=outputs)
#
# model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3), loss=keras.losses.CategoricalCrossentropy())
#
# model.fit(dataset, epochs=10)

import numpy as np
import tensorflow as tf
import keras


# Get the data as Numpy arrays
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Build a simple model
inputs = keras.Input(shape=(28, 28))
x = keras.layers.Rescaling(1.0 / 255)(inputs)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation="relu")(x)
x = keras.layers.Dense(128, activation="relu")(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()

# Compile the model
optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")

# Train the model for 10 epoch from Numpy data
batch_size = 64
print("Fit on NumPy data")
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=10)
#
# # Train the model for 1 epoch using a dataset
# dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
# print("Fit on Dataset")
# history = model.fit(dataset, epochs=1)

model.save('test_model.keras')
