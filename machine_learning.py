import numpy as np
import tensorflow as tf
import keras

# creating the training dataset from the pre-processed cut up letters
dataset = keras.utils.image_dataset_from_directory('individual_letters', batch_size=64, image_size=(None, None))
