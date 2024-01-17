import numpy as np
import tensorflow as tf
import keras
from PIL import Image
from matplotlib import pyplot as plt

loaded_model = keras.models.load_model('test_model.keras')

# Load your image
img_path = 'numbers_classified/9_68.png'
img = Image.open(img_path).convert('L')  # Convert to grayscale
img = img.resize((28, 28))  # Resize to 28x28 pixels
plt.imshow(img, cmap='gray')
plt.show()
img_array = np.expand_dims(np.array(img), axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize pixel values

predictions = loaded_model.predict(img_array)

predicted_class = np.argmax(predictions)


print("Predictions", predictions)
print("Predicted class", predicted_class)

