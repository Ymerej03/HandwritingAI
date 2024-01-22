import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter, gaussian_laplace
from scipy.signal import find_peaks
import os

from image_processing import line_splitter, word_splitter

#
# # specifying the executable location
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
#
# # reads the image and draws bounding boxes around the located words
# img = cv2.imread('contrast_handwriting/processed0_image3.jpeg')
# d = pytesseract.image_to_data(img, output_type=Output.DICT)
# n_boxes = len(d['level'])
# for i in range(n_boxes):
#     if int(d['conf'][i]) > 5:
#         (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# cv2.imwrite('test_output_image5.png', img)
#
#
# print(pytesseract.image_to_string('contrast_handwriting/processed0_numbers.jpeg'))


test_image = cv2.imread('horizontal_strips/strip_18.jpg')
test_word = word_splitter(test_image)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image
axs[0].imshow(test_image, cmap='gray')
axs[0].set_title('Original Image')

axs[1].imshow(test_word, cmap='gray')
axs[1].set_title('Word detection')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()


def blob_detection(line_image):
    # Ensure the input image is grayscale
    if len(line_image.shape) > 2:
        line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    # Define anisotropic standard deviations for x and y axes
    sigma_x = 5
    sigma_y = 2
    # Apply anisotropic Gaussian filter
    smoothed_image = gaussian_filter(line_image, sigma=(sigma_y, sigma_x))

    # Define the standard deviation for Laplacian of Gaussian
    sigma_laplace = 1.0

    # Apply Laplacian of Gaussian to the smoothed image
    result = gaussian_laplace(smoothed_image, sigma=sigma_laplace)

    # Thresholding to highlight blobs
    _, thresholded_result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: Morphological operations for further blob enhancement
    kernel = np.ones((7, 7), np.uint8)
    blob_mask = cv2.morphologyEx(thresholded_result, cv2.MORPH_CLOSE, kernel)

    # Apply the blob mask to the original image
    blobby_image = cv2.bitwise_and(line_image, line_image, mask=blob_mask)

    return blobby_image


def blur_and_threshold(image, blur_size=(10, 20), threshold=220):
    """
    Blur the input image and apply thresholding.

    Parameters:
    - image: Input image (numpy array).
    - blur_size: Size of the rectangle for averaging (tuple of two integers).

    Returns:
    - Resulting image after blurring and thresholding.
    """
    # Blur the image using a rectangle of specified size
    blurred_image = cv2.blur(image, blur_size)

    blurred_image[blurred_image < threshold] = 0

    return blurred_image


# image = cv2.imread('horizontal_strips/strip_19.jpg')
# blob = blob_detection(image)
# threshold = blur_and_threshold(image)
# fig, axs = plt.subplots(1, 3, figsize=(10, 5))
#
# # Display the original image
# axs[0].imshow(image, cmap='gray')
# axs[0].set_title('Original Image')
#
# axs[1].imshow(blob, cmap='gray')
# axs[1].set_title('Blob Detection')
#
# axs[2].imshow(threshold, cmap='gray')
# axs[2].set_title('Blur Detection')
#
# # Adjust layout for better spacing
# plt.tight_layout()
#
# # Show the plots
# plt.show()
