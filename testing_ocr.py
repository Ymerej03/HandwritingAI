import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter, gaussian_laplace
from scipy.signal import find_peaks
import os

from image_processing import line_splitter, word_splitter, blob_detection, increase_handwriting_size


# # specifying the executable location
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
#
# # reads the image and draws bounding boxes around the located words
# img = cv2.imread('horizontal_line/line_9.jpg')
# d = pytesseract.image_to_data(img, lang='eng', config='psm 13', output_type=Output.DICT)
# n_boxes = len(d['level'])
# for i in range(n_boxes):
#     if int(d['conf'][i]) > 5:
#         (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# cv2.imwrite('test_word_split.png', img)


# print(pytesseract.image_to_string('contrast_handwriting/processed0_numbers.jpeg'))

# print(pytesseract.image_to_string('horizontal_line/line_9.jpg'))

def gray_hor_line_removal(image):
    """
    hor_line_removal() takes in an image (black and white) and then detects and removes the horizontal lines using
    hough line detection

    :param image: an ndarray
    :return hor_lines_removed: an nd array, horizontal lines removed
    """

    # making sure the input image is greyscale, so it can be passed to the thresholding function
    if len(image.shape) == 2:
        grayed = image  # Image is already grayscale
    else:
        # Convert the image to grayscale
        grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # denoising and applying some more thresholding so that hough line detection is not "distracted"
    denoised = cv2.fastNlMeansDenoising(grayed, None, h=5, templateWindowSize=10, searchWindowSize=25)
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Use HoughLinesP to detect lines in the image
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=100, minLineLength=250, maxLineGap=10)

    # Draw the detected lines on a black canvas
    line_mask = np.zeros_like(grayed)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Invert the line mask
    line_mask_inv = cv2.bitwise_not(line_mask)

    # Combine the original image with the inverted line mask
    # hor_lines_removed = cv2.bitwise_and(grayed, line_mask_inv)

    gray_hor_lines_removed = cv2.inpaint(grayed, line_mask, 9, cv2.INPAINT_TELEA)
    return gray_hor_lines_removed


# # page = cv2.imread('contrast_handwriting/processed0_image1.jpeg')
# # line_splitter(page)
#
# page = cv2.imread('sample_handwriting/image1.jpeg')
# gray_page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
# # gray_page = gray_hor_line_removal(gray_page)
# # plt.imshow(gray_page,cmap='gray')
# # plt.show()
# line_splitter(gray_page)
# # line = cv2.imread('horizontal_line/line_7.jpg')
# # word_splitter(line, 'individual_words')

# image = cv2.imread('horizontal_line/line_9.jpg')
#
# blob = blob_detection(image)
# # Create a Matplotlib figure and axes
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#
# # Display the first image on the left axis
# axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# axs[0].axis('off')
# axs[0].set_title('Image 1')
#
# # Display the second image on the right axis
# axs[1].imshow(cv2.cvtColor(blob, cv2.COLOR_BGR2RGB))
# axs[1].axis('off')
# axs[1].set_title('Image 2')
#
# # Show the plot
# plt.show()

def extract_words(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply preprocessing (e.g., thresholding)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    increased = increase_handwriting_size(binary_image, 6, 6)
    inverse_image = cv2.bitwise_not(increased)

    # Display the result
    plt.imshow(inverse_image, cmap='gray')
    plt.show()

    # Connected Component Analysis
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(inverse_image, connectivity=8)

    # Iterate through connected components
    for stat in stats[1:]:
        x, y, w, h = stat[0], stat[1], stat[2], stat[3]

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the result
    plt.imshow(image, cmap='gray')
    plt.show()


test = cv2.imread('horizontal_line/line_9.jpg')
# extract_words(image)

word_splitter('horizontal_line/line_5.jpg', 'individual_words')
