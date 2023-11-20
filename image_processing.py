import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract
import os


def hor_line_removal(image):
    """
    hor_line_removal() takes in an image (black and white) and then detects and removed the horizontal lines
    :param image: an ndarray
    :return hor_lines_removed: an nd array, horizontal lines removed
    """
    # modified from user nathancy, https://stackoverflow.com/a/59977864

    denoised = cv2.fastNlMeansDenoising(image, None, h=5, templateWindowSize=10, searchWindowSize=25)
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 2))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    contours = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for c in contours:
        cv2.drawContours(image, [c], -1, (255, 255, 255), thickness=4)

    # smooths the resulting image to fill in any holes present in the letters.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    hor_lines_removed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return hor_lines_removed


def preprocess_image(path):
    """
    preprocess_image() takes a path to the location of an image containing handwriting and processes it to be black and
    white and only include the words.
    :param path: the relative path from the project directory to the image you want to process
    :return: result an ndarray (2 dimensions), the image after the processing has been applied, two colours only
    """

    image = cv2.imread(path)
    grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoising using non-local means denoising
    denoised = cv2.fastNlMeansDenoising(grayed, None, h=5, templateWindowSize=10, searchWindowSize=25)
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=25, sigmaY=25)

    # divides the grayscale image by the blurred version of the image then scales it to ensure the values are in the
    # correct range (0-255). This is contrast normalisation, highlighting the edges/details (words).
    divided = cv2.divide(denoised, blurred, scale=255)

    # DIFFERENT THRESHOLDING METHODS

    # # sets the image to be in black and white only using Otsu's algorithm/method
    # # (maximising variance between classes of pixels)
    # thresh = cv2.threshold(divided, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # sets the image to be in black and white only using ToZero thresholding
    thresh = cv2.threshold(divided, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_TOZERO)[1]

    # Find contours which are edges/letters and draw them on a blank image to create a mask
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour_image = np.zeros_like(thresh)
    cv2.drawContours(contour_image, contours, -1, 255, 2)

    # smooths the resulting image to fill in any holes present in the letters.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Invert the mask and apply to the morphology result
    inverted_contour_mask = cv2.bitwise_not(contour_image)
    result = cv2.bitwise_and(morph, morph, mask=inverted_contour_mask)

    # # adjusting darkness of text
    # row, col = result.shape
    # for i in range(row):
    #     for j in range(col):
    #         if 244 > result[i][j] > 100:
    #             result[i][j] -= 100
    result[(result > 100) & (result < 245)] -= 100
    return result


def process_images_in_folder(input_folder, output_folder):
    """
    process_images_in_folder() processes all images in a folder and then writes them to the specified folder, if the
    output folder does not exist it creates it.
    :param input_folder: relative path to the folder containing images for processing
    :param output_folder: relative path to the destination folder for the processed images
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Process each image in the input directory
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f'processed_{image_file}')

        processed_image = preprocess_image(input_path)
        final_processed_image = hor_line_removal(processed_image)
        # Save the processed image
        cv2.imwrite(output_path, final_processed_image)


def segment_words(input_path, output_folder, min_contour_area):

    # Load the image
    image = cv2.imread(input_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the image
    inverted = cv2.bitwise_not(gray)

    # Find contours
    contours = cv2.findContours(inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract individual words
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out small contours
        if cv2.contourArea(contour) > min_contour_area:
            word_image = image[y:y + h, x:x + w]

            # Save or process the individual word image (e.g., save to file)
            output_path = os.path.join(output_folder, f'word_{i}.png')
            cv2.imwrite(output_path, word_image)
