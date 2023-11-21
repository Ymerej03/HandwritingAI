import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract
import os


def hor_line_removal(image):
    """
    hor_line_removal() takes in an image (black and white) and then detects and removed the horizontal lines,
    modified from user nathancy, https://stackoverflow.com/a/59977864
    :param image: an ndarray
    :return hor_lines_removed: an nd array, horizontal lines removed
    """

    # making sure the input image is greyscale, so it can be passed to the thresholding function
    if len(image.shape) == 2:
        grayed = image  # Image is already grayscale
    else:
        # Convert the image to grayscale
        grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(grayed, None, h=5, templateWindowSize=10, searchWindowSize=25)
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 5))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    contours = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for c in contours:
        cv2.drawContours(image, [c], -1, (255, 255, 255), thickness=3)

    # # smooths the resulting image to fill in any holes present in the letters. the inversion is because it fills in the
    # # image with white where I need it to fill in the image with black (the handwriting), larger ksize fills in more
    # inverted = cv2.bitwise_not(image)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    # morph = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
    # hor_lines_removed = cv2.bitwise_not(morph)
    return image


def preprocess_image(path):
    """
    preprocess_image() takes a path to the location of an image containing handwriting and processes it to be black and
    white and only include the words.
    :param path: the relative path from the project directory to the image you want to process
    :return: result an ndarray (2 dimensions), the image after the processing has been applied, two colours only
    """

    # Read in the image and convert it to greyscale, so it only has width and height (2D)
    image = cv2.imread(path)
    grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoising using non-local means denoising
    denoised = cv2.fastNlMeansDenoising(grayed, None, h=5, templateWindowSize=10, searchWindowSize=25)
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=25, sigmaY=25)

    # divides the grayscale image by the blurred version of the image then scales it to ensure the values are in the
    # correct range (0-255). This is contrast normalisation, highlighting the edges/details (words).
    divided = cv2.divide(denoised, blurred, scale=255)

    # sets the image to be in black and white only using Otsu's algorithm/method
    # based on maximising the variance between classes of pixels
    thresh = cv2.threshold(divided, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours which are edges/letters and draw them on a blank image to create a mask
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour_image = np.zeros_like(thresh)
    cv2.drawContours(contour_image, contours, -1, 255, 3)

    # smooths the resulting image to fill in any holes present in the letters. the inversion is because it fills in the
    # image with white where I need it to fill in the image with black (the handwriting), larger ksize fills in more
    inverted = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    morph = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
    morph = cv2.bitwise_not(morph)

    # Invert the mask and apply to the morphology result
    inverted_contour_mask = cv2.bitwise_not(contour_image)
    result = cv2.bitwise_and(morph, morph, mask=inverted_contour_mask)

    # applying a median blur to get rid of all the thin lines/artifacts that get leftover
    median = cv2.medianBlur(result, 5)

    return median


def process_images_in_folder(input_folder, output_folder, ruled_paper):
    """
    process_images_in_folder() processes all images in a folder and then writes them to the specified folder, if the
    output folder does not exist it creates it.
    :param ruled_paper: whether the sample handwriting is written on ruled paper, if it is the lines need to be removed
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
        # if the output path already exists (a processed image of the same name) we keep trying new output paths
        i = 0
        while os.path.exists(output_path):
            output_path = os.path.join(output_folder, f'processed{i}_{image_file}')
            i += 1

        processed_image = preprocess_image(input_path)
        if ruled_paper:
            processed_image = hor_line_removal(processed_image)
        # Save the processed image
        cv2.imwrite(output_path, processed_image)


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
