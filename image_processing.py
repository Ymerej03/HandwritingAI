import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract
import os


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
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Process each image in the input directory
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f'processed_{image_file}')

        processed_image = preprocess_image(input_path)

        # Save the processed image
        cv2.imwrite(output_path, processed_image)


contrast = preprocess_image('sample_handwriting/image4.jpeg')
plt.imshow(contrast, cmap='gray')
plt.show()
