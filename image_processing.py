import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract
import os


def hor_line_removal(image):
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
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Draw the detected lines on a black canvas
    line_mask = np.zeros_like(grayed)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(grayed, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Invert the line mask
    line_mask_inv = cv2.bitwise_not(line_mask)

    # Combine the original image with the inverted line mask
    hor_lines_removed = cv2.bitwise_and(grayed, line_mask_inv)

    return hor_lines_removed


def preprocess_image(path, ruled):
    """
    preprocess_image() takes a path to the location of an image containing handwriting and processes it to be black and
    white and only include the words.

    :param ruled: boolean, whether the sample is on ruled paper
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

    if ruled:
        image = hor_line_removal(thresh)
        # checking to see if the image is already grayscale, if it is not then it needs to be
        # converted to work with the other functions
        if len(image.shape) == 2:
            thresh = image
        else:
            thresh = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # smooths the resulting image to fill in any holes present in the letters. the inversion is because it fills in the
    # image with white where I need it to fill in the image with black (the handwriting), larger ksize fills in more
    inverted = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
    morph = cv2.bitwise_not(morph)

    # Find contours which are edges/letters and draw them on a blank image to create a mask
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour_image = np.zeros_like(morph)
    cv2.drawContours(contour_image, contours, -1, 255, 7)

    # Invert the mask and apply to the morphology result
    inverted_contour_mask = cv2.bitwise_not(contour_image)
    result = cv2.bitwise_and(morph, morph, mask=inverted_contour_mask)

    # applying a median blur to get rid of all the thin lines/artifacts that get leftover
    median = cv2.medianBlur(result, 7)

    # leaves a weird black 5 pixel border so cropping it out for now
    median[5:-5, 5:-5]

    return median


def process_images_in_folder(input_folder, output_folder, ruled_paper):
    """
    process_images_in_folder() processes all images in a folder and then writes them to the specified folder, if the
    output folder does not exist it creates it.

    :param input_folder: relative path to the folder containing images for processing
    :param output_folder: relative path to the destination folder for the processed images
    :param ruled_paper: whether the sample handwriting is written on ruled paper, if it is the lines need to be removed
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Process each image in the input directory
    for image_file in image_files:
        i = 0
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f'processed{i}_{image_file}')
        # if the output path already exists (a processed image of the same name) we keep trying new output paths
        while os.path.exists(output_path):
            output_path = os.path.join(output_folder, f'processed{i}_{image_file}')
            i += 1

        processed_image = preprocess_image(input_path, ruled_paper)
        cv2.imwrite(output_path, processed_image)


def segment_words(input_path, output_folder, min_contour_area, max_contour_area):
    """
    segment_words() takes a path to an image and uses contours to detect the edges of letters/words will
    "cut up" the image to create new images containing single letters/words which are sent to the output folder.

    :param: input_path: the (relative/absolute) path to the image that you want to be segmented
    :param: output_folder: the (relative/absolute) path to the folder where you want to store the new images
    :param: min_contour_area: minimum area (in pixels) that you want to be considered a "word"
    :param: max_contour_area: maximum area (in pixels) that you want to be considered a "word"
    """
    # Load the image
    image = cv2.imread(input_path)

    # Convert the image to grayscale so that its format is suitable for inversion and contour finding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the image so that the text is white and the background is black
    inverted = cv2.bitwise_not(gray)

    # Find contours, detects white/light colours hence the need for inversion
    contours = cv2.findContours(inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        print("Specified output folder does not exist, creating it now...")
        os.makedirs(output_folder)

    # Extract individual words
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Filter contours that are in specified area range
        if max_contour_area > cv2.contourArea(contour) > min_contour_area:
            word_image = image[y:y + h, x:x + w]

            # Save the individual word image
            output_path = os.path.join(output_folder, f'word_{i}.png')
            cv2.imwrite(output_path, word_image)


def line_splitter(image):
    """
    line_splitter() works by averaging the horizontal darkness values locating the areas of lowest/0 darkness (white)
    and setting that to be a line break. similar can be done to separate into words
    :return:
    """
    # pass in a greyscale image as input, for each line sum the brightness and then divide by the number of pixels in
    # the line to get the average
    image_copy = image.copy()
    if len(image_copy.shape) == 2:
        pass
    else:
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    height, width = image_copy.shape[:2]
    print(height)
    print(width)
    row_avg = np.zeros(height)
    # for each line sum the brightness and then divide by the number of pixels in the line to get the average
    for i in range(height):
        row_avg[i] = np.sum(image_copy[i, :]) / width

    # sort the array and get the value of the highest 10% of brightness value
    row_avg_sorted = np.sort(row_avg)
    # cutoff = row_avg_sorted[int(0.7*height)]

    cutoff = 250
    print(row_avg)
    # print(row_avg_sorted)
    # for testing cutoff purposes set the lines deemed to be bright to grey
    for j in range(height):
        if row_avg[j] >= cutoff:
            pass
            image_copy[j, :] = 127  # testing with a grey value to see if logic works
    return image_copy



def word_splitter():
    pass
