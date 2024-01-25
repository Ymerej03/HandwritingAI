import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract
import os
from scipy.ndimage import gaussian_filter1d, gaussian_filter, gaussian_laplace
from scipy.signal import find_peaks


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

    # leaves a weird black 5 pixel border so cropping it out
    crop_image = median[5:-5, 5:-5]

    return crop_image


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

    # Apply dilation to blob the words
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(inverted, kernel, iterations=2)

    # Find contours, detects white/light colours hence the need for inversion
    contours = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

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


def line_splitter(image, write=False):
    """
    line_splitter() works by averaging the horizontal darkness values, smoothing the projected averages and then
    locating the areas of maximum brightness,  setting that to be a line break. then it takes the locations of the line
    breaks and creates new images for each line.
    :param: image, a grayscale image of handwritten text to split into lines
    :param: write, boolean, whether to write the lines to a folder
    :return:
    """

    height, width = image.shape[:2]
    projection = np.zeros(height)
    for i in range(height):
        projection[i] = np.average(image[i][:])

    smoothed_projection = gaussian_filter1d(projection, sigma=16)

    # Find local maxima
    peaks, _ = find_peaks(smoothed_projection)

    # # for testing, draws lines on the image where the line breaks are detected
    # image_copy = image.copy()
    # for j in range(len(peaks)):
    #     cv2.line(image_copy, (0, peaks[j]), (image_copy.shape[1], peaks[j]), color=(0, 255, 0))
    if write:
        # Create and save horizontal lines
        output_folder = 'horizontal_line'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for i in range(len(peaks) - 1):
            line = image[peaks[i]:peaks[i + 1], :]
            base_filename = f'line_{i + 1}.jpg'
            output_path = os.path.join(output_folder, base_filename)

            # Check if the file already exists
            while os.path.exists(output_path):
                i += 1
                base_filename = f'line_{i + 1}.jpg'
                output_path = os.path.join(output_folder, base_filename)

            cv2.imwrite(output_path, line)
    else:
        lines = []
        for i in range(len(peaks) - 1):
            line = image[peaks[i]:peaks[i + 1], :]
            lines.append(line)
        return lines


def word_splitter_deprecated(line_image):
    # doesnt appear to be working as well as it needs to to be used
    """
        line_splitter() works by averaging the horizontal darkness values, smoothing the projected averages and then
        locating the areas of maximum brightness,  setting that to be a line break. then it takes the locations of the line
        breaks and creates new images for each line.
        :param: image, a grayscale image of handwritten text to split into lines
        :return:
        """

    height, width = line_image.shape[:2]
    projection = np.zeros(width)
    for i in range(width):
        projection[i] = np.average(line_image[:, i])

    smoothed_projection = gaussian_filter1d(projection, sigma=49)

    # Find local maxima
    peaks, _ = find_peaks(smoothed_projection)

    # for testing, draws lines on the image where the line breaks are detected
    image_copy = line_image.copy()
    for j in range(len(peaks)):
        cv2.line(image_copy, (peaks[j], 0), (peaks[j], image_copy.shape[0]), color=(0, 255, 0))

    # # Create and save horizontal lines
    # output_folder = 'segmented_words'
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    # for i in range(len(peaks) - 1):
    #     line = line_image[:, peaks[i]:peaks[i + 1]]
    #     output_path = os.path.join(output_folder, f'word_{i + 1}.jpg')
    #     cv2.imwrite(output_path, line)

    return image_copy


def increase_handwriting_size(image, dilation_factor, iterations):

    image_copy = image.copy()

    # Create a kernel for erosion
    kernel = np.ones((dilation_factor, dilation_factor), np.uint8)

    # erode the binary image
    dilated_image = cv2.erode(image_copy, kernel, iterations=iterations)

    # Invert the eroded image to get black outlines
    outlined_image = cv2.bitwise_not(dilated_image)

    return outlined_image


def blob_detection(line_image):
    # Ensure the input image is grayscale
    if len(line_image.shape) > 2:
        line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    # Define anisotropic standard deviations for x and y axes
    sigma_x = 4
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
    kernel = np.ones((5, 5), np.uint8)
    blob_mask = cv2.morphologyEx(thresholded_result, cv2.MORPH_OPEN, kernel)

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


def word_splitter(image, output_folder=None, write=False, min_contour=1000, max_contour=10000000):
    """
    splits an image of a line of handwritten text into words based on contours found from blobbing the words.
    it orders them based on the english left-to-right reading with contours starting with lower x values added first

    :param image: image to split into words, should be greyscale but will be converted if not
    :param output_folder: the folder you are writing the words/images found to, default None as not writing
    :param write: whether you are writing the split words to a folder, default is to return list of images not write
    :param min_contour: minimum size of contour to be considered
    :param max_contour: maximum size of contour to be considered
    :return word_image: list of image arrays of the segmented words, only returns if write=False
    """

    # making sure the input image is greyscale
    if len(image.shape) == 2:
        gray = image  # Image is already grayscale
    else:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blob = increase_handwriting_size(gray, 6, 6)
    # Find contours, detects white/light colours hence the need for inversion
    contours = cv2.findContours(blob, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    if write:
        if output_folder is not None:
            # Create the output folder if it doesn't exist
            if not os.path.exists(output_folder):
                print("Specified output folder does not exist, creating it now...")
                os.makedirs(output_folder)

            # Extract individual words
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                # Filter contours that are in specified area range
                if max_contour > cv2.contourArea(contour) > min_contour:
                    word_image = image[y:y + h, x:x + w]
                    # Save the individual word image
                    output_path = os.path.join(output_folder, f'{image}_word_{i}.png')
                    if word_image.any():
                        cv2.imwrite(output_path, word_image)
    else:
        word_image = []
        # Extract individual words and sort them based on the leftmost point
        for contour in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
            x, y, w, h = cv2.boundingRect(contour)

            # Filter contours that are in specified area range and append to the list of words
            if max_contour > cv2.contourArea(contour) > min_contour:
                word_image.append(image[y:y + h, x:x + w])

        return word_image


def extract_words_from_image(image_path, ruled):
    """
    takes in the path to the image and then splits the image into lines and then those lines into words, it then

    :param image_path: a string, path to a colour image that we want to extract the words from
    :param ruled: a boolean, whether the paper has horizontal ruled lines
    :return ordered_words: a list of images following standard english reading, 0=top left word ... x=bottom right word
    """
    image = preprocess_image(image_path, ruled)
    # splits the images into lines, image_lines is a list of those lines where image_lines[0] is the topmost line
    # image_lines[1] is the line below etc
    image_lines = line_splitter(image)

    ordered_words = []
    for i in range(len(image_lines)):
        words = word_splitter(image_lines[i])
        for j in range(len(words)):
            ordered_words.append(words[j])
        # could append a symbol here to indicate that a newline has started, then use that to split transcriptions up
        ordered_words.append(np.array([-999]))
    return ordered_words
