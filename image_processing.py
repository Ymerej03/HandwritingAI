import numpy as np
import cv2
import os
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def hor_line_removal(image):
    """hor_line_removal() removes the horizontal lines from a black and white image using hough line detection

    :param image: numpy.ndarray, a 2D array of the image (with horizontal lines)
    :return hor_lines_removed: numpy.ndarray, 2D array of the image with the horizontal lines removed
    """

    # making sure the input image is greyscale, so it can be passed to the thresholding function
    if len(image.shape) == 2:
        # Image is already grayscale
        gray_image = image
    else:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # denoising and applying some more thresholding so that hough line detection is not "distracted"
    denoised = cv2.fastNlMeansDenoising(gray_image, None, h=5, templateWindowSize=10, searchWindowSize=25)
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Use HoughLinesP to detect lines in the image
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Draw the detected lines on a black canvas
    line_mask = np.zeros_like(gray_image)
    # if lines is None then no lines were detected and the whole image should be kept
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(gray_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Invert the line mask
    line_mask_inv = cv2.bitwise_not(line_mask)

    # Combine the original image with the inverted line mask
    hor_lines_removed = cv2.bitwise_and(gray_image, line_mask_inv)

    return hor_lines_removed


def preprocess_image(path, ruled):
    """
    preprocess_image() takes a path to the location of an image containing handwriting and processes it to be black and
    white and only include the words.

    :param path: a string, the relative path to the image you want to process, must end in .jpg, .jpeg or .png.
    :param ruled: a bool, whether the sample is on ruled paper (True) or not (False).
    :return result: numpy.ndarray, a 2D array the image after the processing has been applied, black and white.
    """

    # Check if the provided path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"The provided path '{path}' does not exist.")

    # Read in the image and convert it to greyscale, so it only has width and height (2D)
    image = cv2.imread(path)
    grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoising using non-local means denoising, denoising takes a little while
    denoised = cv2.fastNlMeansDenoising(grayed, None, h=5, templateWindowSize=10, searchWindowSize=25)
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=25, sigmaY=25)

    # divides the grayscale image by the blurred version of the image then scales it to ensure the values are in the
    # correct range (0-255). This is contrast normalisation, highlighting the edges/details (words).
    divided = cv2.divide(denoised, blurred, scale=255)

    # sets the image to be in black and white only using Otsu's algorithm/method
    # based on maximising the variance between classes of pixels
    _, thresh = cv2.threshold(divided, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

    # floodfill to remove outline, this might accidentally remove some words/letters, but I think its worth it
    cv2.floodFill(result, None, (0, 0), 255)

    # applying a median blur to get rid of all the thin lines/artifacts that get leftover
    median = cv2.medianBlur(result, 7)

    # crop image is still used even though border has been removed because I don't want to debug line splitter
    cropped_image = median[:, 5:-5]

    return cropped_image


def increase_handwriting_size(image, dilation_factor, iterations):
    """increase_handwriting_size increases the black areas of an image (writing) using erosion.

    :param image: numpy.ndarray, black and white image.
    :param dilation_factor: int, how large the kernel is for erosion.
    :param iterations: int, how many times to apply the erosion to the image, larger number = larger writing.
    :return increased_image: numpy.ndarray, black and white image.
    """
    image_copy = image.copy()

    # Create a kernel for erosion
    kernel = np.ones((dilation_factor, dilation_factor), np.uint8)

    # erode the binary image, using erosion as the writing is black
    dilated_image = cv2.erode(image_copy, kernel, iterations=iterations)

    # Invert the eroded image to get black outlines
    increased_image = cv2.bitwise_not(dilated_image)

    return increased_image


def extract_outline(image):
    """ extract_outline extracts the outline of the original image but only with the black pixels that border
     white pixels in the 4 cardinal directions.
    :param image: numpy.ndarray, a 2D array of a black and white image.
    :return result_image: numpy.ndarray, a 2D array of a black and white image with the outline of the original image.
    """
    height, width = image.shape
    result_image = np.ones_like(image) * 255

    # Iterate over each pixel in the image
    for row in range(height):
        for col in range(width):
            # Check if the current pixel is black
            if image[row, col] == 0:
                # Check the 4 cardinal directions around the current pixel
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    r_neighbor, c_neighbor = row + dr, col + dc
                    # Check if the neighboring pixel is within bounds and white
                    if (0 <= r_neighbor < height) and (0 <= c_neighbor < width) and image[r_neighbor, c_neighbor] != 0:
                        # Copy the black pixel to the result image
                        result_image[row, col] = 0
                        # Stop checking other directions if a white neighbor is found
                        break

    return result_image


def radial_sweep(image, outlined_image, start_row, start_col, peak):
    """ radial_sweep implements the radial sweep algorith to trace outlines of letters

    :param image: numpy.ndarray, a 2D array of a black and white image.
    :param outlined_image: numpy.ndarray, a 2D array of a black and white image.
    :param start_row: an int, the row to start the radial sweep from.
    :param start_col: an int, the col to start the radial sweep from.
    :param peak: an int, one of the rows of maximum brightness same as start_row.
    :return row, col: int, int, the row and col position that radial sweep ends.
    :return  boundary: list, a list of the pixels traced by radial sweep.
    """
    height, width = image.shape[:2]
    max_iterations = 150
    max_distance = 150

    ccw_iterations = 0
    ccw_image = outlined_image.copy()
    # placeholder direction to check if iterated through all directions
    ccw_directions = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (9, 9)]
    ccw_boundary = []
    # ccw = 1 is success, 2 is failure
    ccw = 0

    cw_iterations = 0
    cw_image = outlined_image.copy()
    # placeholder direction to check if iterated through all directions
    cw_directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (9, 9)]
    cw_boundary = []
    # cw = 1 is success, 2 is failure
    cw = 0

    # Start from the given row and column
    row, col = start_row, start_col
    ccw_row, ccw_col = start_row, start_col

    # # mark starting pixel as grey
    # ccw_image[start_row, start_col] = 100
    # cw_image[start_row, start_col] = 127

    cw_boundary.append((start_row, start_col))
    ccw_boundary.append((start_row, start_col))

    # Radial search in the clockwise direction
    while True:
        for dr, dc in cw_directions:
            # if iterated through all directions must have failed/got stuck on isolated pixel
            if dr == 9:
                cw = 2
                break
            r_new, c_new = row + dr, col + dc  # Move in the radial direction
            if (r_new, c_new) not in cw_boundary:
                # if it is on the list it has been visited before and should be ignored
                if (0 <= r_new < height) and (0 <= c_new < width) and cw_image[r_new, c_new] == 0:
                    # Move to the next black pixel
                    row, col = r_new, c_new
                    cw_boundary.append((row, col))  # Add the boundary pixel to the list
                    cw_iterations += 1
                    break
        if (row, col) == (start_row, start_col):
            # if stuck on isolated pixel have failed
            cw = 2
            break
        elif np.abs(row - peak) > max_distance:
            # if travel too far from the line have failed
            cw = 2
            break
        elif row == peak and cw_iterations >= 3:
            # if looped all the way back to the start have successfully found the contour
            cw = 1
            break
        elif cw_iterations > max_iterations:
            # if it takes too many iterations has failed
            cw = 2
            break
        elif ccw == 1 and cw_iterations > len(ccw_boundary):
            # if already found a solution then this one "fails"
            cw = 2
            break
        elif row == 0 or row == height - 1:
            # reaches either end of image it failed
            cw = 2
            break
        elif cw == 2:
            # already failed
            break
    # Radial search in the counter-clockwise direction
    while True:
        for dr, dc in ccw_directions:
            # if iterated through all directions and haven't found anything must have failed/got stuck on isolated pixel
            if dr == 9:
                ccw = 2
                break
            ccw_r_new, ccw_c_new = ccw_row + dr, ccw_col + dc  # Move in the radial direction
            if (ccw_r_new, ccw_c_new) not in ccw_boundary:
                if (0 <= ccw_r_new < height) and (0 <= ccw_c_new < width) and ccw_image[ccw_r_new, ccw_c_new] == 0:
                    # Move to the next black pixel
                    ccw_row, ccw_col = ccw_r_new, ccw_c_new
                    ccw_boundary.append((ccw_row, ccw_col))  # Add the boundary pixel to the list
                    ccw_iterations += 1
                    break
        if (ccw_row, ccw_col) == (start_row, start_col):
            # if stuck on isolated pixel have failed
            ccw = 2
            break
        elif np.abs(ccw_row - peak) > max_distance:
            # if travel too far from the line have failed
            ccw = 2
            break
        elif ccw_row == peak and ccw_iterations >= 3:
            # if looped all the way back to the start have successfully found the contour
            ccw = 1
            break
        elif ccw_iterations > max_iterations:
            # if it takes too many iterations has failed
            ccw = 2
            break
        elif cw == 1 and ccw_iterations > len(cw_boundary):
            # if already found a solution then this one "fails"
            ccw = 2
            break
        elif ccw_row == 0 or ccw_row == height - 1:
            # reaches either end of image it failed
            ccw = 2
            break
        elif ccw == 2:
            # already failed
            break

    # checking all the cases of radial search
    if (ccw == 2) and (cw == 2):
        # both failed need to drill
        boundary = []
        row = start_row
        col = start_col
        while image[row, col] == 0:
            boundary.append((row, col))
            col -= 1
        return row, col + 1, boundary
    elif ccw == 2 and cw == 1:
        # cw was a success, ccw failed
        return row, col, cw_boundary
    elif ccw == 1 and cw == 2:
        # ccw was a success, cw failed
        return ccw_row, ccw_col, ccw_boundary
    elif ccw == 1 and cw == 1:
        # if both successes select the shorter route
        if len(cw_boundary) <= len(ccw_boundary):
            return row, col, cw_boundary
        else:
            return ccw_row, ccw_col, ccw_boundary
    else:
        raise ValueError("Radial sweep failed with unknown error.")


def droplet_line_splitter(image, write=False):
    """ droplet_line_splitter splits the paragraph image into lines. It does this by locating areas of maximum
    brightness and then at these locations, letting a 'drop' go from right to left, each time it encounters black
    it tries to go around it.

    :param image: numpy.ndarray, 2D array of the black and white image that you want to split into lines.
    :param write: bool, whether writing the lines to a folder, optional argument mainly used for debugging.
    :return list_of_lines: a list, lines split from the image top is first entry bottom is last.
    """
    # Increasing the text size to connect some ascenders and descenders to rest of letter and to remove jagged
    # pieces which cause radial sweep to get stuck
    structure = cv2.getStructuringElement(cv2.MORPH_RECT, [5, 5])
    image_copy = cv2.erode(image, structure)
    image_copy = cv2.erode(image_copy, structure)

    # finding the peaks
    height, width = image.shape[:2]
    projection = np.zeros(height)
    for i in range(height):
        projection[i] = np.average(image[i][:])
    smoothed_projection = gaussian_filter1d(projection, sigma=16)

    # need the outline of the image so that radial sweep doesn't get stuck
    # outlined image is passed into radial sweep as a parameter so that it only has to be calculated once
    outlined_image = extract_outline(image_copy)

    # Find local maxima
    peaks, _ = find_peaks(smoothed_projection)
    list_of_lines = []
    list_of_vertices = [[(0, 0), (0, width - 1)]]

    # performing the water drop technique with radial search for each peak
    for peak in peaks:
        col = width
        row = peak
        line_boundary = []
        while col > 0:
            # if next pixel white draw line and move on
            if image_copy[row, col - 1] != 0:
                image_copy[row, col - 1] = 127
                line_boundary.append((row, col-1))
                col -= 1
            # if hit a black pixel radial sweep
            elif image_copy[row, col - 1] == 0:
                row, col, boundary = radial_sweep(image_copy, outlined_image, row, col - 1, peak)
                for pixel in boundary:
                    line_boundary.append(pixel)
                for location in boundary:
                    image_copy[location[0], location[1]] = 127
        # locations are being appending from right to left, but I want them from left to right hence they are reversed
        reversed_boundary = line_boundary.copy()
        reversed_boundary.reverse()
        list_of_vertices.append(reversed_boundary)

    # adding the bottom corners
    list_of_vertices.append([(height - 1, 0), (height - 1, width - 1)])

    for i in range(len(list_of_vertices) - 1):
        # Create a blank white background image with the same dimensions as the source image
        white_background = np.ones_like(image) * 255

        mask = np.zeros_like(image)

        # the first list needs to be reversed, so it can be read by cv2.fillpoly
        reversed_list_of_vertices = list_of_vertices[i+1].copy()
        reversed_list_of_vertices.reverse()
        vertices = list_of_vertices[i] + reversed_list_of_vertices

        # each pair need to be reversed to be of form (col, row)
        vertices = [(t[1], t[0]) for t in vertices]

        # Convert the list of vertices to a NumPy array
        vertices_array = np.array([vertices], dtype=np.int32)

        # Fill the region defined by the vertices with white (255)
        cv2.fillPoly(mask, [vertices_array], 255)

        # Copy the non-rectangular region from the source image to the destination image using the mask
        white_background[mask == 255] = image[mask == 255]

        # if there is any black on the line append it to the list of locations of black,
        # by taking the first and last black locations we can crop the image (with some padding)
        black = []
        for y in range(height):
            if 0 in white_background[y, :]:
                black.append(y)
        # if there is no black in the image then the line also contains no words and so is not added to the list
        if len(black) > 0:
            if black[0] - 10 >= 0:
                min_y = black[0] - 10
            else:
                min_y = black[0]

            if black[-1] + 10 <= height:
                max_y = black[-1] + 10
            else:
                max_y = black[-1]

            line = white_background[min_y:max_y, :]

            # if the height of the line isn't large enough then there is no room for letters, so it is discarded
            if max_y - min_y > 80:
                list_of_lines.append(line)
    if write:
        # Create and save horizontal lines
        output_folder = 'horizontal_line'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        file_index = 0
        for i in range(len(list_of_lines)):

            base_filename = f'line_{i + 1}.jpg'
            output_path = os.path.join(output_folder, base_filename)

            # Check if the file already exists
            while os.path.exists(output_path):
                file_index += 1
                base_filename = f'line_{file_index + 1}.jpg'
                output_path = os.path.join(output_folder, base_filename)

            cv2.imwrite(output_path, list_of_lines[i])
    return list_of_lines


def word_splitter(image, output_folder=None, write=False, min_contour=1000, max_contour=np.inf):
    """
    splits an image of a line of handwritten text into words based on contours found from blobbing the words.
    it orders them based on the english left-to-right reading with contours starting with lower x values added first

    :param image: numpy.ndarray, image to split into words, should be greyscale but will be converted if not
    :param output_folder: string, the folder you are writing the words/images found to, default None as not writing
    :param write: bool, whether you are writing the split words to a folder, default is to return list of images
    :param min_contour: int, minimum size of contour to be considered
    :param max_contour: int, maximum size of contour to be considered
    :return word_image: list, list of image arrays of the segmented words, only returns if write=False
    """

    # making sure the input image is greyscale
    if len(image.shape) == 2:
        # Image is already grayscale
        gray_image = image.copy()
    else:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blob = increase_handwriting_size(gray_image, 9, 5)

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
                        if np.sum(word_image <= 10) > 250:
                            cv2.imwrite(output_path, word_image)
    else:
        word_image = []
        # Extract individual words and sort them based on the leftmost point
        for contour in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
            x, y, w, h = cv2.boundingRect(contour)

            # Filter contours that are in specified area range and append to the list of words
            if max_contour > cv2.contourArea(contour) > min_contour:
                # filtering out images that don't have enough black and are therefore just dots/noise
                if np.sum(image[y:y + h, x:x + w] <= 10) > 250:
                    word_image.append(image[y:y + h, x:x + w])

        return word_image


def extract_words_from_image(image_path, ruled):
    """ extract_words_from_image takes in the path to the image and then splits the image into lines
     and then splits those lines into words, it then adds them to a list along with some delimiting characters

    :param image_path: a string, path to a colour image that we want to extract the words from
    :param ruled: a bool, whether the paper has horizontal ruled lines
    :return ordered_words: a list, a list of images following standard english reading, 0=top left word ... x=bottom right word
    """
    image = preprocess_image(image_path, ruled)
    # splits the images into lines, image_lines is a list of those lines where image_lines[0] is the topmost line
    # image_lines[1] is the line below etc
    image_lines = droplet_line_splitter(image, False)

    ordered_words = []
    for i in range(len(image_lines)):
        # if line is array of size 0, it contains no words and is skipped as it breaks word_splitter
        if image_lines[i].size == 0:
            continue
        words = word_splitter(image_lines[i], min_contour=1250)
        for word in words:
            ordered_words.append(word)
        # append a symbol to indicate that a newline has started, except for the last non-empty line
        if i < len(image_lines) - 1:
            ordered_words.append(np.array([-999]))
    return ordered_words


def write_words_from_image(input_folder, output_folder):
    """ write_words_from_image uses extract_words_from_image and writes them to a folder

    :param input_folder: string, location of images you want to extract the words from
    :param output_folder: string, where you want to extracted words to be written to
    """
    folder_path = input_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
    else:
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpeg', '.jpg', '.JPG', '.png')):
                image_path = os.path.join(folder_path, filename)
                words = extract_words_from_image(image_path, True)
                for i in range(len(words)):
                    if np.all(words[i] == -999):
                        continue
                    else:
                        image = cv2.cvtColor(words[i], cv2.COLOR_GRAY2BGR)

                        output_word_filename = f"{filename}_word_{i}.png"  # Adjust the file extension accordingly
                        output_word_path = os.path.join(output_folder, output_word_filename)
                        cv2.imwrite(output_word_path, image)
            else:
                print(f"Ignoring non-image file: {filename}")


# import time
# start = time.time()
# write_words_from_image('labelled_images_school', 'words_to_label')
# end = time.time()
# print(end-start)

# split 82 images in 1680.3123326301575 seconds, ~20s per image
# split 97 images in 2180.1419427394867 seconds, ~24s per image
