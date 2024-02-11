import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import image_processing
from image_processing import increase_handwriting_size, preprocess_image


def mark_black_pixels_above_row(image, row_number, resolution=10, colour=127):
    """
    Mark black pixels above the specified row with red color.

    Args:
    - image: The input image.
    - row_number: The row number from which to start traveling upward.
    - colour: The colour value to mark the pixels (tuple of BGR values).
    - resolution: how often to sample the points for vertices default is sampling every 10

    Returns:
    - The image with black pixels above the specified row marked in red.
    - The vertices of the polygon formed by marking the black pixels.
    """
    # Initialize variables
    height, width = image.shape[:2]
    up_y = row_number
    down_y = row_number
    vertices = []

    # need to find a way for it to include the descenders, on the bottom

    # Iterate over columns in the current row
    for x in range(width):
        # Iterate upward from the specified row up to 125 pixels
        for y in range(row_number, row_number - 125, -1):
            if all(image[y, x] == [0, 0, 0]):  # If pixel value is black
                up_y = y
                break  # Stop iterating over rows once a non-black pixel is found
        # Iterate downward from the specified row up to 125 pixels
        for y in range(row_number, row_number + 125, 1):
            if all(image[y, x] == [0, 0, 0]):  # If pixel value is black
                down_y = y
                break  # Stop iterating over rows once a non-black pixel is found

        # Compute average y-coordinate
        avg_y = (up_y + down_y) // 2
        image[avg_y][x] = colour  # Change pixel color

        # Append vertex (x, avg_y) to the list of vertices
        vertices.append((x, avg_y))

    # decreasing the "resolution" of the polygon to improve performance
    new_vertices = []

    for i in range(0, len(vertices), resolution):
        new_vertices.append(vertices[i])

    # each vertex is unique
    if new_vertices[-1] != vertices[-1]:
        new_vertices.append(vertices[-1])

    return image, new_vertices


def curve_line_split(image):
    image_copy = image.copy()
    height, width = image_copy.shape[:2]
    projection = np.zeros(height)
    for i in range(height):
        projection[i] = np.average(image_copy[i][:])

    smoothed_projection = gaussian_filter1d(projection, sigma=16)

    # Find local maxima
    peaks, _ = find_peaks(smoothed_projection)

    list_of_lines = []

    list_of_vertices = [[(0, 0), (width - 1, 0)]]

    for row in peaks:
        _, vertices = mark_black_pixels_above_row(image, row)
        list_of_vertices.append(vertices)
    list_of_vertices.append([(height - 1, height - 1), (0, height - 1)])

    for i in range(len(list_of_vertices) - 1):
        # Create a blank white background image with the same dimensions as the source image
        white_background = np.ones_like(image_copy) * 255

        mask = np.zeros_like(image_copy)

        # the first list needs to be reversed so it can be read by fillpoly in ccw direction
        reversed_list_of_vertices = list_of_vertices[i].copy()
        reversed_list_of_vertices.reverse()
        vertices = reversed_list_of_vertices + list_of_vertices[i+1]

        # Convert the list of vertices to a NumPy array
        vertices_array = np.array([vertices], dtype=np.int32)

        # Fill the region defined by the vertices with white (255)
        cv2.fillPoly(mask, [vertices_array], 255)

        # Copy the non-rectangular region from the source image to the destination image using the mask
        white_background[mask == 255] = image_copy[mask == 255]
        # white_background = cv2.cvtColor(white_background, cv2.COLOR_BGR2GRAY)

        # Split the image into its color channels
        line, _, _ = cv2.split(white_background)
        height, width = line.shape[:2]

        # if there is any black on the line append it to the list of locations of black,
        # by taking the first and last black locations we can crop the image (with some padding)
        black = []
        for y in range(height):
            if 0 in line[y, :]:
                black.append(y)

        if black[0] - 15 >= 0:
            min_y = black[0] - 15
        else:
            min_y = black[0]

        if black[-1] + 15 <= height:
            max_y = black[-1] + 15
        else:
            max_y = black[-1]

        line = line[min_y:max_y, :]

        # if the height of the line isn't large enough then there is no room for letters, so it is discarded
        if max_y - min_y > 80:
            list_of_lines.append(line)
    return list_of_lines


# image = preprocess_image('deprecated/uni_sample_handwriting/IMG_3504.JPG', True)
# image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# lines = image_processing.line_splitter(image)
# curved_lines = curve_line_split(image)


# new plan need to see where words are and where they fit within the lines
# increase handwriting size of everything, everything between the lines is "tagged" as line x
# then for each blob check where the majority of its pixels lie, they lie in line y, "tag" all pixels as line y


def blob_line_split(input_image):
    # might need to draw white lines over where peaks are detected after blobbing
    image_copy = input_image.copy()
    height, width = image_copy.shape[:2]
    projection = np.zeros(height)
    for i in range(height):
        projection[i] = np.average(image_copy[i][:])

    smoothed_projection = gaussian_filter1d(projection, sigma=16)

    # Find local maxima
    peaks = find_peaks(smoothed_projection)[0]

    # inserting a "peak" at the top of the image
    peaks = np.insert(peaks, 0, 0)
    # inserting a "peak" at the bottom of the image
    peaks = np.append(peaks, height - 1)
    regions = []

    image_mask = np.zeros_like(image_copy)

    for i in range(len(peaks) - 1):
        image_mask[peaks[i]:peaks[i + 1], :] = 255 // (len(peaks) - 1) * i
        regions.append(255 // (len(peaks) - 1) * i)

    blobbed = increase_handwriting_size(image_copy, 5, 2)

    for peak in peaks:
        blobbed[peak, :] = 255

    contours = cv2.findContours(blobbed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for contour in contours:
        # Step 3: Calculate centroid of each blob
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid_y = int(M["m01"] / M["m00"])
        else:
            continue  # Skip if centroid cannot be calculated
        for i in range(len(peaks) - 1):
            if peaks[i] < centroid_y < peaks[i+1]:
                cv2.drawContours(image_mask, [contour], -1, color=regions[i], thickness=cv2.FILLED)

    lines = []
    # Iterate through each unique color in the layer mask
    for color in regions:
        # Create a new blank image to copy the regions
        new_image = np.ones_like(image_copy) * 255

        # Copy regions from the original images to the new image based on the mask
        new_image[image_mask == color] = image_copy[image_mask == color]
        lines.append(new_image)

    return blobbed


import time
start = time.time()
image = preprocess_image('deprecated/uni_sample_handwriting/IMG_3517.JPG', True)
image2 = blob_line_split(image)
stop = time.time()
print(stop-start)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[1].imshow(image2, cmap='gray')
plt.show()

# for line in image2:
#     plt.imshow(line, cmap='gray')
#     plt.show()


# https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=23a96b0bcde6434e7b02539207a3157fa6467fc5
# use this instead
