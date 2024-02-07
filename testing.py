import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import image_processing
from image_processing import increase_handwriting_size, preprocess_image


def apply_canny_edge_detection(input_image_path, lower_threshold=95, upper_threshold=150):
    # Read the input image
    original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # Denoising using non-local means denoising
    denoised = cv2.fastNlMeansDenoising(original_image, None, h=5, templateWindowSize=10, searchWindowSize=25)
    blurred_image = cv2.GaussianBlur(denoised, (7, 7), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, lower_threshold, upper_threshold)
    inv_edge = cv2.bitwise_not(edges)

    eroded = cv2.erode(inv_edge, (7, 7), iterations=2)
    eroded = increase_handwriting_size(eroded, 2, 2)

    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the original and the edges side by side using Matplotlib
    # edges = cv2.bitwise_not(edges)
    # axes[0].imshow(edges, cmap='gray')
    # axes[0].set_title('Edges')

    # eroded = cv2.bitwise_not(eroded)
    # axes[1].imshow(eroded, cmap='gray')
    # axes[1].set_title('Eroded')
    # plt.show()

    # Apply closing to the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closing = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)

    words = increase_handwriting_size(inv_edge, 3, 1)

    # Use HoughLinesP to detect lines in the image
    lines = cv2.HoughLinesP(closing, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    closing_copy = np.copy(closing)
    # Draw the detected lines on a black canvas
    line_mask = np.zeros_like(closing)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(closing_copy, (x1, y1), (x2, y2), (0, 0, 0), 2)

    # Invert the line mask
    line_mask_inv = cv2.bitwise_not(line_mask)

    # Combine the original image with the inverted line mask
    hor_lines_removed = cv2.bitwise_and(closing_copy, line_mask_inv)

    # median blurring is a good final step to remove the horizontal lines however it doesnt work as the letters havent
    # been filled in properly
    median = cv2.medianBlur(hor_lines_removed, 5)

    preprocessed = preprocess_image(input_image_path, True)

    # fig, axes = plt.subplots(1, 5, figsize=(10, 5))
    #
    # # Display the original and the edges side by side using Matplotlib
    # axes[0].imshow(original_image, cmap='gray')
    # axes[0].set_title('Original Image')
    #
    # edges = cv2.bitwise_not(edges)
    # axes[1].imshow(edges, cmap='gray')
    # axes[1].set_title('Edges')
    #
    # closing = cv2.bitwise_not(closing)
    # axes[2].imshow(closing, cmap='gray')
    # axes[2].set_title('Closed Edges')
    #
    # median = cv2.bitwise_not(median)
    # axes[3].imshow(median, cmap='gray')
    # axes[3].set_title('Horizontal Lines Removed')
    #
    # # # black = find_black_regions(closing)
    # # black = fill_small_black_regions(median, 100)
    # axes[4].imshow(preprocessed, cmap='gray')
    # axes[4].set_title('Preprocessed')
    # plt.show()
    #
    # plt.imshow(edges, cmap='gray')
    # plt.show()
    # plt.imshow(preprocessed, cmap='gray')
    # plt.show()

    crop_edges = edges[:, 5:-5]
    crop_edges_inv = cv2.bitwise_not(crop_edges)
    # preprocessed_inv = cv2.bitwise_not(preprocessed)
    mask = words[:, 5:-5]
    master = cv2.bitwise_or(crop_edges_inv, preprocessed, mask=mask)

    plt.imshow(crop_edges_inv, cmap='gray')
    plt.show()
    plt.imshow(preprocessed, cmap='gray')
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the original and the edges side by side using Matplotlib
    axes[0].imshow(preprocessed, cmap='gray')
    axes[0].set_title('Preprocessed')

    axes[1].imshow(master, cmap='gray')
    axes[1].set_title('Processed or Edges')
    plt.show()


def edge_detection(image):
    # if len(image.shape) == 3:  # Check if the image has three channels (e.g., BGR)
    #     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # else:
    #     image_gray = image.copy()  # If already grayscale, no need to convert

    # converting to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    cl = clahe.apply(l_channel)
    # merge the CLAHE enhanced L-channel with the a and b channel
    limage = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_image = cv2.cvtColor(limage, cv2.COLOR_LAB2BGR)
    # # Stacking the original image with the enhanced image
    # result = np.hstack((image, enhanced_image))
    # plt.imshow(result, cmap='gray')
    # plt.show()
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to create a binary image
    _, binary_image = cv2.threshold(enhanced_image, 124, 255, cv2.THRESH_BINARY)
    binary_image = image_processing.hor_line_removal(binary_image)
    length, width = binary_image.shape
    result = binary_image.copy()
    for i in range(1, length):
        for j in range(1, width):
            # If the current pixel is black and either the pixel above or the pixel to the left is white,
            # keep the pixel white. Otherwise, set it to black.
            if binary_image[i][j] == 0 and (binary_image[i-1][j] == 255 or binary_image[i][j-1] == 255):
                result[i][j] = 0
            else:
                result[i][j] = 255
    # # this loop gets a crosshatch pattern because it is updating the array while checking the array
    # for i in range(1, length):
    #     for j in range(1, width):
    #         # If the current pixel is black and either the pixel above or the pixel to the left is white,
    #         # keep the pixel white. Otherwise, set it to black.
    #         if binary_image[i][j] == 0 and (binary_image[i-1][j] == 255 or binary_image[i][j-1] == 255):
    #             binary_image[i][j] = 0
    #         else:
    #             binary_image[i][j] = 255
    return result

# # Example usage:
# input_image_path = 'deprecated/uni_sample_handwriting/IMG_3524.JPG'
#
#
# # apply_canny_edge_detection(input_image_path)
#
# image = cv2.imread('deprecated/uni_sample_handwriting/IMG_3524.JPG')
# image2 = edge_detection(image)
#
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#
# # Display the original and the edges side by side using Matplotlib
# axes[0].imshow(image, cmap='gray')
# axes[0].set_title('original')
#
# axes[1].imshow(image2, cmap='gray')
# axes[1].set_title('edge')
# plt.show()


def mark_black_pixels_above_row(image, row_number, colour):
    """
    Mark black pixels above the specified row with red color.

    Args:
    - image: The input image.
    - row_number: The row number from which to start traveling upward.

    Returns:
    - The image with black pixels above the specified row marked in red.
    """
    # Iterate over columns in the current row
    length, width = image.shape[:2]
    up_y = row_number
    down_y = row_number
    # initialising the vertices with the top row of the image
    # top left, top right
    vertices = [(0, 0), (width - 1, 0)]
    for x in range(width):
        # Iterate upward from the specified row up to 125 pixels
        for y in range(row_number, row_number-125, -1):
            # If the pixel value is black (0), mark it as red
            if all(image[y, x] == [0, 0, 0]):
                up_y = y
                # image[y][x] = (255, 0, 0)  # Change pixel color to red
                break  # Stop iterating over rows once a non-black pixel is found
            if y == row_number-124:
                up_y = y
                # image[y][x] = (255, 0, 0)
        # Iterate downward from the specified row up to 125 pixels
        for y in range(row_number, row_number+125, 1):
            # If the pixel value is black (0), mark it as red
            if all(image[y, x] == [0, 0, 0]):
                down_y = y
                # image[y][x] = (255, 0, 0)  # Change pixel color to red
                break  # Stop iterating over rows once a non-black pixel is found
            if y == row_number+124:
                down_y = y
                # image[y][x] = (255, 0, 0)

        avg_y = (up_y + down_y) // 2
        image[avg_y][x] = (colour, 0, 0)

        # vertices.append((avg_y, x))
        # Append (x, avg_y) instead of (avg_y, x)
        vertices.append((x, avg_y))
    # appending the last vertices corresponding to the last row of the image
    # bottom left
    vertices.append((0, length - 1))
    # bottom right
    vertices.append((width - 1, length - 1))
    return image, vertices


def curve_line_split(image):
    image_copy = image.copy()
    height, width = image_copy.shape[:2]
    projection = np.zeros(height)
    for i in range(height):
        projection[i] = np.average(image_copy[i][:])

    smoothed_projection = gaussian_filter1d(projection, sigma=16)

    # Find local maxima
    peaks, _ = find_peaks(smoothed_projection)

    # # draw peaks on image
    # image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
    # for row_number in peaks:
    #     cv2.line(image_copy, (0, row_number), (image_copy.shape[1], row_number), (0, 255, 0), 3)
    # plt.imshow(image_copy)
    # plt.show()

    colour_increment = 128 // len(peaks)
    colour = 127
    list_of_vertices = []
    for row in peaks:
        image, vertices = mark_black_pixels_above_row(image, row, colour)
        # cv2.line(image, (0, row), (image.shape[1], row), (0, 255, 0), 1)
        list_of_vertices.append(vertices)
        colour += colour_increment

    for i in range(len(list_of_vertices) - 1):
        # Create a blank destination image with the same dimensions as the source image
        destination_image = np.zeros_like(image_copy)
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
        destination_image[mask == 255] = image_copy[mask == 255]
        plt.imshow(destination_image)
        plt.show()
        breakpoint2 = 1
    cv2.imwrite('curve_line_split_test.jpg', image)
    # image = mark_black_pixels_above_row(image, 500)
    plt.imshow(image)
    plt.show()



image = preprocess_image('deprecated/uni_sample_handwriting/IMG_3524.JPG', True)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
curve_line_split(image)
