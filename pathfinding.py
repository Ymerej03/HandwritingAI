import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import image_processing
from image_processing import preprocess_image


def extract_outline(image):
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
                    if 0 <= r_neighbor < height and 0 <= c_neighbor < width and image[r_neighbor, c_neighbor] != 0:
                        # Copy the black pixel to the result image
                        result_image[row, col] = 0
                        break  # Stop checking other directions if a white neighbor is found

    return result_image


def radial_sweep(image, outlined_image, start_row, start_col, peak):
    height, width = image.shape[:2]
    max_iterations = 150
    max_distance = 150

    ccw_iterations = 0
    ccw_image = outlined_image.copy()
    ccw_directions = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (9, 9)]
    ccw_boundary = []
    # ccw = 1 is success, 2 is failure
    ccw = 0

    cw_iterations = 0
    cw_image = outlined_image.copy()
    cw_directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (9, 9)]
    cw_boundary = []
    # cw = 1 is success, 2 is failure
    cw = 0

    # Start from the given row and column
    row, col = start_row, start_col
    ccw_row, ccw_col = start_row, start_col

    # mark starting pixel as grey
    ccw_image[start_row, start_col] = 100
    cw_image[start_row, start_col] = 127

    cw_boundary.append((start_row, start_col))
    ccw_boundary.append((start_row, start_col))

    # Radial search in the clockwise direction
    while True:
        for dr, dc in cw_directions:
            # if iterated through all directions and havent found anything must have failed/got stuck on isolated pixel
            if dr == 9:
                cw = 2
                break
            r_new, c_new = row + dr, col + dc  # Move in the radial direction
            if 0 <= r_new < height and 0 <= c_new < width and cw_image[r_new, c_new] == 0:
                # Move to the next black pixel
                row, col = r_new, c_new
                cw_image[row, col] = 127  # Mark the pixel as gray
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
            # if takes too many iterations has failed
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
            # if iterated through all directions and havent found anything must have failed/got stuck on isolated pixel
            if dr == 9:
                ccw = 2
                break
            ccw_r_new, ccw_c_new = ccw_row + dr, ccw_col + dc  # Move in the radial direction
            if 0 <= ccw_r_new < height and 0 <= ccw_c_new < width and ccw_image[ccw_r_new, ccw_c_new] == 0:
                # Move to the next black pixel
                ccw_row, ccw_col = ccw_r_new, ccw_c_new
                ccw_image[ccw_row, ccw_col] = 100  # Mark the pixel as gray
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
            # if takes too many iterations has failed
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
        print("Error")


def droplet_line_splitter(image):
    output_boundary = []
    height, width = image.shape[:2]
    projection = np.zeros(height)
    for i in range(height):
        projection[i] = np.average(image[i][:])

    smoothed_projection = gaussian_filter1d(projection, sigma=16)

    # outlined image takes a while, would probably do from passing the outlined image in instead of calculating it everytime
    outlined_image = extract_outline(image)

    # Find local maxima
    peaks, _ = find_peaks(smoothed_projection)
    # peaks = [1022]
    for peak in peaks:
        col = width
        row = peak
        line_boundary = []
        while col > 0:
            # if next pixel white draw line and move on
            if image[row, col - 1] != 0:
                image[row, col - 1] = 127
                line_boundary.append((row, col-1))
                col -= 1
            elif image[row, col - 1] == 0:
                # if hit a black pixel radial sweep
                row, col, boundary = radial_sweep(image, outlined_image, row, col - 1, peak)
                for pixel in boundary:
                    line_boundary.append(pixel)
                for location in boundary:
                    image[location[0], location[1]] = 127
        output_boundary.append(line_boundary)
    return image, output_boundary


image = preprocess_image('deprecated/uni_sample_handwriting/IMG_3514.JPG', True)
# plt.imshow(image)
# plt.show()

structure = cv2.getStructuringElement(cv2.MORPH_RECT, [5, 5])
image = cv2.erode(image, structure)
image = cv2.erode(image, structure)
# plt.imshow(image)
# plt.show()

image, boundary = droplet_line_splitter(image)
white = np.ones_like(image) * 255
for i in range(len(boundary)):
    for location in boundary[i]:
        white[location[0], location[1]] = 0
plt.imshow(white)
plt.show()

