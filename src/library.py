#----------------------------
# Author: Proyash Saha
#----------------------------

from collections import namedtuple
import numpy as np
import math

FILE_TYPE = "P2"  # to verify the file type
PGMFile = namedtuple('PGMFile', ['max_shade', 'data'])  # named tuple


# This function receives the name of a file, reads it in, verifies that
# the type is P2, and returns the corresponding PGMFile
def read_pgm(filename):
    rows = 0
    cols = 0
    max_shade = 0
    pixel_array = []
    line_no = 1

    try:
        f = open(filename)
    except:
        print(f"\nError: The file named \'{filename}\' does not exist!")
    else:
        try:
            for line in f:  # reading one line at a time from the file
                if line != "\n":  # checking for blank lines
                    end_index = line.find("\n")
                    line = line[0:end_index]
                    if line.find("#") != -1:  # checking for annoying cooments
                        end_index = line.find("#")
                        line = line[0:end_index]
                    line = line.strip()
                    if len(line) != 0:
                        if line_no == 1:  # checking for file type in line 1
                            if line == FILE_TYPE:
                                line_no += 1
                            else:
                                print("Error: The input file is not a P2 image!")
                        elif line_no == 2:  # getting the width and height of the image from line 2
                            dimensions = line.split()
                            rows = int(dimensions[1])  # rows = height
                            cols = int(dimensions[0])  # columns = width
                            line_no += 1
                        elif line_no == 3:  # getting the maximum shade value from line 3
                            max_shade = int(line)
                            line_no += 1
                        else:
                            line_array = line.split()  # storing all the numbers into a list after removing all the white spaces
                            for i in range(len(line_array)):
                                pixel_array.append(int(line_array[i]))
        except:
            print("\nError: The input file could not be read properly!")
        else:
            data = np.array(pixel_array).reshape(rows, cols)  # creating a 2D numpy array
            return PGMFile(max_shade, data)  # returning the corresponding  PGMFile


# This function receives  a  file  name  and  a  PGMFile,  and  creates  the  corresponding image file
def create_image(filename, pgm):
    with open(filename, "w") as f:
        print(f"{FILE_TYPE}\n{pgm.data.shape[1]} {pgm.data.shape[0]}\n{pgm.max_shade}\n", file=f)
        for row in pgm.data:
            for i in range(0, len(row)):
                print(str(row[i]), end=" ", file=f)
            print("", file=f)


# This function reflects a pgm image from left to right
def reflect_left_to_right(pgm_file):
    matrix = np.flip(pgm_file.data, axis=1)
    return PGMFile(pgm_file.max_shade, matrix)


# This function reflects a pgm image from top to bottom
def reflect_top_to_bottom(pgm_file):
    matrix = np.flip(pgm_file.data, axis=0)
    return PGMFile(pgm_file.max_shade, matrix)


# This function inverts the black and white pixels in a pgm image
def invert_black_white(pgm_file):
    matrix = np.subtract(pgm_file.max_shade, pgm_file.data)
    return PGMFile(pgm_file.max_shade, matrix)


# This function brightens a pgm image by 10%
def brighten(pgm_file, increase_by):
    brightness = int((increase_by/100) * (np.sum(pgm_file.data, dtype=np.uint64) / pgm_file.data.size))
    matrix = np.add(brightness, pgm_file.data)  # adding the brightness value to each pixel of the image
    matrix = np.clip(matrix, 0, pgm_file.max_shade)  # some pixels will be > 255, so bringing those values down to 255
    return PGMFile(pgm_file.max_shade, matrix)


# A function that receives a standard deviation σ and number of neighbors r, and returns the corresponding
# 1D dimensional Gaussian kernel of length 2r+ 1, normalized so that its entries sum to 1
def one_d_gaussian_kernel(sigma, r):
    size = (2*r)+1
    gaussian_kernel = []
    for i in range(size):
        x = i-r
        p_x = 1/(sigma*math.sqrt(2*math.pi)) * (math.pow(math.e, (-1/2)*(math.pow(x, 2)/math.pow(sigma, 2))))
        gaussian_kernel.append(p_x)
    gaussian_kernel = np.array(gaussian_kernel)
    gaussian_kernel = np.divide(gaussian_kernel, np.sum(gaussian_kernel))
    return gaussian_kernel


# A helper function to truncate and normalize the 1D Gaussian kernel
def truncate_normalize_1d_gaussian(kernel, left, right):
    highest_col_index = kernel.size-1
    new_kernel = np.copy(kernel)

    if left != 0:
        col_nums = [y for y in range(left)]  # storing the column numbers to be deleted from the left, in a list
        new_kernel = np.delete(new_kernel, col_nums)
        highest_col_index = new_kernel.size - 1
    if right != 0:
        col_nums = [highest_col_index-y for y in range(right)]  # storing the column numbers to be deleted from the right, in a list
        new_kernel = np.delete(new_kernel, col_nums)

    new_kernel = np.divide(new_kernel, np.sum(new_kernel))  # normalizing the kernel
    return new_kernel


def convolve_1dkernel_hrzntl(kernel, image_matrix):
    r = kernel.size // 2
    num_rows, num_cols = image_matrix.shape

    # traversing through the image matrix
    for row in range(num_rows):
        for col in range(num_cols):
            left = col  # num of cols to the left of current pixel
            right = (num_cols-1) - col  # num of cols to the right of current pixel

            trunc_left = 0
            trunc_right = 0
            if left < r:
                trunc_left = r - left  # num of cols to truncate from left of 1D Gaussian
            if right < r:
                trunc_right = r - right  # num of cols to truncate from left of 1D Gaussian
            new_kernel = truncate_normalize_1d_gaussian(kernel, trunc_left, trunc_right)

            curr_pixel_value = 0
            if left > r:
                for x in range(new_kernel.size):
                    curr_pixel_value += new_kernel[x] * image_matrix[row][x+(left-r)]
            else:
                for x in range(new_kernel.size):
                    curr_pixel_value += new_kernel[x] * image_matrix[row][x]

            image_matrix[row][col] = curr_pixel_value    # updating the current pixel value
    return image_matrix


# A function that convolves a 2D image with a 1D kernel twice in succession:  first horizontally, then vertically
def convolve_1dkernel(kernel, pgm_file):
    img_matrix = np.copy(pgm_file.data)
    img_matrix = convolve_1dkernel_hrzntl(kernel, img_matrix)   # convolving horizontally
    img_matrix = np.transpose(img_matrix)   # changing the orientation of the image
    img_matrix = convolve_1dkernel_hrzntl(kernel, img_matrix)   # convolving vertically
    img_matrix = np.transpose(img_matrix)   # changing the orientation of the image
    max_pixel = np.amax(img_matrix)
    return PGMFile(max_pixel, img_matrix)


# A helper function to build the gradient vector matrix
def get_gradient_vector(smooth_image):
    img_matrix = smooth_image.data
    num_rows, num_cols = img_matrix.shape
    gradient_vector = []
    cd_j = 0 # in j direction - horizontal
    cd_k = 0 # in k direction - vertical

    for row in range(num_rows):
        top = row
        bottom = (num_rows-1)-row
        cols = []
        for col in range(num_cols):
            left = col
            right = (num_cols-1)-col

            if left >= 1 and right >= 1:
                cd_j = (img_matrix[row][col+1] - img_matrix[row][col-1])/2
            elif left < 1 and right >= 1:
                cd_j = img_matrix[row][col+1] - img_matrix[row][col]
            elif left >= 1 and right < 1:
                cd_j = img_matrix[row][col] - img_matrix[row][col-1]

            if top >= 1 and bottom >= 1:
                cd_k = (img_matrix[row+1][col] - img_matrix[row-1][col])/2
            elif top < 1 and bottom >= 1:
                cd_k = img_matrix[row+1][col] - img_matrix[row][col]
            elif top >= 1 and bottom < 1:
                cd_k = img_matrix[row][col] - img_matrix[row-1][col]
            cols.append((cd_j, cd_k))
        gradient_vector.append(cols)

    return gradient_vector  # returns gradient vector matrix as a matrix of tuples


# a helper function to get the theta of the gradient vector
def get_angle(gradient_vector):
    result = 0
    angle = np.arctan2(gradient_vector[1], gradient_vector[0]) * 180 / np.pi
    if angle > 337.5 or angle <= 22.5 or (angle > 157.5 and angle <= 202.5):
        result = 0
    elif (angle > 22.5 and angle <= 67.5) or (angle > 202.5 and angle <= 247.5):
        result = 45
    elif (angle > 67.5 and angle <= 112.5) or (angle > 247.5 and angle <= 292.5):
        result = 90
    elif (angle > 112.5 and angle <= 157.5) or (angle > 292.5 and angle <= 337.5):
        result = 135
    return result


# A function to detect edges in the image
def edge_detection(pgm_file, gradient_vector_matrix):
    img_matrix = pgm_file.data
    num_rows, num_cols = img_matrix.shape
    temp_matrix = np.copy(pgm_file.data)

    for row in range(num_rows):
        for col in range(num_cols):
            gradient_vector = gradient_vector_matrix[row][col]
            mag_grad_vector = math.sqrt(math.pow(gradient_vector[0], 2) + math.pow(gradient_vector[1], 2))
            temp_matrix[row][col] = mag_grad_vector

    max_pixel = np.amax(temp_matrix)
    return PGMFile(max_pixel, temp_matrix)


# A function to thin the edges in the image
def edge_thinning(pgm_file, gradient_vector_matrix):
    img_matrix = pgm_file.data
    num_rows, num_cols = img_matrix.shape
    temp_matrix = np.copy(pgm_file.data)

    for row in range(num_rows):
        top = row
        bottom = (num_rows-1)-row
        for col in range(num_cols):
            left = col
            right = (num_cols-1)-col

            gradient_vector = gradient_vector_matrix[row][col]
            angle = get_angle(gradient_vector)
            curr_pixel = img_matrix[row][col]

            if angle == 0:
                if left >= 1 and right >= 1:
                    if curr_pixel < img_matrix[row][col-1] or curr_pixel < img_matrix[row][col+1]:
                        temp_matrix[row][col] = 0
                elif left < 1 and right >= 1:
                    if curr_pixel < img_matrix[row][col+1]:
                        temp_matrix[row][col] = 0
                elif left >= 1 and right < 1:
                    if curr_pixel < img_matrix[row][col-1]:
                        temp_matrix[row][col] = 0

            elif angle == 45:
                if left >= 1 and right >= 1 and top >= 1 and bottom >= 1:
                    if curr_pixel < img_matrix[row-1][col+1] or curr_pixel < img_matrix[row+1][col-1]:
                        temp_matrix[row][col] = 0
                elif left >= 1 and bottom >= 1:
                    if curr_pixel < img_matrix[row+1][col-1]:
                        temp_matrix[row][col] = 0
                elif right >= 1 and top >= 1:
                    if curr_pixel < img_matrix[row-1][col+1]:
                        temp_matrix[row][col] = 0

            elif angle == 90:
                if top >= 1 and bottom >= 1:
                    if curr_pixel < img_matrix[row-1][col] or curr_pixel < img_matrix[row+1][col]:
                        temp_matrix[row][col] = 0
                elif top < 1 and bottom >= 1:
                    if curr_pixel < img_matrix[row+1][col]:
                        temp_matrix[row][col] = 0
                elif top >= 1 and bottom < 1:
                    if curr_pixel < img_matrix[row-1][col]:
                        temp_matrix[row][col] = 0

            elif angle == 135:
                if left >= 1 and right >= 1 and top >= 1 and bottom >= 1:
                    if curr_pixel < img_matrix[row-1][col-1] or curr_pixel < img_matrix[row+1][col+1]:
                        temp_matrix[row][col] = 0
                elif left >= 1 and top >= 1:
                    if curr_pixel < img_matrix[row-1][col-1]:
                        temp_matrix[row][col] = 0
                elif right >= 1 and bottom >= 1:
                    if curr_pixel < img_matrix[row+1][col+1]:
                        temp_matrix[row][col] = 0
    max_pixel = np.amax(temp_matrix)
    return PGMFile(max_pixel, temp_matrix)


# A function tosuppress the noise in the image
def noise_suppress(pgm_file, low_thresh, high_thresh):
    img_matrix = pgm_file.data
    num_rows, num_cols = img_matrix.shape
    temp_matrix = np.copy(pgm_file.data)
    max_pixel = np.amax(img_matrix)
    low_thresh = low_thresh * max_pixel
    high_thresh = high_thresh * max_pixel

    for row in range(num_rows):
        top = row
        bottom = (num_rows-1)-row
        for col in range(num_cols):
            left = col
            right = (num_cols-1)-col
            curr_pixel = img_matrix[row][col]

            if left >= 1 and right >= 1 and top >= 1 and bottom >= 1:
                if curr_pixel < low_thresh:
                    temp_matrix[row][col] = 0
                elif low_thresh <= curr_pixel < high_thresh:
                    if img_matrix[row][col-1] <= high_thresh:
                        if img_matrix[row-1][col-1] <= high_thresh:
                            if img_matrix[row-1][col] <= high_thresh:
                                if img_matrix[row-1][col+1] <= high_thresh:
                                    if img_matrix[row][col+1] <= high_thresh:
                                        if img_matrix[row+1][col+1] <= high_thresh:
                                            if img_matrix[row+1][col] <= high_thresh:
                                                if img_matrix[row+1][col-1] <= high_thresh:
                                                    temp_matrix[row][col] = 0
    max_pixel = np.amax(temp_matrix)
    return PGMFile(max_pixel, temp_matrix)
