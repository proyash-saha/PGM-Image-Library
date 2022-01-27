#----------------------------
# Author: Proyash Saha
#----------------------------

import library
import sys

# Main body
if __name__ == "__main__":

    SIGMA = 1.5
    R = 3   # number of neighbours
    BRGHTNESS_INCREASE = 10 # 10%
    LOW_THRESHOLD = 0.10  # 10%
    HIGH_THRESHOLD = 0.20   # 20%

    try:
        filename = sys.argv[1]  # getting the name of the input file from command line
    except:
        print(f"\nError: Please enter the name of a pgm file as an argument.\n")
    else:
        pgmlib = library
        
        # reading the pgm file
        BASE_NAME = filename[0:filename.find(".pgm")]
        pgm_file = pgmlib.read_pgm(filename)

        print("\nStarting image processing...\n")

        # reflecting the image from left to right
        pgm = pgmlib.reflect_left_to_right(pgm_file)
        pgmlib.create_image(BASE_NAME+"-reflected-left-to-right.pgm", pgm)
        print("Reflected Left to Right")

        # reflecting the image from top to bottom
        pgm = pgmlib.reflect_top_to_bottom(pgm_file)
        pgmlib.create_image(BASE_NAME+"-reflected-top-to-bottom.pgm", pgm)
        print("Reflected Top to bottom")

        # inverting the black and white colours of the image
        pgm = pgmlib.invert_black_white(pgm_file)
        pgmlib.create_image(BASE_NAME+"-inverted-black-and-white.pgm", pgm)
        print("Inverted Black and White")

        # increasing the brightness of image by the BRGHTNESS_INCREASE value
        pgm = pgmlib.brighten(pgm_file, BRGHTNESS_INCREASE)
        pgmlib.create_image(BASE_NAME+"-brightened.pgm", pgm)
        print(f"Brightened by {BRGHTNESS_INCREASE}%")

        # -------------------------------------------------------------------------------------------
        # Note: The following operations must be done according to the order they are in.
        # -------------------------------------------------------------------------------------------

        one_d_kernel = pgmlib.one_d_gaussian_kernel(SIGMA, R)

        # smoothening the image
        smooth_image = pgmlib.convolve_1dkernel(one_d_kernel, pgm_file)
        pgmlib.create_image(BASE_NAME+"-smooth.pgm", smooth_image)
        print("Smoothened")

        # constructing the gradient vector matrix
        gradient_vector_matrix = pgmlib.get_gradient_vector(smooth_image)

        # detecting edges in the image
        edge_detect = pgmlib.edge_detection(smooth_image, gradient_vector_matrix)
        pgmlib.create_image(BASE_NAME+"-edge.pgm", edge_detect)
        print("Edge detected")

        # thinning the edges in the image
        edge_thin = pgmlib.edge_thinning(edge_detect, gradient_vector_matrix)
        pgmlib.create_image(BASE_NAME+"-thin-edge.pgm", edge_thin)
        print("Made edges thinner")

        # suppressing noise in the image
        noise_supp = pgmlib.noise_suppress(edge_thin, LOW_THRESHOLD, HIGH_THRESHOLD)
        pgmlib.create_image(BASE_NAME+"-noise-suppressed.pgm", noise_supp)
        print("Noise Suppressed")

        print("\nFinished image processing\n")
