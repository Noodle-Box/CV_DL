# Header Libraries #
from warnings import filters
from skimage import data, color, feature, filters, io # for inbuilt images
from skimage.color import *
from skimage.feature import corner_harris, corner_peaks, corner_subpix
from skimage.filters import threshold_otsu
import numpy as np
from skimage.morphology import dilation, erosion, disk, square, diamond, footprint_rectangle # for morphological operators and masks
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

####################### Function Definitions for this Lab ##########################
def point_operator(image, amount):
    """
    Applies a point operator to the image by adding a specified amount to each pixel value.
    
    Parameters:
    image: The input grayscale image (2D numpy array).
    amount: An integer specifying the amount to add to each pixel value. 
            Positive values will brighten the image, while negative values will darken it.
    """
    # Add the specified amount to each pixel value and clip the result to be within [0, 1]
    adjusted_image = np.clip(image + amount, 0, 1)
    return adjusted_image

#####################################################################################

##### Task 1: Compute a Histogram of an image and adjust Point Operators with histogram comparisons #####

# Load image and convert to grayscale
task1_image = data.chelsea()
task1_gray_image = color.rgb2gray(task1_image)

# Compute histogram

# 1. Divide range of possible pixel values into 256 bins. 80bit image has 256 possible grayscale values 
# 2. Finding exact middle point of each bin: Slice takes everything from start of array up to (not including) last item (Left edge).
        # Then, skip the first item and take everything else to end (Right edges)
        # Finally, add left and right edfe and divide by 2, that is exact cetner point of bin
histogram, bin_edges = np.histogram(task1_gray_image, bins=256, range=(0, 1))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Create a figure with two subplots
#fig, axarr = plt.subplots(1, 2, figsize=(12, 5))

# # Display the grayscale 'astronaut' image
# axarr[0].imshow(task1_gray_image, cmap='gray')
# axarr[0].set_title('Grayscale Astronaut Image')
# axarr[0].axis('off') # Turn off axes for image display
# axarr[1].plot(bin_centers, histogram, color='red') # Plot histogram with bin centers on x-axis and frequency on y-axis
# axarr[1].set_title('Histogram of Grayscale Astronaut Image')
# axarr[1].set_xlabel('Pixel Intensity')
# axarr[1].set_ylabel('Frequency')
# # Adjust layout and display the figure
# plt.tight_layout()
# plt.show()

# Next step, use point operator to adjust image brightness and compare historgrams
# Normalize points from 0-255 to 0-1 scale

Points = -100 #Modify this value!!

normalized_points = Points / 255.0
adjusted_image = point_operator(task1_gray_image, normalized_points)

# Compute histogram of adjusted image and also compute bin centers
adjusted_histogram, adjusted_bin_edges = np.histogram(adjusted_image, bins=256, range=(0, 1))
adjusted_bin_centers = (adjusted_bin_edges[:-1] + adjusted_bin_edges[1:]) / 2

# Create a figure with two subplots
#fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
# # Display the darkened image and histogram
# axarr[0].imshow(adjusted_image, cmap='gray')
# axarr[0].set_title(f'Adjusted Grayscale Astronaut Image (Darkened)\nwith Point Operator: {Points}')
# axarr[0].axis('off') # Turn off axes for image display
# axarr[1].plot(adjusted_bin_centers, adjusted_histogram, color='green') # Plot histogram with bin centers on x-axis and frequency on y-axis
# axarr[1].set_title(f'Histogram of Adjusted Grayscale Astronaut Image\n with Point Operator: {Points}')
# axarr[1].set_xlabel('Pixel Intensity')
# axarr[1].set_ylabel('Frequency')
# # Adjust layout and display the figure
# plt.tight_layout()
# plt.show()



####### Task 2: Use the Harris detector to find edges, corners and featuers ####

# Load image and convert to grayscale as harris detector
Image_gray = data.checkerboard()

# Apply the Harris corner detection to the grayscale image

# Method = k (standard formula)
# k = 0.04 - 0.06 (empirically determined constant)
# sigma = 1 sigma for Gaussian blurring window

Image_harris = corner_harris(Image_gray, method='k', k=1, sigma=1)


# Now, use the Corner Peaks function to do Non-Maximum Suppression (finds absolute highlest local peaks and ignores surrounding pixels)
# It also returns the coordinates of the detected corners in the image

Image_corners = corner_peaks(Image_harris, min_distance=5)

# Finally, use the Corner Subpixel to refine the corner locations to subpixel accuracy
Image_subpixels = corner_subpix(Image_gray, Image_corners, window_size=11)


fig, axarr = plt.subplots(1, 3, figsize=(20, 5))
# TODO: to show the figures including 1. original image; 2. gray scale image; 3. image with the detected corners
axarr[0].imshow(Image_gray, cmap='gray')
axarr[0].set_title('Original Image')
axarr[1].imshow(Image_gray, cmap='gray')
axarr[1].set_title('Grayscale Image')
axarr[2].imshow(Image_gray, cmap='gray')
axarr[2].plot(Image_subpixels[:, 1], Image_subpixels[:, 0], '+r', markersize=15)
axarr[2].set_title('Detected Corners')
plt.show()
