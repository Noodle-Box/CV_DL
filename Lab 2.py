# Header Libraries #
from warnings import filters

from skimage import data, color, feature, filters, io # for inbuilt images
from skimage.color import *
from skimage.filters import threshold_otsu
import numpy as np
from skimage.morphology import dilation, erosion, disk, square, diamond, footprint_rectangle # for morphological operators and masks
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

##### Functions for Task 2 (Blurring Methods) ####
# 2. Mean Box Filter Method
def apply_mean_box_filter(image, filter_size):
    """
    Applies a Mean Box Filter to the image.
    
    Parameters:
    image: The input grayscale image (2D numpy array).
    filter_size: An integer specifying the size of the box 
                 (e.g., 3 for a 3x3 filter, 5 for a 5x5 filter).
    """
    # The uniform_filter in scipy calculates the multidimensional box filter
    blurred_image = uniform_filter(image, size=filter_size)
    return blurred_image

# 3. Gaussian Smoothing Filter Method
def apply_gaussian_filter(image, sigma_value):
    """
    Applies a Gaussian Smoothing Filter to the image.
    
    Parameters:
    image: The input grayscale image (2D numpy array).
    sigma_value: A float specifying the standard deviation (sigma). 
                 Higher values mean more blur.
    """
    # skimage's built-in gaussian filter
    blurred_image = filters.gaussian(image, sigma=sigma_value)
    return blurred_image


def canny_edge_detection(image, sigma_value):
    """
    Applies Canny Edge Detection to the image.
    
    Parameters:
    image: The input grayscale image (2D numpy array).
    sigma_value: A float specifying the standard deviation (sigma) for the Gaussian filter 
                 used in the Canny edge detection. Higher values mean more blur before edge detection.
    """
    edges = feature.canny(image, sigma=sigma_value)
    return edges

########################################################################################################


## TASK 1: Read the coin image and perform dilation and erosion with different masks ##
inputImage = data.coins() # coin is a standard image

# Convert RGB to gray scale only when needed
if inputImage.ndim == 3 and inputImage.shape[-1] in (3, 4):
    inputImage = color.rgb2gray(inputImage)

def make_square_se(n):
    if n < 1 or n % 2 == 0:
        raise ValueError('n must be odd and >= 1')
    return np.ones((n, n), dtype=bool)


def make_diamond_se(n):
    if n < 1 or n % 2 == 0:
        raise ValueError('n must be odd and >= 1')
    return diamond(n)


def make_cross_se(n):
    if n < 1 or n % 2 == 0:
        raise ValueError('n must be odd and >= 1')
    se = np.zeros((n, n), dtype=bool)
    center = n // 2
    se[center, :] = True
    se[:, center] = True
    return se


def make_se(n, se_type='square'):
    se_type = se_type.lower()
    if se_type == 'square':
        return make_square_se(n)
    if se_type == 'diamond':
        return make_diamond_se(n)
    if se_type == 'cross':
        return make_cross_se(n)
    raise ValueError("se_type must be 'square', 'cross', or 'diamond'")


# choose your structuring elements easily
se_custom_dilate = make_se(5, 'square')
se_custom_erode = make_se(5, 'square')

# Dilate the image 
inputImage1 = dilation(inputImage, se_custom_dilate)

# Erode the image
inputImage2 = erosion(inputImage, se_custom_erode)

# fig, axes = plt.subplots(1, ncols=3, figsize=(16, 8))
# axes[0].imshow(inputImage, cmap='gray')
# axes[0].axis('off')
# axes[0].set_title('original grayscale')
# axes[1].imshow(inputImage1, cmap='gray')
# axes[1].axis('off')
# axes[1].set_title('After dilation')
# axes[2].imshow(inputImage2, cmap='gray')
# axes[2].axis('off')
# axes[2].set_title('After erosion')
# plt.show()


## TASK 2: Perform image segmentation on an original face image and a blurred image ##

# Load the image
task2_image = io.imread('TheSquad.jpg')


# Optional: Blurring Methods (uncomment when needed)
# Apply a 5x5 Mean Box Filter
#box_blurred = apply_mean_box_filter(task2_image_gray, filter_size=5)

# Apply a Gaussian Filter with sigma value of 2.0
#gaussian_blurred = apply_gaussian_filter(task2_image, sigma_value=5.0)



# Method 1: Global Thresholding using Otsu's method

task2_image_rgb = color.rgb2gray(task2_image) # Convert RGB image into gray scale
otsu_threshold = threshold_otsu(task2_image_rgb) # Otsu's thresholding

# Binary mask where pixels brighter than threshold are True (1) and others are False (0)
global_mask = task2_image_rgb > otsu_threshold



# Method 2 - Color based segmentation using HSV

# Convert the array to HSV
hsv = color.rgb2hsv(task2_image)

#Extract Hue and Saturation channels
H = hsv[..., 0] # Hue channel
S = hsv[..., 1] # Saturation channel
V = hsv[..., 2] # Value channel (may not be needed always for HSV)

# Define thresholds
# For Hues: red orange and yellow (near 0.0 or 1.0), so 0.1 > H < 0.9
# For saturation, avoid pur white and highly satured red objects (0.2 or 0.7) so 0.2 < S < 0.7
skin_mask = (H < 0.1) | (H > 0.9) & (S > 0.2) & (S < 0.7)



## INSERT THRESHOLDED IMAGE INTO EDGE DETECTOR ##
# Cusstomize sigma value for Gaussian Blurring (if implemented, remove prior blurring)

edge_mask = canny_edge_detection(global_mask, sigma_value = 9.0)


# Plotting both results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(task2_image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(global_mask, cmap='gray')
axes[1].set_title('Otsu Thresholding (Skin Mask)')
axes[1].axis('off')

axes[2].imshow(edge_mask, cmap='gray')
axes[2].set_title('Canny Edge Detection, \n sigma=9.0')
axes[2].axis('off')

plt.tight_layout()
plt.show()


