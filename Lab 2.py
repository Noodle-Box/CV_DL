from skimage import data # for inbuilt images
from skimage import color
from skimage.color import rgb2gray # for rgb2gray
import numpy as np
from skimage.morphology import dilation, erosion, disk, square, diamond, footprint_rectangle # for morphological operators and masks
import matplotlib.pyplot as plt


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

fig, axes = plt.subplots(1, ncols=3, figsize=(16, 8))
axes[0].imshow(inputImage, cmap='gray')
axes[0].axis('off')
axes[0].set_title('original grayscale')
axes[1].imshow(inputImage1, cmap='gray')
axes[1].axis('off')
axes[1].set_title('After dilation')
axes[2].imshow(inputImage2, cmap='gray')
axes[2].axis('off')
axes[2].set_title('After erosion')
plt.show()



## ## TASK 2: Take the UQ image and perform opening and closing with different masks ##