#standard images in skimage
from skimage import data
from skimage import io, color
import numpy as np
# for plotting images


## TASK 1: Read and display an image ##
import matplotlib.pyplot as plt
inputImage = data.coins() #coin is a standard image
#plt.imshow(inputImage)
#plt.axis('off') # to hide axis
#plt.show()

## TASK 2: Read and display an image from your system ##
inputImage = io.imread('SG_UQ.jpeg') # Please change the path to your image
#imgplot = plt.imshow(inputImage)
#plt.show()

height, width, depth = inputImage.shape[:3]
print('Image Width: ', width, 'Image Height: ', height, 'Number of Channels', depth)

# Now, copy that image onto a new image but only the left hand side
outputImage_1 = np.zeros((height, width, depth), dtype=inputImage.dtype)

#Copy pixel value across three channels
for iRow in range(0,height):
    for iCol in range(0,width):
        for iChannel in range (0,depth):
            outputImage_1[iRow,int(iCol/2),iChannel] = inputImage[iRow,iCol,iChannel]

# Display output image
#plt.imshow(outputImage_1)
#plt.axis('off') # to hide axis
#plt.show()


## TASK 3: Take the UQ image and increase and decrease contrast and brightness ##
outImageContrast = inputImage.copy()
outImageContrast = color.rgb2gray(outImageContrast) # Convert RGB image into gray scale
print("dtype:", outImageContrast.dtype, "min/max:", outImageContrast.min(), outImageContrast.max())
#imgplot = plt.imshow(outImageContrast, cmap='gray')

# Contrast changing
alpha = 1.0     # contrast control >1 increase, <1 decrease
beta = 0.5      # brightness control (shift scalar)

outImageContrast = alpha * outImageContrast + beta
outImageContrast = np.clip(outImageContrast, 0, 1) # Clip values to be in the range [0, 1]

#plt.imshow(outImageContrast, cmap='gray')
#plt.title('Image with customizable contrast and brightness')


## TASK 4: Skin Map of an output image ##
img = io.imread('TheSquad.jpg')
hsv = color.rgb2hsv(img)

H = hsv[..., 0] # Hue channel
S = hsv[..., 1] # Saturation channel
V = hsv[..., 2] # Value channel

# Skin thresholds
h_min, h_max = 0.0, 0.16
s_min, s_max = 0.25, 0.8
v_min, v_max = 0.35, 1.0

# Apply skin mask to the image
skin_mask = (
            (H >= h_min) & (H <= h_max) & 
            (S >= s_min) & (S <= s_max) & 
            (V >= v_min) & (V <= v_max)
)

# Additional RGB rules for skin detection
R = img[..., 0].astype(int)
G = img[..., 1].astype(int)
B = img[..., 2].astype(int)

skin_rgb = (
    (R > 95) & (G > 40) & (B > 20) &
    ((np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B)) > 15) &
    (np.abs(R - G) > 15) &
    (R > G) & (R > B)
)

skin_mask = skin_mask & skin_rgb

skin_map = np.zeros_like(img, dtype=np.uint8)
skin_map[..., 0] = skin_mask * 255
skin_map[..., 1] = skin_mask * 255
skin_map[..., 2] = skin_mask * 255

plt.figure(figsize=(12,6))
plt.subplot(1,2,1); plt.imshow(img); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(skin_map); plt.title("Skin Mask"); plt.axis("off")
plt.show()



