####### Code for Question 3 - ELEC4630 A1 - Tevyn Vergara ######

############################################### Import Libraries ################################################################

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, restoration, morphology, measure
from skimage.util import img_as_ubyte, img_as_float
from sklearn.cluster import MiniBatchKMeans


################################################ Function Definitions #############################################################

def get_foreground_mask(image_uint8):
    """
    Updated Skin Segmentation with stronger Morphological Closing.
    Fills in the 'holes' (eyes/mouth) to create a solid silhouette.
    """
    hsv_image = color.rgb2hsv(image_uint8)
    hue, sat, val = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]
    
    # Refined Skin Thresholds
    skin_mask = (hue < 0.15) & (sat > 0.15) & (val > 0.15)
    
    # Stronger Closing: This fills in facial features for a solid cartoon look
    # Uses a larger disk as discussed in Lecture 2 for region properties
    cleaned_mask = morphology.binary_closing(skin_mask, morphology.disk(25))
    cleaned_mask = morphology.remove_small_objects(cleaned_mask, min_size=500)
    
    labels = measure.label(cleaned_mask)
    props = measure.regionprops(labels)
    if not props: return np.ones(image_uint8.shape, dtype=bool)
    
    largest_label = max(props, key=lambda x: x.area).label
    final_mask = (labels == largest_label)
    return np.stack((final_mask,) * 3, axis=-1)

def apply_kmeans(image_float, k=5): # Reduced K to 5 for flatter, "Target" look
    """
    Upgraded Method: Spatial Median Filtering + Quantization.
    Median filters are excellent for preserving edges while removing the 
    speckle noise seen in your previous result.
    """
    print("   -> Applying Median Filter for surface smoothing...")
    # Median filter effectively removes the 'ugly' noise splotches
    from skimage.filters import median
    from skimage.morphology import disk
    
    # Process each channel to maintain color integrity
    smoothed = np.zeros_like(image_float)
    for i in range(3):
        smoothed[:,:,i] = median(image_float[:,:,i], disk(7))

    h, w, c = smoothed.shape
    pixel_values = smoothed.reshape(h * w, c)
    
    print(f"   -> Quantizing to {k} colors...")
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(pixel_values)
    centers = kmeans.cluster_centers_

    return centers[labels].reshape(h, w, c)

def apply_bilateral(image_float):
    """
    Method 2: Bilateral Filtering.
    Scikit-Image's bilateral filter preserves edges while smoothing textures.
    """
    filtered = image_float
    # Apply iteratively for a stronger painted effect
    for _ in range(3):
        filtered = restoration.denoise_bilateral(
            filtered, win_size=9, sigma_color=0.1, sigma_spatial=15, channel_axis=-1
        )
    return filtered

def extract_edges(image_float):
    """
    Cleaned Edge Detection: Uses a higher threshold to avoid 'messy' lines.
    """
    gray = color.rgb2gray(image_float)
    # Stronger blur here prevents the 'hairy' edges in the results
    blurred_gray = filters.gaussian(gray, sigma=2.0)
    edge_magnitude = filters.sobel(blurred_gray)
    
    # Increased threshold (0.05) ensures only major outlines are drawn
    binary_edges = edge_magnitude < 0.05 
    return np.stack((binary_edges,) * 3, axis=-1)

def cartoonize(image_path, method='kmeans'):
    """
    Main pipeline using Scikit-Image's floating-point image standard.
    """
    # 1. Load the image. skimage loads as RGB by default.
    image_uint8 = io.imread(image_path)
    if image_uint8 is None:
        print(f"Error: Could not load image at {image_path}")
        return

    print(f"Processing image using the '{method}' method...")

    # Convert to float (0.0 to 1.0) for optimal skimage filter performance
    image_float = img_as_float(image_uint8)

    # 2. Extract the mask (MediaPipe needs the uint8 version)
    condition = get_foreground_mask(image_uint8)

    # 3. Process Background: Heavy Gaussian Blur
    # skimage's gaussian filter handles standard deviation (sigma) directly
    blurred_bg = filters.gaussian(image_float, sigma=15, channel_axis=-1)

    # 4. Process Foreground: Apply the chosen method
    if method == 'kmeans':
        cartoon_fg = apply_kmeans(image_float, k=6) 
    elif method == 'bilateral':
        # Note: skimage's bilateral filter is highly mathematically accurate 
        # and may take a few seconds longer to run than OpenCV's approximation.
        cartoon_fg = apply_bilateral(image_float)
    else:
        print("Invalid method. Please use 'kmeans' or 'bilateral'.")
        return

    # 5. Extract and overlay edges
    edges = extract_edges(image_float)
    # Multiply the cartoon colors by the boolean edge mask (darkens the edges to 0)
    cartoon_fg_with_edges = cartoon_fg * edges

    # 6. Composite the final image
    output_image_float = np.where(condition, cartoon_fg_with_edges, blurred_bg)

    # Convert back to standard 8-bit format for saving and display
    final_output = img_as_ubyte(output_image_float)
    
    # Save the output
    output_filename = f"skimage_cartoonized_{method}.jpg"
    io.imsave(output_filename, final_output)
    print(f"Saved output as {output_filename}")

    # Display using Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_uint8)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(final_output)
    axes[1].set_title(f"Cartoonized ({method.capitalize()})")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test Method 1: K-Means Clustering
    cartoonize('Profile_Test.jpg', method='kmeans')
    
    # Test Method 2: Bilateral Filtering (Uncomment to test)
    # cartoonize('image_710e43.jpg', method='bilateral')