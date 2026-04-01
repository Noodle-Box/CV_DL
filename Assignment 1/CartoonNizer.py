####### Code for Question 3 - ELEC4630 A1 - Tevyn Vergara ######

############################################### Import Libraries ################################################################

import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from skimage import io, color, filters, restoration
from skimage.util import img_as_ubyte, img_as_float
from sklearn.cluster import MiniBatchKMeans

################################################ Function Definitions #############################################################

def get_foreground_mask(image_uint8):
    """
    Uses MediaPipe to isolate the subject. 
    MediaPipe requires standard 8-bit RGB images.
    """
    # The correct direct import for modern MediaPipe versions
    from mediapipe.solutions import selfie_segmentation as mp_selfie_segmentation
    
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        results = selfie_segmentation.process(image_uint8)
        
        if results.segmentation_mask is None:
            return np.ones(image_uint8.shape, dtype=bool) 
            
        mask = results.segmentation_mask
        condition = np.stack((mask,) * 3, axis=-1) > 0.5
        return condition

def apply_kmeans(image_float, k=6):
    """
    Method 1: Color Quantization via K-Means Clustering.
    Using MiniBatchKMeans from scikit-learn is significantly faster for images.
    """
    h, w, c = image_float.shape
    # Flatten the image to a 2D array of pixels
    pixel_values = image_float.reshape(h * w, c)
    
    # Fit the K-Means algorithm
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(pixel_values)
    centers = kmeans.cluster_centers_
    
    # Reconstruct the image with the quantized color centers
    cartoonized_image = centers[labels].reshape(h, w, c)
    return cartoonized_image

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
    Uses the Sobel filter to find strong gradients and creates a dark edge mask.
    """
    gray = color.rgb2gray(image_float)
    # Sobel calculates the gradient magnitude
    edge_magnitude = filters.sobel(gray)
    
    # Threshold the gradients to isolate the strongest edges
    # We invert it so edges are False (dark) and flat regions are True (bright)
    binary_edges = edge_magnitude < 0.04 
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