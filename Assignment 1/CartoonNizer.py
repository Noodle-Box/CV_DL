import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io, color, filters, restoration, morphology, measure
from skimage.util import img_as_ubyte, img_as_float
from sklearn.cluster import MiniBatchKMeans

def get_foreground_mask(image_uint8):
    """
    Silhouette Extraction Method:
    Finds the borders of the subject and mathematically fills everything 
    inside the boundaries to ensure no internal details (like logos) are lost.
    """
    hsv_image = color.rgb2hsv(image_uint8)
    hue, sat, val = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]
    
    # 1. Build the "Fence" (Outer colors)
    skin_mask = ((hue < 0.15) | (hue > 0.9)) & (sat > 0.15) & (val > 0.15)
    shirt_mask = (hue > 0.5) & (hue < 0.7) & (val > 0.1)
    hair_mask = val < 0.35
    
    # 2. Add structural edges to reinforce the fence
    gray = color.rgb2gray(image_uint8)
    edges = filters.sobel(gray) > 0.04
    
    # Combine colors and edges
    combined_fence = skin_mask | shirt_mask | hair_mask | edges
    
    # 3. Close the perimeter to ensure there are no gaps in the outline
    closed_outline = morphology.binary_closing(combined_fence, morphology.disk(15))
    
    # 4. THE MAGIC STEP: Flood-fill everything inside the closed outline
    # This guarantees the logo and any internal shading are captured
    filled_silhouette = ndimage.binary_fill_holes(closed_outline)
    
    # 5. Isolate the largest connected component (You)
    labels = measure.label(filled_silhouette)
    props = measure.regionprops(labels)
    if not props: return np.ones(image_uint8.shape, dtype=bool)
        
    largest_label = max(props, key=lambda x: x.area).label
    final_mask = (labels == largest_label)
    
    # Smooth the outer boundary slightly so the cutout isn't jagged
    final_mask = morphology.binary_opening(final_mask, morphology.disk(5))
    
    return np.stack((final_mask,) * 3, axis=-1)

def apply_kmeans(image_float, k=8): 
    """
    Upgraded for strong, vibrant vector colors.
    """
    print("   -> Boosting color saturation for vibrant cartoon look...")
    # 1. Artificially boost saturation by 40% before clustering
    hsv = color.rgb2hsv(image_float)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.4, 0, 1) 
    enhanced_color = color.hsv2rgb(hsv)
    
    print("   -> Applying Balanced Smoothing Filter...")
    from skimage.filters import median
    from skimage.morphology import disk
    
    # 2. Reduced disk size to 5 (from 13). 
    # This smooths the skin but is small enough to PRESERVE the logo!
    smoothed = np.zeros_like(enhanced_color)
    for i in range(3):
        smoothed[:,:,i] = median(enhanced_color[:,:,i], disk(5))

    h, w, c = smoothed.shape
    pixel_values = smoothed.reshape(h * w, c)
    
    print(f"   -> Quantizing to {k} colors...")
    # 3. Increased K to 8 to ensure the red/white of the logo get their own clusters
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(pixel_values)
    centers = kmeans.cluster_centers_

    return centers[labels].reshape(h, w, c)

def apply_kmeans(image_float, k=8): 
    """
    Upgraded for strong, vibrant vector colors.
    """
    print("   -> Boosting color saturation for vibrant cartoon look...")
    # 1. Artificially boost saturation by 40% before clustering
    hsv = color.rgb2hsv(image_float)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.4, 0, 1) 
    enhanced_color = color.hsv2rgb(hsv)
    
    print("   -> Applying Balanced Smoothing Filter...")
    from skimage.filters import median
    from skimage.morphology import disk
    
    # 2. Reduced disk size to 5 (from 13). 
    # This smooths the skin but is small enough to PRESERVE the logo!
    smoothed = np.zeros_like(enhanced_color)
    for i in range(3):
        smoothed[:,:,i] = median(enhanced_color[:,:,i], disk(5))

    h, w, c = smoothed.shape
    pixel_values = smoothed.reshape(h * w, c)
    
    print(f"   -> Quantizing to {k} colors...")
    # 3. Increased K to 8 to ensure the red/white of the logo get their own clusters
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(pixel_values)
    centers = kmeans.cluster_centers_

def cartoonize(image_path, method='kmeans'):
    """
    Main pipeline featuring Background Hue Suppression.
    """
    image_uint8 = io.imread(image_path)
    if image_uint8 is None: return

    print(f"Processing image using the '{method}' method...")
    image_float = img_as_float(image_uint8)

    # 1. Extract the solid silhouette mask
    condition = get_foreground_mask(image_uint8)

    # 2. Process Background: Heavy Blur AND Hue Suppression
    print("   -> Blurring and desaturating background...")
    blurred_bg = filters.gaussian(image_float, sigma=15, channel_axis=-1)
    
    # Convert blurred background to HSV to kill the colors
    bg_hsv = color.rgb2hsv(blurred_bg)
    bg_hsv[:,:,1] *= 0.3 # Drop saturation to 30% (Suppresses Hue)
    bg_hsv[:,:,2] *= 0.8 # Darken slightly by 20% to make the foreground pop
    suppressed_bg = color.hsv2rgb(bg_hsv)

    # 3. Process Foreground
    if method == 'kmeans':
        cartoon_fg = apply_kmeans(image_float, k=8) 
    elif method == 'bilateral':
        cartoon_fg = apply_bilateral(image_float)

    # 4. Extract and overlay edges
    edges = extract_edges(image_float)
    cartoon_fg_with_edges = cartoon_fg * edges

    # 5. Composite using the suppressed background
    output_image_float = np.where(condition, cartoon_fg_with_edges, suppressed_bg)

    final_output = img_as_ubyte(output_image_float)
    
    output_filename = f"skimage_cartoonized_{method}.jpg"
    io.imsave(output_filename, final_output)
    print(f"Saved output as {output_filename}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_uint8)
    axes[0].set_title("Original")
    axes[0].axis('off')
    axes[1].imshow(final_output)
    axes[1].set_title(f"Cartoonized ({method.capitalize()})")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()