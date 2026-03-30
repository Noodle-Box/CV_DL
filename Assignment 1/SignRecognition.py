import os
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage import io, color, feature, morphology, exposure, restoration, transform
from skimage.draw import polygon_perimeter, circle_perimeter, polygon, disk as draw_disk, line as draw_line

# ==========================================
# 1. Multi-Class Shape Generation (Saved for Phase 2)
# ==========================================
def get_shape_vertices(shape_type, size):
    half_size = (size - 1) / 2.0
    
    if shape_type == 'Octagon':
        s = size / 3.0
        y = np.array([-half_size+s, half_size-s, half_size, half_size, half_size-s, -half_size+s, -half_size, -half_size])
        x = np.array([-half_size, -half_size, -half_size+s, half_size-s, half_size, half_size, half_size-s, -half_size+s])
    elif shape_type == 'Triangle_Up':
        y = np.array([-half_size, half_size, half_size])
        x = np.array([0, -half_size, half_size])
    elif shape_type == 'Triangle_Down':
        y = np.array([half_size, -half_size, -half_size])
        x = np.array([0, half_size, -half_size])
    elif shape_type == 'Square':
        y = np.array([-half_size, half_size, half_size, -half_size])
        x = np.array([-half_size, -half_size, half_size, half_size])
    elif shape_type == 'Rectangle':
        h_half, w_half = half_size, half_size * 0.7 
        y = np.array([-h_half, h_half, h_half, -h_half])
        x = np.array([-w_half, -w_half, w_half, w_half])
    elif shape_type == 'Diamond':
        y = np.array([-half_size, 0, half_size, 0])
        x = np.array([0, half_size, 0, -half_size])
        
    return y, x

def create_binary_templates(size):
    # ADDED 'Diamond' to the template generation list
    shapes = ['Octagon', 'Triangle_Up', 'Triangle_Down', 'Square', 'Rectangle', 'Diamond']
    templates = {}
    
    for shape in shapes:
        template = np.zeros((size, size), dtype=bool)
        y_centered, x_centered = get_shape_vertices(shape, size)
        
        r_idx = (y_centered + (size - 1) / 2.0).astype(int)
        c_idx = (x_centered + (size - 1) / 2.0).astype(int)
        
        rr, cc = polygon_perimeter(r_idx, c_idx)
        valid = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
        template[rr[valid], cc[valid]] = True
        templates[shape] = template

    # Add Circle
    circle_template = np.zeros((size, size), dtype=bool)
    center = int((size - 1) / 2.0)
    rr, cc = circle_perimeter(center, center, center)
    valid = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
    circle_template[rr[valid], cc[valid]] = True
    templates['Circle'] = circle_template
    
    return templates

# ==========================================
# 2. Main Edge Extraction Execution
# ==========================================
def detect_and_filter_sign(image_path, params):
    image = io.imread(image_path)
    
    # --- PNG FIX: Drop the Alpha Channel (RGBA -> RGB) ---
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
        
    gray_img = color.rgb2gray(image)

    # 1. Enhanced Edge Detection (CLAHE)
    enhanced_gray = exposure.equalize_adapthist(gray_img, clip_limit=params['clahe_clip_limit'])
    
    # 2. Edge-Preserving Smoothing
    blurred = restoration.denoise_bilateral(enhanced_gray, 
                                            sigma_color=params['bilateral_color'], 
                                            sigma_spatial=params['bilateral_spatial'])
                                            
    # 3. Manual Canny Thresholds
    raw_edges = feature.canny(blurred, 
                              sigma=params['canny_sigma'],
                              low_threshold=params['canny_low'],
                              high_threshold=params['canny_high'])
    
    # 4. The Retroreflective Gatekeeper
    hsv = color.rgb2hsv(image)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    
    red_mask = ((H < 0.05) | (H > 0.90)) & (S > params['color_s_min']) & (V > params['color_v_min'])
    yellow_mask = (H > 0.05) & (H < 0.3) & (S > params['yellow_s_min']) & (V > params['yellow_v_min'])
    
    combined_color_mask = red_mask | yellow_mask
    dilated_color_mask = morphology.dilation(combined_color_mask, morphology.disk(25))
    
    # 5. Filter & Prune
    color_gated_edges = raw_edges & dilated_color_mask
    pruned_edges = morphology.remove_small_objects(color_gated_edges, 
                                                   min_size=params['min_edge_length'], 
                                                   connectivity=2)
    
    # ==========================================
    # 6. Straight Line Vector Extraction (Probabilistic Hough)
    # ==========================================
    print("Extracting straight lines from pruned edges...")
    
    lines = transform.probabilistic_hough_line(pruned_edges, 
                                               threshold=params['hough_threshold'], 
                                               line_length=params['hough_min_line_length'],
                                               line_gap=params['hough_line_gap'])
    
    # Sort the detected lines from longest to shortest
    def get_line_length(l):
        p0, p1 = l
        return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
        
    lines.sort(key=get_line_length, reverse=True)
    print(f"Found {len(lines)} straight line segments.")

    # ==========================================
    # 7. Visualization & Drawing
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(25, 6))
    
    axes[0].imshow(pruned_edges, cmap='gray')
    axes[0].set_title("1. Masked & Pruned Edges\n(Input for Line Detection)")
    
    vector_canvas = np.zeros_like(image)
    overlay_image = image.copy()
    
    # Draw the extracted lines
    # Red = Top 8 longest (targeting 4 edges of diamond + 4 edges of rectangle)
    # Blue = Remaining shorter lines
    for i, line_seg in enumerate(lines):
        p0, p1 = line_seg
        x0, y0 = p0
        x1, y1 = p1
        
        if i < 8:
            line_color = [255, 0, 0]  # Red
            thickness = 3
        else:
            line_color = [0, 100, 255] # Blue
            thickness = 1
            
        rr, cc = draw_line(y0, x0, y1, x1)
        
        # Thicken lines for the visualizer
        for dr in range(-thickness//2 + 1, thickness//2 + 1):
            for dc in range(-thickness//2 + 1, thickness//2 + 1):
                r_thick = np.clip(rr + dr, 0, image.shape[0] - 1)
                c_thick = np.clip(cc + dc, 0, image.shape[1] - 1)
                vector_canvas[r_thick, c_thick] = line_color
                overlay_image[r_thick, c_thick] = line_color

    axes[1].imshow(vector_canvas)
    axes[1].set_title("2. Extracted Straight Lines\n(Red = Top 8 Longest Segments)")
    
    axes[2].imshow(overlay_image)
    axes[2].set_title("3. Lines Overlaid on Original")
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

# ==========================================
if __name__ == "__main__":
    target_img_file = "A1_Streetsigns/Sign2.jpg" 
    
    gatekeeper_params = {
            # Edge Detection Control
            'gaussian_sigma': 1.0,     
            'canny_sigma': 1.0,        
            'clahe_clip_limit': 0.03,  

            'bilateral_spatial': 10,   
            'bilateral_color': 0.1,    

            'canny_high': 0.40,        
            'canny_low': 0.10,         
            'min_edge_length': 50,     

            # --- Retroreflective Color Thresholds ---
            'color_s_min': 0.40,       
            'color_v_min': 0.30,       
            
            # TIGHT YELLOW TUNING
            'yellow_s_min': 0.43,      
            'yellow_v_min': 0.50,      

            # TIGHT WHITE TUNING
            'white_s_max': 0.15,       
            'white_v_min': 0.85,       
            
            # --- Vector Extraction (Hough) ---
            # These are now set very conservatively to avoid hallucinations
            'hough_threshold': 20,         # Minimum pixels required to register a line
            'hough_min_line_length': 40,   # Ignore tiny fragments
            'hough_line_gap': 5,           # Only bridge very small gaps (no jumping across the image!)
        }
    
    if os.path.exists(target_img_file):
        detect_and_filter_sign(target_img_file, gatekeeper_params)
    else:
        print(f"Error: Could not find '{target_img_file}'.")