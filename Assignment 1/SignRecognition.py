import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, feature, morphology, exposure, restoration
from skimage.draw import polygon_perimeter, circle_perimeter, polygon, disk as draw_disk

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
        
    return y, x

def create_binary_templates(size):
    shapes = ['Octagon', 'Triangle_Up', 'Triangle_Down', 'Square', 'Rectangle']
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
    # Visualization (The Debug Layout)
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(25, 6))
    
    axes[0].imshow(image)
    axes[0].set_title("1. Original Image")
    
    axes[1].imshow(raw_edges, cmap='gray')
    axes[1].set_title("2. RAW Canny Edges")
    
    axes[2].imshow(pruned_edges, cmap='gray')
    axes[2].set_title("3. Color Masked & Pruned")
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

# ==========================================
if __name__ == "__main__":
    target_img_file = "A1_Streetsigns/Sign5.png" # Test with a PNG!
    
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
            'yellow_s_min': 0.40,      
            'yellow_v_min': 0.50,      

            # TIGHT WHITE TUNING
            'white_s_max': 0.15,       
            'white_v_min': 0.85,       
        }
    
    if os.path.exists(target_img_file):
        detect_and_filter_sign(target_img_file, gatekeeper_params)
    else:
        print(f"Error: Could not find '{target_img_file}'.")