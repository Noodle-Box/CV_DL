import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, feature, filters, morphology, measure, exposure, restoration
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
from scipy.signal import fftconvolve
from skimage.draw import polygon_perimeter, circle_perimeter, polygon, disk as draw_disk

# ==========================================
# 1. Multi-Class Shape Generation
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
# 2. Create the Filtering Mask
# ==========================================
def generate_filtering_mask(image_shape, match_shape_type, match_scale, match_loc_tl):
    mask = np.zeros(image_shape, dtype=bool)
    y_tl, x_tl = match_loc_tl
    center_y = y_tl + (match_scale - 1) / 2.0
    center_x = x_tl + (match_scale - 1) / 2.0
    
    if match_shape_type == 'Circle':
        radius = (match_scale - 1) / 2.0
        rr, cc = draw_disk((center_y, center_x), radius, shape=image_shape)
        mask[rr, cc] = True
    else:
        y_centered, x_centered = get_shape_vertices(match_shape_type, match_scale)
        r_idx = y_centered + center_y
        c_idx = x_centered + center_x
        rr, cc = polygon(r_idx, c_idx, shape=image_shape)
        mask[rr, cc] = True
        
    return mask

# ==========================================
# 3. Main Execution
# ==========================================
def detect_and_filter_sign(image_path, params):
    image = io.imread(image_path)
    gray_img = color.rgb2gray(image)

    # ==========================================
    # 1. Enhanced Edge Detection (CLAHE)
    # ==========================================
    # Maximize local contrast so yellow/white signs "pop" against bright skies
    enhanced_gray = exposure.equalize_adapthist(gray_img, clip_limit=params['clahe_clip_limit'])
    
    # 1. Edge Detection (Now using tunable params)
    # Edge-Preserving Smoothing: Flattens pixelation but keeps geometric borders sharp
    blurred = restoration.denoise_bilateral(enhanced_gray, 
                                            sigma_color=params['bilateral_color'], 
                                            sigma_spatial=params['bilateral_spatial'])
    # 1. Manual Canny Thresholds
    # By forcing high_threshold up, we ignore weak background clutter
    raw_edges = feature.canny(blurred, 
                              sigma=params['canny_sigma'],
                              low_threshold=params['canny_low'],
                              high_threshold=params['canny_high'])
    
    # ==========================================
    # 1.5 The Retroreflective Gatekeeper
    # ==========================================
    hsv = color.rgb2hsv(image)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    
    # Standard Signs (Red, Blue)
    red_mask = ((H < 0.05) | (H > 0.90)) & (S > params['color_s_min']) & (V > params['color_v_min'])
    # blue_mask = (H > 0.55) & (H < 0.70) & (S > params['color_s_min']) & (V > params['color_v_min'])
    
    # TIGHT YELLOW: Desert sand has low saturation. Manufactured signs are highly saturated.
    yellow_mask = (H > 0.05) & (H < 0.3) & (S > params['yellow_s_min']) & (V > params['yellow_v_min'])
    
    # TIGHT WHITE: Must be extraordinarily bright (High V) with almost zero color bleed (Low S).
    # white_mask = (S < params['white_s_max']) & (V > params['white_v_min'])
    
    # Combine masks and dilate slightly to ensure it covers the edge gradients
    combined_color_mask = red_mask | yellow_mask
    dilated_color_mask = morphology.dilation(combined_color_mask, morphology.disk(25))
    
    # Filter the raw edges: Delete any edge that isn't a retroreflective color
    color_gated_edges = raw_edges & dilated_color_mask
    
    # NO CLOSING: We delete the bridge completely so the text doesn't blob together.
    
    # THE GEOMETRY PURIFIER: 
    # We prune the raw gated edges directly. By raising min_size heavily, 
    # we delete the dirt, the text, and the animals, leaving ONLY the giant borders!
    pruned_edges = morphology.remove_small_objects(color_gated_edges, 
                                                   min_size=params['min_edge_length'], 
                                                   connectivity=2)
    
    # Compute Distance Transform on the purified borders
    dt_image = distance_transform_edt(~pruned_edges)
    
    print("Initiating Multi-Scale FFT Chamfer Search...")
    
    best_overall_cost = np.inf
    best_scale_size = None
    best_match_loc_tl = None
    best_shape_class = None
    
    img_h, img_w = dt_image.shape
    min_dim = min(img_h, img_w)
    min_size = max(20, int(min_dim * 0.05))
    max_size = min(400, int(min_dim * 0.40))
    step_size = 5 
    
    # 3. Slide templates over the DT using fast FFT
    for template_size in range(min_size, max_size + 1, step_size):
        if template_size > img_h or template_size > img_w:
            break
            
        templates = create_binary_templates(size=template_size)
        
        for shape_name, template in templates.items():
            num_edge_pixels = np.sum(template)
            if num_edge_pixels == 0: continue
            
            cost_map = fftconvolve(dt_image, template[::-1, ::-1].astype(float), mode='valid') / num_edge_pixels
            min_cost = np.min(cost_map)
            
            if min_cost < best_overall_cost:
                best_overall_cost = min_cost
                best_scale_size = template_size
                best_shape_class = shape_name
                best_match_loc_tl = np.unravel_index(np.argmin(cost_map), cost_map.shape)

    if best_shape_class is None:
        print("Error: Could not lock onto a shape.")
        return

    print(f"Locked onto Shape: {best_shape_class} @ {best_scale_size}px")
    
    # 4. Generate spatial filter mask
    sign_mask = generate_filtering_mask(
        image_shape=(img_h, img_w), 
        match_shape_type=best_shape_class, 
        match_scale=best_scale_size, 
        match_loc_tl=best_match_loc_tl
    )
    
    # 5. Rough Filter (Drop edges completely outside the template mask)
    masked_edges = raw_edges & sign_mask
    
    # ==========================================
    # 6. THE GEOMETRIC GATEKEEPER
    # ==========================================
    closed_edges = morphology.closing(masked_edges, morphology.disk(params['closing_radius']))
    filled_regions = ndimage.binary_fill_holes(closed_edges)
    labeled_regions = measure.label(filled_regions)
    regions = measure.regionprops(labeled_regions)
    
    final_clean_edges = np.zeros_like(masked_edges)
    
    for region in regions:
        if region.area < params['min_area'] or region.area > params['max_area']:
            continue
        if region.solidity < params['min_solidity']:
            continue
            
        min_r, min_c, max_r, max_c = region.bbox
        aspect_ratio = (max_c - min_c) / float(max_r - min_r)
        if aspect_ratio < params['min_aspect'] or aspect_ratio > params['max_aspect']:
            continue
            
        if region.perimeter == 0:
            continue
        circularity = (4 * np.pi * region.area) / (region.perimeter ** 2)
        if circularity < params['min_circularity']:
            continue
            
        for coords in region.coords:
            final_clean_edges[coords[0], coords[1]] = masked_edges[coords[0], coords[1]]

# ==========================================
    # 7. Visualization (The Debug Layout)
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(25, 6))
    
    axes[0].imshow(image)
    axes[0].set_title("1. Original Image")
    
    # Let's look at what Canny is doing BEFORE the color mask!
    axes[1].imshow(raw_edges, cmap='gray')
    axes[1].set_title(f"2. RAW Canny Edges\n(If it's broken here, Canny is the problem)")
    
    # Now let's look at what survives the mask and the bridging
    axes[2].imshow(pruned_edges, cmap='gray')
    axes[2].set_title(f"3. Color Masked, Bridged, & Pruned")
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

# ==========================================
if __name__ == "__main__":
    target_jpg_file = "A1_Streetsigns/Sign4.jpg" 
    
    # ==========================================
    # EDGE EXTRACTION & GEOMETRY PARAMETERS
    # ==========================================
    gatekeeper_params = {
            # Edge Detection Control
            'gaussian_sigma': 1.0,     # Lowered to prevent over-smoothing
            'canny_sigma': 1.0,        
            'clahe_clip_limit': 0.03,  

            'bilateral_spatial': 10,   
            'bilateral_color': 0.1,    

            'canny_high': 0.40,        # Lowered heavily to allow the yellow/blue contrast to survive
            'canny_low': 0.10,         
            'min_edge_length': 50,     # Lowered to preserve the animals and text inside the signs

            # --- Retroreflective Color Thresholds ---
            'color_s_min': 0.40,       
            'color_v_min': 0.30,       
            
            # TIGHT YELLOW TUNING
            'yellow_s_min': 0.40,      # Relaxed to account for sunlight glare and compression
            'yellow_v_min': 0.50,      

            # TIGHT WHITE TUNING
            'white_s_max': 0.15,       
            'white_v_min': 0.85,       

            # Geometry Control
            'closing_radius': 3,       
            'min_area': 800,           
            'max_area': 500000,        
            'min_solidity': 0.85,      
            'min_aspect': 0.3,         
            'max_aspect': 3.0,         
            'min_circularity': 0.45    
        }
    
    if os.path.exists(target_jpg_file):
        detect_and_filter_sign(target_jpg_file, gatekeeper_params)
    else:
        print(f"Error: Could not find '{target_jpg_file}'.")