import os
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage import io, color, feature, morphology, exposure, restoration, transform
from skimage.draw import polygon_perimeter, circle_perimeter, polygon, disk as draw_disk, line as draw_line
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

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
    elif shape_type == 'Diamond':
        y = np.array([-half_size, 0, half_size, 0])
        x = np.array([0, half_size, 0, -half_size])
        
    return y, x

def create_binary_templates(size):
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
    
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
        
    gray_img = color.rgb2gray(image)

    enhanced_gray = exposure.equalize_adapthist(gray_img, clip_limit=params['clahe_clip_limit'])
    
    blurred = restoration.denoise_bilateral(enhanced_gray, 
                                            sigma_color=params['bilateral_color'], 
                                            sigma_spatial=params['bilateral_spatial'])
                                            
    raw_edges = feature.canny(blurred, 
                              sigma=params['canny_sigma'],
                              low_threshold=params['canny_low'],
                              high_threshold=params['canny_high'])
    
    hsv = color.rgb2hsv(image)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    
    red_mask = ((H < 0.05) | (H > 0.90)) & (S > params['color_s_min']) & (V > params['color_v_min'])
    yellow_mask = (H > 0.05) & (H < 0.3) & (S > params['yellow_s_min']) & (V > params['yellow_v_min'])
    blue_mask = (H > 0.50) & (H < 0.75) & (S > params['color_s_min']) & (V > params['color_v_min'])
    
    combined_color_mask = red_mask | yellow_mask | blue_mask
    dilated_color_mask = morphology.dilation(combined_color_mask, morphology.disk(25))
    
    # ==========================================
    # 5. Filter, Weld & Prune
    # ==========================================
    color_gated_edges = raw_edges & dilated_color_mask
    
    # WELD: Fuse the dashed fragments of the circles into solid rings for the Chamfer matcher
    welded_edges = morphology.binary_closing(color_gated_edges, morphology.disk(2))
    
    # BORDER WIPE: Erase the extreme 3 pixels of the image to kill outer JPEG frames
    welded_edges[0:3, :] = False
    welded_edges[-3:, :] = False
    welded_edges[:, 0:3] = False
    welded_edges[:, -3:] = False

    pruned_edges = morphology.remove_small_objects(welded_edges, 
                                                   min_size=params['min_edge_length'], 
                                                   connectivity=2)
    
    # ==========================================
    # 6a. Straight Line Extraction & Corner Chaining
    # ==========================================
    print("Extracting straight lines and chaining corners...")
    
    raw_lines = transform.probabilistic_hough_line(pruned_edges, 
                                               threshold=params['hough_threshold'], 
                                               line_length=params['hough_min_line_length'],
                                               line_gap=params['hough_line_gap'])
    
    def point_dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # ==========================================
    # THE COLLAGE KILLER
    # Photo seams are massive lines that span top-to-bottom. Signs are localized.
    # We instantly delete any line larger than 85% of the image height.
    # ==========================================
    max_allowed_len = image.shape[0] * 0.85
    filtered_lines = [line for line in raw_lines if point_dist(line[0], line[1]) < max_allowed_len]
    raw_lines = filtered_lines

    polygons = []
    max_gap = params.get('max_corner_gap', 20)
    used_lines = set()
    
    for i, l1 in enumerate(raw_lines):
        if i in used_lines: continue
        
        current_shape = [l1]
        used_lines.add(i)
        
        added_new = True
        while added_new:
            added_new = False
            for j, l2 in enumerate(raw_lines):
                if j in used_lines: continue
                
                connects = False
                for existing_line in current_shape:
                    dists = [
                        point_dist(existing_line[0], l2[0]), point_dist(existing_line[0], l2[1]),
                        point_dist(existing_line[1], l2[0]), point_dist(existing_line[1], l2[1])
                    ]
                    if min(dists) <= max_gap:
                        connects = True
                        break
                
                if connects:
                    current_shape.append(l2)
                    used_lines.add(j)
                    added_new = True
                    
        polygons.append(current_shape)

    valid_shapes = [poly for poly in polygons if len(poly) >= 3]
    print(f"Chained lines into {len(valid_shapes)} distinct geometric region proposals.")

# ==========================================
    # 6b. Localized Multi-Scale Chamfer Matching (With Noise Filtering)
    # ==========================================
    print("Running Fractional Chamfer Math on localized regions...")
    
    dt_image = distance_transform_edt(~pruned_edges)
    chamfer_tol = params.get('chamfer_tolerance', 2) 
    hit_mask = (dt_image <= chamfer_tol).astype(float)
    
    unsorted_detections = []
    
    for poly in valid_shapes:
        pts = []
        for line in poly:
            pts.extend([line[0], line[1]])
        pts = np.array(pts)
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        
        width = max_x - min_x
        height = max_y - min_y
        size = int(max(width, height))
        aspect_ratio = width / height if height > 0 else 0
        
        # FILTER 1: Size and Aspect Ratio check
        if size < 60 or aspect_ratio < 0.5 or aspect_ratio > 1.8: 
            continue 
        
        pad = int(size * 0.25) 
        p_min_y = max(0, int(min_y - pad))
        p_max_y = min(hit_mask.shape[0], int(max_y + pad))
        p_min_x = max(0, int(min_x - pad))
        p_max_x = min(hit_mask.shape[1], int(max_x + pad))
        
        patch_hit_mask = hit_mask[p_min_y:p_max_y, p_min_x:p_max_x]
        
        best_score = 0.0
        best_shape = "Unknown"
        
        for scale in [0.9, 1.0, 1.1]:
            test_size = int(size * scale)
            if test_size < 15: continue
            
            templates = create_binary_templates(test_size)
            
            for shape_name, template in templates.items():
                # Precision filter for specific geometries
                if shape_name in ['Circle', 'Square', 'Octagon', 'Diamond']:
                    if aspect_ratio < 0.7 or aspect_ratio > 1.3:
                        continue 
                        
                if patch_hit_mask.shape[0] < template.shape[0] or patch_hit_mask.shape[1] < template.shape[1]:
                    pad_y = max(0, template.shape[0] - patch_hit_mask.shape[0])
                    pad_x = max(0, template.shape[1] - patch_hit_mask.shape[1])
                    patch_hit_mask = np.pad(patch_hit_mask, ((0, pad_y), (0, pad_x)), mode='constant')

                num_edge_pixels = np.sum(template)
                if num_edge_pixels == 0: continue
                
                match_counts = fftconvolve(patch_hit_mask, template[::-1, ::-1].astype(float), mode='valid')
                max_score = np.max(match_counts) / num_edge_pixels
                
                if max_score > best_score:
                    best_score = max_score
                    best_shape = shape_name
        
        # FILTER 2: Minimum Chamfer Score to remove weak background noise
        if best_score > 0.60:
            unsorted_detections.append({
                'poly': poly,
                'shape': best_shape,
                'score': best_score * 100,
                'bbox': (min_x, min_y, max_x, max_y),
                'min_x': min_x
            })

    final_detections = sorted(unsorted_detections, key=lambda d: d['min_x'])

    print(f"\nFinal Sorted Detections (Left to Right):")
    for i, det in enumerate(final_detections):
        print(f" Sign {i+1}: {det['shape']} ({det['score']:.1f}%) at x={int(det['min_x'])}")

    # ==========================================
    # 7. YOLO-Style Visualization & Drawing
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(25, 6))
    axes[0].imshow(pruned_edges, cmap='gray')
    axes[0].set_title("1. Masked, Welded & Pruned Edges")
    
    vector_canvas = np.zeros_like(image)
    overlay_image = image.copy()
    
    # Background noise in blue
    for i, line_seg in enumerate(raw_lines):
        is_shape = any(line_seg in det['poly'] for det in final_detections)
        if not is_shape:
            p0, p1 = line_seg
            rr, cc = draw_line(p0[1], p0[0], p1[1], p1[0])
            vector_canvas[rr, cc] = [0, 100, 255]
            overlay_image[rr, cc] = [0, 100, 255]

    for det in final_detections:
        min_x, min_y, max_x, max_y = det['bbox']
        
        # Valid detection vectors in Red
        for line_seg in det['poly']:
            p0, p1 = line_seg
            rr, cc = draw_line(p0[1], p0[0], p1[1], p1[0])
            vector_canvas[rr, cc] = [255, 0, 0]
                    
        # Yellow Bounding Box
        rr, cc = polygon_perimeter([min_y, min_y, max_y, max_y], [min_x, max_x, max_x, min_x])
        overlay_image[rr, cc] = [255, 255, 0]

        axes[2].text(min_x, min_y - 15, f"{det['shape']}\n{det['score']:.1f}%", 
                     color='black', fontsize=10, fontweight='bold',
                     bbox=dict(facecolor='yellow', alpha=0.9, edgecolor='none', pad=2),
                     ha='left', va='bottom')

    axes[1].imshow(vector_canvas)
    axes[1].set_title(f"2. Valid Signs\n({len(final_detections)} Detected)")
    axes[2].imshow(overlay_image)
    axes[2].set_title("3. Final Classification")
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.show()

# ==========================================
if __name__ == "__main__":
    target_img_file = "A1_Streetsigns/Sign5.png" 
    
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
            'hough_threshold': 15,         
            'hough_min_line_length': 15,   
            'hough_line_gap': 10,          
            
            'max_corner_gap': 20,          # Dropped slightly to tighten the bounding box loops
            
            # --- Chamfer System ---
            'chamfer_tolerance': 3,        # Raised to 3 to accommodate the slightly thicker 'welded' edges!
        }
    
    if os.path.exists(target_img_file):
        detect_and_filter_sign(target_img_file, gatekeeper_params)
    else:
        print(f"Error: Could not find '{target_img_file}'.")