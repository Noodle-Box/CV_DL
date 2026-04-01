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
    """
    PURPOSE: Calculates local vertex coordinates for geometric road sign templates.
    INPUTS: 
        - shape_type (str): The class of shape to generate (e.g., 'Octagon', 'Diamond').
        - size (int): The diameter/bounding dimension of the template in pixels.
    OUTPUTS:
        - y (np.array), x (np.array): Arrays containing the centered vertex coordinates.
    LINKED TO: create_binary_templates() uses these vertices to draw mathematical perimeters.
    """
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
    """
    PURPOSE: Generates a dictionary of binary 1-pixel thick wireframe images for shape matching.
    INPUTS: 
        - size (int): The pixel dimension of the template canvas.
    OUTPUTS:
        - templates (dict): Dictionary mapping shape names to 2D boolean edge arrays.
    LINKED TO: detect_and_filter_sign() convolved these templates against image patches for Chamfer scoring.
    """
    shapes = ['Octagon', 'Triangle_Up', 'Triangle_Down', 'Square', 'Rectangle', 'Diamond']
    templates = {}
    
    for shape in shapes:
        template = np.zeros((size, size), dtype=bool)
        y_centered, x_centered = get_shape_vertices(shape, size)
        
        # Shift coordinates to center of the template canvas
        r_idx = (y_centered + (size - 1) / 2.0).astype(int)
        c_idx = (x_centered + (size - 1) / 2.0).astype(int)
        
        # Rasterize the polygon boundary
        rr, cc = polygon_perimeter(r_idx, c_idx)
        valid = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
        template[rr[valid], cc[valid]] = True
        templates[shape] = template

    # Add Circle separately using skimage's native circle drawer
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
def detect_and_filter_sign(image_path, params, output_dir=None):
    """
    PURPOSE: Processes a raw image to detect, cluster, and classify traffic sign geometries.
    INPUTS: 
        - image_path (str): File path to image.
        - params (dict): Tuning parameters for edge detection, color, and matching.
        - output_dir (str): Optional folder to save diagnostic plot panels.
    OUTPUTS:
        - Displays or saves a 3-panel visualization showing the detection lifecycle.
    LINKED TO: Called in a loop by the main batch processor to evaluate the entire dataset.
    """
    image = io.imread(image_path)
    
    # Pre-processing: Standardize to RGB and convert to grayscale/HSV
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    gray_img = color.rgb2gray(image)

    # 1. Contrast Enhancement to help signs pop in varying light
    enhanced_gray = exposure.equalize_adapthist(gray_img, clip_limit=params['clip_limit'])
    
    # 2. Noise Reduction: Bilateral filter to smooth texture while keeping structural edges sharp
    blurred = restoration.denoise_bilateral(enhanced_gray, 
                                            sigma_color=params['bilateral_color'], 
                                            sigma_spatial=params['bilateral_spatial'])
                                            
    # 3. Edge Detection: Canny transform isolates high-gradient structural lines
    raw_edges = feature.canny(blurred, 
                              sigma=params['canny_sigma'],
                              low_threshold=params['canny_low'],
                              high_threshold=params['canny_high'])
    
    # 4. Color Gating: HSV isolation of Red, Yellow, and Blue retroreflective pixels
    hsv = color.rgb2hsv(image)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    
    red_mask = ((H < 0.05) | (H > 0.90)) & (S > params['color_s_min']) & (V > params['color_v_min'])
    yellow_mask = (H > 0.05) & (H < 0.3) & (S > params['yellow_s_min']) & (V > params['yellow_v_min'])
    blue_mask = (H > 0.50) & (H < 0.75) & (S > params['color_s_min']) & (V > params['color_v_min'])
    
    combined_color_mask = red_mask | yellow_mask | blue_mask
    dilated_color_mask = morphology.dilation(combined_color_mask, morphology.disk(25))
    
    # 5. Morphological Cleanup: Filter edges by color and prune isolated noise fragments
    color_gated_edges = raw_edges & dilated_color_mask
    welded_edges = morphology.binary_closing(color_gated_edges, morphology.disk(2))
    
    # Edge Wipe: Remove photo borders from interfering with line extraction
    welded_edges[0:3, :] = False
    welded_edges[-3:, :] = False
    welded_edges[:, 0:3] = False
    welded_edges[:, -3:] = False

    pruned_edges = morphology.remove_small_objects(welded_edges, 
                                                   min_size=params['min_edge_length'], 
                                                   connectivity=2)
    
    # 6a. Straight Line Extraction: Converts pixels into mathematical vector segments
    print("Extracting straight lines and chaining corners...")
    raw_lines = transform.probabilistic_hough_line(pruned_edges, 
                                               threshold=params['hough_threshold'], 
                                               line_length=params['hough_min_line_length'],
                                               line_gap=params['hough_line_gap'])
    
    def point_dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Collage Filter: Ignore massive vertical lines caused by photo borders in a collage
    max_allowed_len = image.shape[0] * 0.85
    filtered_lines = [line for line in raw_lines if point_dist(line[0], line[1]) < max_allowed_len]
    raw_lines = filtered_lines

    # Corner Chaining: Groups touching lines into "Polygon" region proposals
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
                    dists = [point_dist(existing_line[0], l2[0]), point_dist(existing_line[0], l2[1]),
                             point_dist(existing_line[1], l2[0]), point_dist(existing_line[1], l2[1])]
                    if min(dists) <= max_gap:
                        connects = True; break
                if connects:
                    current_shape.append(l2); used_lines.add(j); added_new = True
        polygons.append(current_shape)

    valid_shapes = [poly for poly in polygons if len(poly) >= 3]
    print(f"Chained lines into {len(valid_shapes)} distinct geometric region proposals.")

    # 6b. Localized Chamfer Matching: Classifies regions based on geometric template scores
    print("Running Fractional Chamfer Math on localized regions...")
    
    dt_image = distance_transform_edt(~pruned_edges) # Distance from every black pixel to nearest white pixel
    chamfer_tol = params.get('chamfer_tolerance', 2) 
    hit_mask = (dt_image <= chamfer_tol).astype(float) # Areas where a template pixel is considered an "edge hit"
    
    unsorted_detections = []
    
    for poly in valid_shapes:
        # Generate sign bounding box and scale-normalized size
        pts = np.array([p for line in poly for p in line])
        min_x, min_y = np.min(pts, axis=0); max_x, max_y = np.max(pts, axis=0)
        width, height = max_x - min_x, max_y - min_y
        size = int(max(width, height)); aspect_ratio = width / height if height > 0 else 0
        
        if size < 60 or aspect_ratio < 0.5 or aspect_ratio > 1.8: continue 
        
        # Patch Cropping: Localizes math to a specific sign region to boost speed
        pad = int(size * 0.25) 
        p_min_y, p_max_y = max(0, int(min_y-pad)), min(hit_mask.shape[0], int(max_y+pad))
        p_min_x, p_max_x = max(0, int(min_x-pad)), min(hit_mask.shape[1], int(max_x+pad))
        patch_hit_mask = hit_mask[p_min_y:p_max_y, p_min_x:p_max_x]
        
        best_score = 0.0; best_shape = "Unknown"
        
        # Testing templates at 0.9x, 1.0x, and 1.1x scales to accommodate distance variations
        for scale in [0.9, 1.0, 1.1]:
            test_size = int(size * scale)
            if test_size < 15: continue
            templates = create_binary_templates(test_size)
            
            for shape_name, template in templates.items():
                # Aspect Ratio Gatekeeper: Prevent shape hallucination on skewed boxes
                if shape_name in ['Circle', 'Square', 'Octagon', 'Diamond'] and (aspect_ratio < 0.7 or aspect_ratio > 1.3):
                    continue 
                
                # Padding Fix: Ensures template and patch align for FFT convolution
                if patch_hit_mask.shape[0] < template.shape[0] or patch_hit_mask.shape[1] < template.shape[1]:
                    pad_y = max(0, template.shape[0] - patch_hit_mask.shape[0])
                    pad_x = max(0, template.shape[1] - patch_hit_mask.shape[1])
                    patch_hit_mask = np.pad(patch_hit_mask, ((0, pad_y), (0, pad_x)), mode='constant')

                # Fractional Scoring: Percentage of template pixels that land on an image edge
                num_edge_pixels = np.sum(template)
                if num_edge_pixels == 0: continue
                match_counts = fftconvolve(patch_hit_mask, template[::-1, ::-1].astype(float), mode='valid')
                max_score = np.max(match_counts) / num_edge_pixels
                
                if max_score > best_score:
                    best_score = max_score; best_shape = shape_name
        
        # Filtering low-confidence detections
        if best_score > 0.60:
            unsorted_detections.append({'poly': poly, 'shape': best_shape, 'score': best_score * 100,
                                        'bbox': (min_x, min_y, max_x, max_y), 'min_x': min_x})

    # Output Sorting: Organizing detections left-to-right for terminal consistency
    final_detections = sorted(unsorted_detections, key=lambda d: d['min_x'])

    print(f"\nFinal Sorted Detections (Left to Right):")
    for i, det in enumerate(final_detections):
        print(f" Sign {i+1}: {det['shape']} ({det['score']:.1f}%) at x={int(det['min_x'])}")

    # 7. Visualization: YOLO-style Bounding Box and Chamfer Classification display
    fig, axes = plt.subplots(1, 3, figsize=(25, 6))
    axes[0].imshow(pruned_edges, cmap='gray'); axes[0].set_title("1. Masked, Welded & Pruned Edges")
    vector_canvas = np.zeros_like(image); overlay_image = image.copy()
    
    # Blue drawing for background lines (rejected noise)
    for i, line_seg in enumerate(raw_lines):
        if not any(line_seg in det['poly'] for det in final_detections):
            rr, cc = draw_line(line_seg[0][1], line_seg[0][0], line_seg[1][1], line_seg[1][0])
            vector_canvas[rr, cc] = [0, 100, 255]; overlay_image[rr, cc] = [0, 100, 255]

    for det in final_detections:
        min_x, min_y, max_x, max_y = det['bbox']
        # Red drawing for sign edges (accepted polygons)
        for line_seg in det['poly']:
            rr, cc = draw_line(line_seg[0][1], line_seg[0][0], line_seg[1][1], line_seg[1][0])
            vector_canvas[rr, cc] = [255, 0, 0]
        # Yellow Bounding Box
        rr, cc = polygon_perimeter([min_y, min_y, max_y, max_y], [min_x, max_x, max_x, min_x])
        overlay_image[rr, cc] = [255, 255, 0]
        axes[2].text(min_x, min_y - 15, f"{det['shape']}\n{det['score']:.1f}%", color='black', 
                     fontsize=10, fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.9), ha='left', va='bottom')

    axes[1].imshow(vector_canvas); axes[1].set_title(f"2. Valid Signs\n({len(final_detections)} Detected)")
    axes[2].imshow(overlay_image); axes[2].set_title("3. Final Classification")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"detected_{os.path.basename(image_path)}"), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

# ==========================================
if __name__ == "__main__":
    """
    MAIN ENTRY POINT: Sets up the batch environment and defines critical mathematical parameters.
    """
    input_folder = "A1_Streetsigns"
    output_folder = "Sign Detection results"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # PARAMETER TUNING RATIONALE:
    # 'clahe_clip_limit': Balances local contrast vs. noise amplification.
    # 'canny_high/low': Strictness of edge hysteresis. Controls sensitivity to faint sign borders.
    # 'min_edge_length': Rejects "Salt and Pepper" noise pixels.
    # 'hough_min_line_length': Minimum pixels to be considered a geometric side.
    # 'max_corner_gap': Tolerance for gaps in the edge map when chaining lines together.
    # 'chamfer_tolerance': The "forgiveness" bubble (px) allowed between template and real edge.
    
    gatekeeper_params = {
            # Canny and Contrast parametes
            'gaussian_sigma': 1.0, 'canny_sigma': 1.0, 'clip_limit': 0.03,

            # Bilateral filter for noise reduction while preserving edges
            'bilateral_spatial': 10, 'bilateral_color': 0.1,
            
            # Canny thresholds for edge detection and gradient thesholding
            'canny_high': 0.40, 'canny_low': 0.10, 'min_edge_length': 50,

            # HSV color gating thresholds for Red, Yellow, and Blue sign isolation
            'color_s_min': 0.40, 'color_v_min': 0.30,

            'yellow_s_min': 0.43, 'yellow_v_min': 0.50,

            # White color gating thresholds
            'white_s_max': 0.15, 'white_v_min': 0.85,


            # Hough Transform parameters for line extraction and corner chaining
            'hough_threshold': 15, 'hough_min_line_length': 15, 'hough_line_gap': 10,

            # Corner and chamfering parameters
            'max_corner_gap': 20, 'chamfer_tolerance': 3,
        }
    
    if os.path.exists(input_folder):
        print(f"Starting batch processing in '{input_folder}'...")
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                detect_and_filter_sign(os.path.join(input_folder, filename), gatekeeper_params, output_folder)
        print("\nBatch processing complete!")
    else:
        print(f"Error: Input folder '{input_folder}' missing.")