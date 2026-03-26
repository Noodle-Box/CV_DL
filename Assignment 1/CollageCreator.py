import os
import random
import math
import numpy as np
from skimage import io, color, data, feature, filters, morphology, measure, exposure, transform
from scipy import ndimage

# =====================================================================
# --- NON-MAXIMUM SUPPRESSION (IoM UPGRADE) ---
# =====================================================================
def non_max_suppression(boxes, overlapThresh=0.4):
    if len(boxes) == 0:
        return []
    r = np.array([b['r'] for b in boxes], dtype=float)
    c = np.array([b['c'] for b in boxes], dtype=float)
    w = np.array([b['width'] for b in boxes], dtype=float)
    h = np.array([b['height'] for b in boxes], dtype=float)
    y1, x1 = r, c
    y2, x2 = r + h, c + w
    area = w * h
    idxs = np.argsort(y2)
    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(boxes[i])
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w_int = np.maximum(0, xx2 - xx1)
        h_int = np.maximum(0, yy2 - yy1)
        area_overlap = w_int * h_int
        area_min = np.minimum(area[i], area[idxs[:last]])
        overlap = area_overlap / (area_min + 1e-5) 
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return pick

# =====================================================================
# --- GOLDEN PARAMETERS (The Turner 16 Set) ---
# =====================================================================
# Turn 16 Golden params
PAD_PERCENT_Y = 0.40  
PAD_PERCENT_X = 0.60  
BLUR_SIGMA = 1.2
CANNY_SIGMA = 1.0
# Goldilocks range (Lecture 3/4 theory)
CANNY_LOW = 0.04 
CANNY_HIGH = 0.12 
DILATION_SIZE = 7

def segment_pipeline(face_crop_color, blur_sigma, canny_sigma, dilation_size):
    if face_crop_color.size == 0:
        return None
    hsv = color.rgb2hsv(face_crop_color)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    skin_hue = ((h < 0.16) | (h > 0.70))
    chromatic_pass = skin_hue & (s > 0.10) & (v > 0.15)
    dark_features_pass = (v < 0.35) 
    is_green = (h > 0.18) & (h < 0.50) 
    color_mask = (chromatic_pass | dark_features_pass) & ~is_green
    color_mask = ndimage.binary_fill_holes(color_mask)
    
    gray = color.rgb2gray(face_crop_color)
    enhanced_gray = exposure.equalize_hist(gray)
    smoothed = filters.gaussian(enhanced_gray, sigma=blur_sigma)
    edges = feature.canny(smoothed, sigma=canny_sigma, low_threshold=CANNY_LOW, high_threshold=CANNY_HIGH)
    edges = edges & color_mask
    edges[-1, :] = True
    closed_edges = morphology.dilation(edges, morphology.disk(dilation_size))
    closed_edges[0, :] = False; closed_edges[:, 0] = False; closed_edges[:, -1] = False  
    filled = ndimage.binary_fill_holes(closed_edges)
    break_size = dilation_size + 1
    snapped_mask = morphology.erosion(filled, morphology.disk(break_size))
    labeled = measure.label(snapped_mask)
    h_idx, w_idx = snapped_mask.shape
    center_label = labeled[h_idx//2, w_idx//2]
    if center_label != 0:
        local_mask = (labeled == center_label)
    else:
        regions = measure.regionprops(labeled)
        if not regions: return None
        local_mask = (labeled == max(regions, key=lambda r: r.area).label)
    local_mask = morphology.dilation(local_mask, morphology.disk(break_size - 1))
    return ndimage.binary_fill_holes(local_mask) & color_mask

def apply_vintage_tone(img_float):
    """Applies a standard Sepia transformation matrix."""
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    vintage_img = img_float.dot(sepia_matrix.T)
    return np.clip(vintage_img, 0, 1)

def apply_vignette(canvas):
    """Applies a dark radial vignette to the canvas edges."""
    h, w = canvas.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    max_dist = np.sqrt(cy**2 + cx**2)
    
    # Non-linear falloff: 1.0 at center, drops heavily towards the edge
    vignette = 1.0 - 0.75 * (dist / max_dist)**1.5
    vignette = np.clip(vignette, 0, 1)
    
    # Multiply the canvas by the vignette mask
    return np.clip(canvas * vignette[..., None], 0, 1)

def apply_edge_fade(mask):
    """
    Forces the outer 15% of the mask to organically fade to 0. 
    This eliminates all flat lines from bounding box cutoffs.
    """
    h, w = mask.shape
    fade_y, fade_x = max(1, int(h * 0.15)), max(1, int(w * 0.15))
    
    fade_mask = np.ones((h, w), dtype=float)
    
    # Top and bottom linear gradients
    grad_y = np.linspace(0.0, 1.0, fade_y)
    fade_mask[:fade_y, :] *= grad_y[:, None]       # Fade Top
    fade_mask[-fade_y:, :] *= grad_y[::-1, None]   # Fade Bottom
    
    # Left and right linear gradients
    grad_x = np.linspace(0.0, 1.0, fade_x)
    fade_mask[:, :fade_x] *= grad_x[None, :]       # Fade Left
    fade_mask[:, -fade_x:] *= grad_x[None, ::-1]   # Fade Right
    
    return mask * fade_mask

# =====================================================================
# --- MAIN PIPELINE ---
# =====================================================================
def main():
    target_image = "Extracted Pictures/frame_01.png"
    output_path = "Artistic_Vintage_Mosaic.png"

    img = io.imread(target_image)
    gray_img = color.rgb2gray(img)
    detector = feature.Cascade(data.lbp_frontal_face_cascade_filename())

    raw_faces = detector.detect_multi_scale(img=gray_img, scale_factor=1.05, 
                                            step_ratio=1.05, min_size=(100, 100), 
                                            max_size=(350, 350))

    filtered_faces = non_max_suppression(raw_faces, overlapThresh=0.4)
    face_data = []

    for face in filtered_faces:
        r, c, w, h = face['r'], face['c'], face['width'], face['height']
        pad_r, pad_c = int(h * 0.40), int(w * 0.60)
        r_start, c_start = max(0, r - pad_r), max(0, c - pad_c)
        r_end, c_end = min(img.shape[0], r+h+pad_r), min(img.shape[1], c+w+pad_c)
        
        crop = img[r_start:r_end, c_start:c_end]
        mask = segment_pipeline(crop, 1.2, 1.0, 7)

        if mask is not None and np.any(mask):
            rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            face_only = crop[rmin:rmax, cmin:cmax]
            mask_only = mask[rmin:rmax, cmin:cmax].astype(float)
            score = np.mean(np.std(face_only[mask_only > 0.5], axis=0))
            face_data.append({'img': face_only, 'mask': mask_only, 'score': score})

    face_data.sort(key=lambda x: x['score'])

    num_faces = len(face_data)
    dim = int(math.ceil(math.sqrt(num_faces)))
    CELL_SIZE = 280  
    CANVAS_SIZE = dim * CELL_SIZE 
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3))
    
    all_cells = [(r, c) for r in range(dim) for c in range(dim)]
    corners = [(0, 0), (0, dim-1), (dim-1, 0), (dim-1, dim-1)]
    center_idx = (dim - 1) / 2.0
    remaining_cells = [cell for cell in all_cells if cell not in corners]
    remaining_cells.sort(key=lambda x: np.sqrt((x[0]-center_idx)**2 + (x[1]-center_idx)**2))

    placement_order = corners + remaining_cells

    print(f"Assembling high-variance full-coverage mosaic ({num_faces} faces)...")

    for i in range(num_faces):
        entry = face_data[i]
        (grid_r, grid_c) = placement_order[i]
        
        orig_h, orig_w = entry['img'].shape[:2]
        scale_type = random.random()
        
        if scale_type < 0.30:
            scale_factor = random.uniform(0.2, 0.8)
        elif scale_type > 0.70:
            scale_factor = random.uniform(1.5, 2.0)
        else:
            scale_factor = random.uniform(0.8, 1.5)
            
        new_h = int(orig_h * (CELL_SIZE / 200) * scale_factor)
        new_w = int(orig_w * (CELL_SIZE / 200) * scale_factor)
        
        if new_h < 10 or new_w < 10:
            continue
            
        face_res = transform.resize(entry['img'], (new_h, new_w), anti_aliasing=True)
        mask_res = transform.resize(entry['mask'], (new_h, new_w), anti_aliasing=True)
        
        # --- THE FIX: Eliminate Flat Lines ---
        # Run the mask through the linear gradient fader before blurring
        faded_mask = apply_edge_fade(mask_res)
        
        # High sigma blur now acts on the faded mask for ultra-smooth transitions
        soft_alpha = filters.gaussian(faded_mask, sigma=12.0)[..., None]

        jitter_allowance = int(CELL_SIZE * 0.35) 
        y_jitter = random.randint(-jitter_allowance, jitter_allowance)
        x_jitter = random.randint(-jitter_allowance, jitter_allowance)
        
        y_pos = int(grid_r * CELL_SIZE + y_jitter + (CELL_SIZE - new_h)//2)
        x_pos = int(grid_c * CELL_SIZE + x_jitter + (CELL_SIZE - new_w)//2)
        
        y_start, y_end = max(0, y_pos), min(CANVAS_SIZE, y_pos + new_h)
        x_start, x_end = max(0, x_pos), min(CANVAS_SIZE, x_pos + new_w)
        
        face_patch = face_res[:(y_end-y_start), :(x_end-x_start)]
        alpha_patch = soft_alpha[:(y_end-y_start), :(x_end-x_start)]
        bg_patch = canvas[y_start:y_end, x_start:x_end]
        
        if face_patch.shape == bg_patch.shape:
            canvas[y_start:y_end, x_start:x_end] = (face_patch * alpha_patch) + (bg_patch * (1.0 - alpha_patch))

    # --- THE FIX: Vintage Filter + Dark Vignette ---
    print("Applying tone and vignette...")
    final_canvas = apply_vintage_tone(canvas)
    final_canvas = apply_vignette(final_canvas)
    
    io.imsave(output_path, (final_canvas * 255).astype(np.uint8))
    print(f"Success! Artistic mosaic saved as {output_path}")

if __name__ == "__main__":
    main()