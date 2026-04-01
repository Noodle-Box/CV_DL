## QUESTION 1: COLLAGE CREATOR - FINAL DOCUMENTED VERSION

import os
import random
import math
import numpy as np
from skimage import io, color, data, feature, filters, morphology, measure, exposure, transform
from scipy import ndimage

############################################ Function Defines ##############################################

def non_max_suppression(boxes, overlap_thresh):
    """
    PURPOSE: Filters through raw face detections to remove redundant, overlapping boxes.
    INPUTS:
        - boxes (list): A list of dictionaries containing 'r', 'c', 'width', and 'height'.
        - overlap_thresh (float): Sensitivity for merging (0.0 to 1.0).
    OUTPUTS:
        - pick (list): A refined list of unique face dictionaries.
    USED IN: main() - It takes the raw output of the Cascade detector and cleans it up 
             before the individual segmentation loop begins.
    """
    if len(boxes) == 0: return []
    
    # Extract coordinates into numpy arrays for vectorized math
    r, c = np.array([b['r'] for b in boxes]), np.array([b['c'] for b in boxes])
    w, h = np.array([b['width'] for b in boxes]), np.array([b['height'] for b in boxes])
    y2, x2 = r + h, c + w
    area, idxs = w * h, np.argsort(y2)
    pick = []
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(boxes[i])
        
        # Calculate intersection area with all other boxes
        xx1, yy1 = np.maximum(c[i], c[idxs[:last]]), np.maximum(r[i], r[idxs[:last]])
        xx2, yy2 = np.minimum(x2[i], x2[idxs[:last]]), np.minimum(y2[i], y2[idxs[:last]])
        w_int, h_int = np.maximum(0, xx2 - xx1), np.maximum(0, yy2 - yy1)
        
        # Intersection over Minimum (IoM) calculation
        overlap = (w_int * h_int) / (np.minimum(area[i], area[idxs[:last]]) + 1e-5)
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    return pick

def segment_pipeline(face_crop, p):
    """
    PURPOSE: Generates a binary mask that isolates a person's head and hair from the background.
    INPUTS:
        - face_crop (ndarray): The RGB image patch of a single detected face.
        - p (dict): The global parameter dictionary containing sigma and threshold values.
    OUTPUTS:
        - mask (ndarray/None): A binary mask (True/False) or None if segmentation fails.
    USED IN: main() - The output mask is used to perform a "tight crop" and calculate 
             the color variation score for sorting.
    """
    if face_crop.size == 0: return None
    
    # --- Spectral Filtering (Skin & Hair Tone) ---
    hsv = color.rgb2hsv(face_crop)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    color_mask = (((h < 0.16) | (h > 0.70)) & (s > 0.10) & (v > 0.15)) | (v < 0.35)
    color_mask = color_mask & ~((h > 0.18) & (h < 0.50)) # Eliminate green screen artifacts
    color_mask = ndimage.binary_fill_holes(color_mask)
    
    # --- Geometric Filtering (Head Outlines) ---
    gray = exposure.equalize_hist(color.rgb2gray(face_crop))
    smoothed = filters.gaussian(gray, sigma=p['blur_sigma'])
    edges = feature.canny(smoothed, sigma=p['canny_sigma'], 
                          low_threshold=p['canny_low'], high_threshold=p['canny_high'])
    edges = edges & color_mask
    edges[-1, :] = True # Seal neck to allow hole filling
    
    # --- Connectivity & Morphology ---
    filled = ndimage.binary_fill_holes(morphology.dilation(edges, morphology.disk(p['dilation_size'])))
    snapped = morphology.erosion(filled, morphology.disk(p['dilation_size'] + 1))
    labeled = measure.label(snapped)
    
    # Select the largest central object
    center_label = labeled[labeled.shape[0]//2, labeled.shape[1]//2]
    if center_label != 0:
        mask = (labeled == center_label)
    else:
        regions = measure.regionprops(labeled)
        mask = (labeled == max(regions, key=lambda r: r.area).label) if regions else None
    
    if mask is None: return None
    return ndimage.binary_fill_holes(morphology.dilation(mask, morphology.disk(p['dilation_size']))) & color_mask

def apply_edge_fade(mask, ratio):
    """
    PURPOSE: Eliminates hard straight lines by fading the edges of a mask to zero transparency.
    INPUTS:
        - mask (ndarray): A 2D array representing the facial mask.
        - ratio (float): The percentage of the border to be faded (e.g., 0.15 = 15%).
    OUTPUTS:
        - mask (ndarray): The input mask multiplied by a 4-way linear gradient.
    USED IN: main() - Applied to every face right before alpha-blending onto the canvas.
    """
    h, w = mask.shape
    fy, fx = max(1, int(h * ratio)), max(1, int(w * ratio))
    m, gy, gx = np.ones((h, w)), np.linspace(0, 1, fy), np.linspace(0, 1, fx)
    # Apply vertical and horizontal gradient multipliers
    m[:fy, :] *= gy[:, None]; m[-fy:, :] *= gy[::-1, None]
    m[:, :fx] *= gx[None, :]; m[:, -fx:] *= gx[None, ::-1]
    return mask * m

def apply_vintage_tone(img):
    """
    PURPOSE: Simulates old photography by shifting RGB values toward Sepia tones.
    INPUTS: img (ndarray): The fully assembled RGB collage canvas.
    OUTPUTS: img (ndarray): The color-shifted image in float format.
    USED IN: main() - Applied as the first global post-processing step.
    """
    sepia = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
    return np.clip(img.dot(sepia.T), 0, 1)

def apply_vignette(img, intensity, falloff):
    """
    PURPOSE: Adds an artistic lens effect by darkening the corners of the canvas.
    INPUTS:
        - img (ndarray): The RGB canvas.
        - intensity (float): How dark the corners become.
        - falloff (float): The curve of the darkening (1.0 = linear).
    OUTPUTS: img (ndarray): The final vignetted image.
    USED IN: main() - The very last step before saving the file.
    """
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - h/2)**2 + (x - w/2)**2)
    vig = np.clip(1.0 - intensity * (dist / np.sqrt((h/2)**2 + (w/2)**2))**falloff, 0, 1)
    return np.clip(img * vig[..., None], 0, 1)

# =====================================================================
# --- MAIN EXECUTION ENGINE ---
# =====================================================================
def main(p):
    """
    PURPOSE: Orchestrates the entire process of face detection, segmentation, and collage assembly.
    INPUTS: p (dict): A dictionary of all tunable parameters passed from the main block.
    OUTPUTS: None. It saves the resulting image to the specified 'out_path'.
    """
    img = io.imread(p['target_path'])
    detector = feature.Cascade(data.lbp_frontal_face_cascade_filename())
    
    # 1. Detection
    raw_faces = detector.detect_multi_scale(img=color.rgb2gray(img), scale_factor=p['det_scale'], 
                                            step_ratio=p['det_step'], min_size=p['min_f'], max_size=p['max_f'])
    filtered = non_max_suppression(raw_faces, p['nms_thresh'])
    
    # 2. Segmentation & Metric Extraction
    face_data = []
    for f in filtered:
        r, c, w, h = f['r'], f['c'], f['width'], f['height']
        pr, pc = int(h * p['pad_y']), int(w * p['pad_x'])
        crop = img[max(0, r-pr):min(img.shape[0], r+h+pr), max(0, c-pc):min(img.shape[1], c+w+pc)]
        mask = segment_pipeline(crop, p)
        if mask is not None and np.any(mask):
            rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
            rmin, rmax, cmin, cmax = np.where(rows)[0][0], np.where(rows)[0][-1], np.where(cols)[0][0], np.where(cols)[0][-1]
            face_img, mask_img = crop[rmin:rmax, cmin:cmax], mask[rmin:rmax, cmin:cmax].astype(float)
            # Output stored in dictionary for assembly
            face_data.append({'img': face_img, 'mask': mask_img, 'score': np.mean(np.std(face_img[mask_img > 0.5], axis=0))})

    # 3. Assembly Logic
    face_data.sort(key=lambda x: x['score'])
    dim = int(math.ceil(math.sqrt(len(face_data))))
    canvas = np.zeros((dim * p['cell_size'], dim * p['cell_size'], 3))
    
    # Map grid cells prioritizing corners
    cells = [(r, c) for r in range(dim) for c in range(dim)]
    corners = [(0, 0), (0, dim-1), (dim-1, 0), (dim-1, dim-1)]
    rem = sorted([c for c in cells if c not in corners], key=lambda x: np.sqrt((x[0]-(dim-1)/2)**2 + (x[1]-(dim-1)/2)**2))
    order = corners + rem

    for i in range(len(face_data)):
        entry, (gr, gc) = face_data[i], order[i]
        
        # Proportional Stochastic Scaling
        roll = random.random()
        scale = random.uniform(*p['s_small']) if roll < p['p_small'] else (random.uniform(*p['s_large']) if roll > (1-p['p_large']) else random.uniform(*p['s_med']))
        nh, nw = int(entry['img'].shape[0] * (p['cell_size']/200) * scale), int(entry['img'].shape[1] * (p['cell_size']/200) * scale)
        if nh < 10 or nw < 10: continue
        
        f_res, m_res = transform.resize(entry['img'], (nh, nw)), transform.resize(entry['mask'], (nh, nw))
        
        # Soften edges
        alpha = filters.gaussian(apply_edge_fade(m_res, p['fade_r']), sigma=p['alpha_sigma'])[..., None]
        
        jit = int(p['cell_size'] * p['jitter'])
        y_pos = int(gr * p['cell_size'] + random.randint(-jit, jit) + (p['cell_size'] - nh)//2)
        x_pos = int(gc * p['cell_size'] + random.randint(-jit, jit) + (p['cell_size'] - nw)//2)
        
        ys, ye, xs, xe = max(0, y_pos), min(canvas.shape[0], y_pos+nh), max(0, x_pos), min(canvas.shape[1], x_pos+nw)
        fp, ap, bp = f_res[:ye-ys, :xe-xs], alpha[:ye-ys, :xe-xs], canvas[ys:ye, xs:xe]
        if fp.shape == bp.shape: canvas[ys:ye, xs:xe] = (fp * ap) + (bp * (1.0 - ap))

    # 4. Global Polish
    final = apply_vignette(apply_vintage_tone(canvas), p['vig_i'], p['vig_f'])
    io.imsave(p['out_path'], (final * 255).astype(np.uint8))
    print(f"Success! Collage saved to {p['out_path']}")

##################### Main Tuning Logic ######################
if __name__ == "__main__":
    if not os.path.exists("Collage Results"): os.makedirs("Collage Results")
    
    # This dictionary acts as the centralized "Control Room" for the entire script
    collage_params = {
        'target_path': "Extracted Pictures/frame_01.png",
        'out_path': "Collage Results/Artistic_Vintage_Mosaic.png",

        # Face Detection Params
        'det_scale': 1.05, 'det_step': 1.05, 'min_f': (100, 100), 'max_f': (350, 350), 'nms_thresh': 0.4,

        # Segmentation Tuning
        'pad_y': 0.40, 'pad_x': 0.60, 'blur_sigma': 1.2, 'canny_sigma': 1.0, 
        'canny_low': 0.04, 'canny_high': 0.12, 'dilation_size': 7,
        
        # Blending & Visual Polish
        'fade_r': 0.15, 'alpha_sigma': 12.0, 'cell_size': 280, 'jitter': 0.35,
        'p_small': 0.30, 'p_large': 0.30, 's_small': (0.2, 0.8), 's_large': (1.5, 2.0), 's_med': (0.8, 1.5),
        'vig_i': 0.75, 'vig_f': 1.5
    }

    main(collage_params)