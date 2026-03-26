import os
import numpy as np
from skimage import io, color, data, feature, filters, morphology, measure, exposure
from scipy import ndimage
from skimage.util import img_as_ubyte

# =====================================================================
# --- NON-MAXIMUM SUPPRESSION (IoM UPGRADE) ---
# =====================================================================
def non_max_suppression(boxes, overlapThresh=0.4):
    """
    Upgraded NMS using Intersection over Minimum Area (IoM).
    Crushes false positives nested inside larger bounding boxes.
    """
    if len(boxes) == 0:
        return []

    r = np.array([b['r'] for b in boxes], dtype=float)
    c = np.array([b['c'] for b in boxes], dtype=float)
    w = np.array([b['width'] for b in boxes], dtype=float)
    h = np.array([b['height'] for b in boxes], dtype=float)

    y1 = r
    x1 = c
    y2 = r + h
    x2 = c + w
    
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

        # Intersection over Minimum Area (IoM)
        area_overlap = w_int * h_int
        area_min = np.minimum(area[i], area[idxs[:last]])
        overlap = area_overlap / (area_min + 1e-5) 

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return pick

# =====================================================================
# --- HIGH-PRECISION DETECTION PARAMETERS (The Golden Set) ---
# =====================================================================
MIN_FACE_SIZE = (100, 100)   # Floor: Kills floating bookshelf/collar boxes
MAX_FACE_SIZE = (350, 350)   # Ceiling: Kills giant watermark boxes
SCALE_FACTOR = 1.05          # Fine sweep for reliability
STEP_RATIO = 1.05            

# =====================================================================
# --- YOUR 4-STEP PIPELINE PARAMETERS ---
# =====================================================================
PADDING_PERCENT_Y = 0.30
PADDING_PERCENT_X = 0.60
BLUR_SIGMA = 1.0             
CANNY_SIGMA = 1.0            
CANNY_LOW = 0.06             
CANNY_HIGH = 0.15            
DILATION_SIZE = 8            

def segment_pipeline(face_crop_color, blur_sigma, canny_sigma, dilation_size):
    if face_crop_color.size == 0:
        return None

    # --- UPDATED SPECTRAL FILTERING ---
    hsv = color.rgb2hsv(face_crop_color)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    # Loosened range for darker hair
    skin_hue = ((h < 0.16) | (h > 0.70))
    chromatic_pass = skin_hue & (s > 0.10) & (v > 0.15)
    dark_features_pass = (v < 0.35) # Catch all deep shadows/hair

    # green be gone
    is_green = (h > 0.18) & (h < 0.50) 
    
    color_mask = (chromatic_pass | dark_features_pass) & ~is_green

    # green be gone
    is_green = (h > 0.18) & (h < 0.50) 
    
    # =====================================================================
    # --- CRITICAL FIX 1: PATCH THE INTERNAL HOLES ---
    # We fill the holes in the COLOR MASK before intersecting with edges.
    # This keeps eyes, teeth, and internal shadows intact.
    # =====================================================================
    color_mask = ndimage.binary_fill_holes(color_mask)
    
    # --- CONTINUE PIPELINE ---
    gray = color.rgb2gray(face_crop_color)
    enhanced_gray = exposure.equalize_hist(gray)
    smoothed = filters.gaussian(enhanced_gray, sigma=blur_sigma)

    edges = feature.canny(smoothed, sigma=canny_sigma, 
                          low_threshold=0.01, 
                          high_threshold=0.05)

    # Intersect edges with the now-solid color mask
    edges = edges & color_mask

    # Standard Segmentation logic...
    edges[-1, :] = True
    closed_edges = morphology.dilation(edges, morphology.disk(dilation_size))
    
    # Boundary cleanup
    closed_edges[0, :] = False; closed_edges[:, 0] = False; closed_edges[:, -1] = False  

    filled = ndimage.binary_fill_holes(closed_edges)
    
    # Bridge breaking...
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
    
    # Patch final local holes and enforce the color check
    local_mask = ndimage.binary_fill_holes(local_mask)
    return local_mask & color_mask

def main():
    target_image = "Extracted Pictures/frame_01.png"
    output_path = "All_Faces_Pipeline_Isolated.png"

    if not os.path.exists(target_image):
        print(f"Error: Could not find '{target_image}'.")
        return

    print(f"Loading image for processing: {target_image}")
    img = io.imread(target_image)
    gray_img = color.rgb2gray(img)

    global_mask = np.zeros(img.shape[:2], dtype=bool)

    trained_file = data.lbp_frontal_face_cascade_filename()
    detector = feature.Cascade(trained_file)

    print(f"Sweeping frame for raw faces...")
    raw_faces = detector.detect_multi_scale(img=gray_img,
                                            scale_factor=SCALE_FACTOR,
                                            step_ratio=STEP_RATIO,
                                            min_size=MIN_FACE_SIZE,
                                            max_size=MAX_FACE_SIZE)

    if not raw_faces:
        print("Error: No faces detected.")
        return
     
    # Apply the tuned NMS logic
    filtered_faces = non_max_suppression(raw_faces, overlapThresh=0.4)
    num_faces = len(filtered_faces)
    
    print(f"SUCCESS! Filtered down to {num_faces} faces.")
    print("Executing Adjusted 4-Step Edge Pipeline on all faces...")

    # Reverted to process all filtered faces
    for i, face in enumerate(filtered_faces):
        r, c, w, h = face['r'], face['c'], face['width'], face['height']

        pad_r = int(h * PADDING_PERCENT_Y)
        pad_c = int(w * PADDING_PERCENT_X)
        
        r_start = max(0, r - pad_r)
        c_start = max(0, c - pad_c)
        r_end = min(img.shape[0], r + h + pad_r)
        c_end = min(img.shape[1], c + w + pad_c)

        face_crop_color = img[r_start:r_end, c_start:c_end]

        local_mask = segment_pipeline(face_crop_color, BLUR_SIGMA, CANNY_SIGMA, DILATION_SIZE)

        if local_mask is not None:
            global_mask[r_start:r_end, c_start:c_end] |= local_mask
        else:
            print(f"Warning: Segmentation failed for Face {i+1}.")

    print("Feathering edges and compositing final image...")
    
    # =====================================================================
    # --- FEATHERING & ALPHA BLENDING (Smoothing the Background) ---
    # Convert boolean mask to float to allow for soft edges
    # Apply a slight blur to create a gradient edge transition
    # =====================================================================
    float_mask = global_mask.astype(float)
    smoothed_mask = filters.gaussian(float_mask, sigma=1.5)
    
    white_bg = np.ones_like(img) * 255.0
    
    # Mathematically blend the original image and white background using the soft mask
    final_composite = (img * smoothed_mask[..., None] + white_bg * (1.0 - smoothed_mask[..., None]))
    
    # Clip bounds to ensure valid pixel values before saving
    final_composite = np.clip(final_composite, 0, 255).astype(np.uint8)

    io.imsave(output_path, final_composite) 
    print(f"Done. Smooth pipeline composite saved as '{output_path}'.")

if __name__ == "__main__":
    main()