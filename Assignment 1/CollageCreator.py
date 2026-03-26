import os
import numpy as np
from skimage import io, color, data, feature, filters, morphology, measure, exposure
from scipy import ndimage
from skimage.util import img_as_ubyte

# =====================================================================
# --- HIGH-PRECISION DETECTION PARAMETERS ---
# =====================================================================
MIN_FACE_SIZE = (30, 30)     
MAX_FACE_SIZE = (450, 450)   
SCALE_FACTOR = 1.05          
STEP_RATIO = 1.05            

# =====================================================================
# --- YOUR 4-STEP PIPELINE PARAMETERS ---
# =====================================================================
PADDING_PERCENT = 0.40       

# 1. Adjustable Gradients (The "Light Switch")
#GAMMA_CORRECTION = 0.5       # <--- CHANGE THIS (Try 0.3 or 0.25 to lighten dark hair)

# 2. Gaussian Blurring
BLUR_SIGMA = 1.0             

# 3. Edge Detection (The "Edge Net")
CANNY_SIGMA = 1.0            
CANNY_LOW = 0.02             # <--- CHANGE THIS (Try 0.01 to catch fainter hair lines)
CANNY_HIGH = 0.08            # <--- CHANGE THIS (Try 0.05)

# 4. Segmentation (The "Caulk")
DILATION_SIZE = 8            # <--- CHANGE THIS (Try 10 or 12 to plug boundary leaks)

def segment_pipeline(face_crop_color, blur_sigma, canny_sigma, dilation_size):
    """
    Executes the strict 4-step pipeline using Histogram Equalization 
    to dynamically maximize contrast before edge detection.
    """
    if face_crop_color.size == 0:
        return None

    # --- STEP 1: Grayscale & Histogram Equalization (Lecture 1) ---
    gray = color.rgb2gray(face_crop_color)
    
    # This automatically flattens the pixel distribution. 
    # Dark hair and dark backgrounds will be forced apart into distinct shades.
    enhanced_gray = exposure.equalize_hist(gray)

    # --- STEP 2: Gaussian Blurring (Lecture 1/2) ---
    smoothed = filters.gaussian(enhanced_gray, sigma=blur_sigma)

    # --- STEP 3: Edge Detection (Lecture 2) ---
    edges = feature.canny(smoothed, sigma=canny_sigma, 
                          low_threshold=CANNY_LOW, 
                          high_threshold=CANNY_HIGH)

    # --- STEP 4: Face, Hair, Neck Segmentation (Lecture 3) ---
    
    # 1. Seal the neck at the bottom of the frame
    edges[-1, :] = True

    # 2. Dilate to bridge gaps
    footprint = morphology.disk(dilation_size)
    closed_edges = morphology.dilation(edges, footprint)

    # 3. The Frame Boundary Fix (Keep corners open)
    closed_edges[0, :] = False   
    closed_edges[:, 0] = False   
    closed_edges[:, -1] = False  

    # 4. Fill the interior
    filled = ndimage.binary_fill_holes(closed_edges)

    labeled = measure.label(filled)
    
    # 5. Center Prior
    h, w = filled.shape
    center_label = labeled[h//2, w//2]
    
    if center_label != 0:
        local_mask = (labeled == center_label)
    else:
        regions = measure.regionprops(labeled)
        if not regions:
            return None
        largest_region = max(regions, key=lambda r: r.area)
        local_mask = (labeled == largest_region.label)
    
    # 6. Erosion to remove the halo
    local_mask = morphology.erosion(local_mask, morphology.disk(dilation_size - 2))
    
    # Final internal polish
    local_mask = ndimage.binary_fill_holes(local_mask)

    return local_mask

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

    print(f"Sweeping frame for all faces...")
    detected_faces = detector.detect_multi_scale(img=gray_img,
                                                  scale_factor=SCALE_FACTOR,
                                                  step_ratio=STEP_RATIO,
                                                  min_size=MIN_FACE_SIZE,
                                                  max_size=MAX_FACE_SIZE)

    if not detected_faces:
        print("Error: No faces detected.")
        return
     
     # PARAMETER TUNING STUFF
    test_face = [detected_faces[57]]
    


#     num_faces = len(detected_faces)
#     print(f"SUCCESS! Detected {num_faces}/36 faces.")
#     print("Executing Adjusted 4-Step Edge Pipeline...")

 

    for i, face in enumerate(test_face):
        r, c, w, h = face['r'], face['c'], face['width'], face['height']

        pad_r = int(h * PADDING_PERCENT)
        pad_c = int(w * PADDING_PERCENT)
        
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

    print("Compositing final image...")
    white_bg = np.ones_like(img) * 255
    final_composite = np.where(global_mask[..., None], img, white_bg)

    io.imsave(output_path, img_as_ubyte(final_composite)) 
    print(f"Done. Pipeline composite saved as '{output_path}'.")

if __name__ == "__main__":
    main()