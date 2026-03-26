import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, color, data, feature

# =====================================================================
# --- NON-MAXIMUM SUPPRESSION (NMS) ALGORITHM ---
# =====================================================================
def non_max_suppression(boxes, overlapThresh=0.3):
    """
    Filters out overlapping bounding boxes using Intersection over Union (IoU).
    If two boxes overlap by more than overlapThresh (e.g., 30%), the redundant 
    one is deleted.
    """
    if len(boxes) == 0:
        return []

    # Extract coordinates from the list of dictionaries
    r = np.array([b['r'] for b in boxes], dtype=float)
    c = np.array([b['c'] for b in boxes], dtype=float)
    w = np.array([b['width'] for b in boxes], dtype=float)
    h = np.array([b['height'] for b in boxes], dtype=float)

    # Convert to standard bounding box coordinates (top-left, bottom-right)
    y1 = r
    x1 = c
    y2 = r + h
    x2 = c + w
    
    # Calculate area of all boxes
    area = w * h
    
    # Sort boxes by the bottom-right y-coordinate
    idxs = np.argsort(y2)
    pick = []

    while len(idxs) > 0:
        # Grab the last index in the sorted list and add to the pick list
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(boxes[i])

        # Find the largest (x, y) coordinates for the start of the bounding box
        # and the smallest (x, y) coordinates for the end
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box overlap
        w_int = np.maximum(0, xx2 - xx1)
        h_int = np.maximum(0, yy2 - yy1)

        # THE UPGRADE: Intersection over Minimum Area (IoM)
        area_overlap = w_int * h_int
        area_min = np.minimum(area[i], area[idxs[:last]])
        
        # Prevent division by zero just in case
        overlap = area_overlap / (area_min + 1e-5)

        # Delete all indexes from the list that have an overlap greater than the threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return pick

# =====================================================================
# --- MAIN EXECUTION ---
# =====================================================================
def main():
    # 1. High-Precision Parameters
    MIN_FACE_SIZE = (89, 89)     
    MAX_FACE_SIZE = (350, 350)   
    SCALE_FACTOR = 1.05          
    STEP_RATIO = 1.05            

    # Update this to dynamically point to whichever frame you want to analyze
    target_image = "Extracted Pictures/frame_01.png" 
    
    if not os.path.exists(target_image):
        print(f"Error: Could not find '{target_image}'.")
        return

    img = io.imread(target_image)
    gray_img = color.rgb2gray(img)

    trained_file = data.lbp_frontal_face_cascade_filename()
    detector = feature.Cascade(trained_file)

    print("Sweeping frame for raw detections...")
    raw_faces = detector.detect_multi_scale(img=gray_img,
                                            scale_factor=SCALE_FACTOR,
                                            step_ratio=STEP_RATIO,
                                            min_size=MIN_FACE_SIZE,
                                            max_size=MAX_FACE_SIZE)

    print(f"Found {len(raw_faces)} raw overlapping boxes. Applying NMS...")
    
    # 2. Pass the raw detections through the NMS filter
    # If it detects 35 or 37 faces instead of 36, tweak the overlapThresh slightly (e.g., 0.25 or 0.35)
    filtered_faces = non_max_suppression(raw_faces, overlapThresh=0.1)
    
    print(f"SUCCESS! Filtered down to {len(filtered_faces)} distinct faces.")

    # 3. Plotting the Image and Indices
    fig, ax = plt.subplots(figsize=(16, 12)) 
    ax.imshow(img)

    for i, face in enumerate(filtered_faces):
        r, c, w, h = face['r'], face['c'], face['width'], face['height']
        
        # Draw Bounding Box
        rect = patches.Rectangle((c, r), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Draw Index Number above the box
        ax.text(c, r - 15, f"Index: {i}", color='yellow', fontsize=12, weight='bold', 
                bbox=dict(facecolor='red', alpha=0.7, edgecolor='none', pad=2))

    plt.title(f"Dynamic Face Index Map (Total Faces: {len(filtered_faces)})")
    plt.axis('off')

    output_map = "Face_Index_Map_Filtered.png"
    plt.savefig(output_map, bbox_inches='tight')
    print(f"Index Map saved as '{output_map}'.")
    plt.show()

if __name__ == "__main__":
    main()