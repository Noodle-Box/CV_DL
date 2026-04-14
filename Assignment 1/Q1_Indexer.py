######################### Helper Function for Assignment 1 #########################

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, color, data, feature

# 1. High-Precision Parameters (Matches your main script)
MIN_FACE_SIZE = (30, 30)     
MAX_FACE_SIZE = (450, 450)   
SCALE_FACTOR = 1.05          
STEP_RATIO = 1.05            

target_image = "Extracted Pictures/frame_01.png"
img = io.imread(target_image)
gray_img = color.rgb2gray(img)

trained_file = data.lbp_frontal_face_cascade_filename()
detector = feature.Cascade(trained_file)

print("Sweeping frame to generate Index Map...")
detected_faces = detector.detect_multi_scale(img=gray_img,
                                              scale_factor=SCALE_FACTOR,
                                              step_ratio=STEP_RATIO,
                                              min_size=MIN_FACE_SIZE,
                                              max_size=MAX_FACE_SIZE)

# 2. Plotting the Image and Indices
fig, ax = plt.subplots(figsize=(16, 12)) # Large figure for visibility
ax.imshow(img)

for i, face in enumerate(detected_faces):
    r, c, w, h = face['r'], face['c'], face['width'], face['height']
    
    # Draw Bounding Box
    rect = patches.Rectangle((c, r), w, h, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    
    # Draw Index Number above the box
    ax.text(c, r - 15, f"Index: {i}", color='yellow', fontsize=12, weight='bold', 
            bbox=dict(facecolor='red', alpha=0.7, edgecolor='none', pad=2))

plt.title(f"Face Index Map (Total: {len(detected_faces)})")
plt.axis('off')

# Save and show
plt.savefig("Face_Index_Map.png", bbox_inches='tight')
plt.show()