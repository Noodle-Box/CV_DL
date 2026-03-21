import subprocess
import os
# --- Configuration --- #
input_video_path = "/content/sample_video.mp4" # Replace with your video file path
output_folder = "/content/extracted_frames" # Folder to save frames
frame_rate = 1 # Extract 1 frame per second (videos have ususal 30fps or 25fps)
image_naming_pattern = "frame_%04d.png" # Naming convention for output images
# --- Create output folder if it doesn't exist --- #
os.makedirs(output_folder, exist_ok=True)
# --- FFmpeg command to extract frames --- #
# -i: input file
# -vf fps=N: extracts N frames per second
# -q:v 2: sets video quality (2 is high quality, 31 is low)
# os.path.join: safely creates the output path
ffmpeg_command = [
"ffmpeg",
"-i", input_video_path,
"-vf", f"fps={frame_rate}",