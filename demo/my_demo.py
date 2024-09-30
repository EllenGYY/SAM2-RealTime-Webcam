import os
import torch
import numpy as np
from PIL import Image
import cv2
import time
import glob  

from sam2.build_sam import build_sam2_camera_predictor, build_sam2_video_predictor

# Function to get the latest complete JPG file
def get_latest_jpg():
    jpg_files = glob.glob(os.path.join("temp_data", "camera_frames", "*[0-9][0-9][0-9][0-9].jpg"))
    if not jpg_files:
        return None
    
    # Sort files by number in descending order
    sorted_files = sorted(jpg_files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0][-4:]), reverse=True)
    
    for file in sorted_files:
        try:
            img = cv2.imread(file)
            if img is not None:
                return file
        except Exception:
            pass
    
    return None

def wait_for_image(target):
    while True:
        # Look for jpg files ending with 0000
        matching_files = glob.glob(f'temp_data/camera_frames/*{target}.jpg')
        if matching_files:
            print(f"Found image: {matching_files[0]}")
            return matching_files[0]
        else:
            print(f"Waiting for image file {target}")
            time.sleep(1)  # Wait for 1 second before checking again

def add_mask_overlay(frame, out_obj_ids, out_mask_logits):
    height, width = frame.shape[:2]
    # Check mask dimensions
    mask = (out_mask_logits[0] > 0.0).cpu().numpy()    
    if mask.shape[0] == 1:
        mask = mask.squeeze(0)  # Remove the extra dimension
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    red_mask = np.zeros((height, width, 3), dtype=np.uint8)
    red_mask[mask == 1] = [0, 0, 255]  # Red color
    alpha = 0.5  # Transparency factor
    return cv2.addWeighted(frame, 1, red_mask, alpha, 0)


device = torch.device("cuda")
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

file_name = wait_for_image('0000')
frame = cv2.imread(file_name)

# read the first frame
predictor.load_first_frame(frame)
if_init = True

ann_frame_idx = 0  # the frame index we interact with

ann_obj_id = (
    1  # give a unique id to each object we interact with (it can be any integers)
)

points = np.array([[320, 240]], dtype=np.float32) # center of the image

# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], dtype=np.int32)

_, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# out_obj_ids, out_mask_logits = predictor.track(frame)
overlay = add_mask_overlay(frame, out_obj_ids, out_mask_logits)
# Ensure the directory exists
# os.makedirs("temp_data", "input_images", exist_ok=True)

# Get the base filename without path
base_filename = os.path.basename(file_name)
# Construct the full output path
output_path = os.path.join("temp_data", "input_images", f"result_{base_filename}")
# Write the image
cv2.imwrite(output_path, overlay)


# start tracking videos
while 1:
    ann_frame_idx += 1
    # Format the file names with leading zeros
    # target_file = f"{ann_frame_idx:04d}"
    
    # Wait for the image file using the wait_for_image function
    # file_name = wait_for_image(target_file)
    file_name = get_latest_jpg()
    if file_name:
        frame = cv2.imread(file_name)
        # Track objects in the new frame
        out_obj_ids, out_mask_logits = predictor.track(frame)
        overlay = add_mask_overlay(frame, out_obj_ids, out_mask_logits)
        # Construct the full output path
        output_path = os.path.join("temp_data", "input_images", f"result_{os.path.basename(file_name)}")  
        # Write the image
        cv2.imwrite(output_path, overlay)
    else:
        print("Error: no frame read")
        break





