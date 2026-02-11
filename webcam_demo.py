import torch
import numpy as np
import cv2
import time
from sam2.build_sam import build_sam2_camera_predictor

def add_mask_overlay(frame, out_obj_ids, out_mask_logits):
    height, width = frame.shape[:2]
    # Check mask dimensions
    if len(out_mask_logits) == 0:
        return frame
        
    mask = (out_mask_logits[0] > 0.0).cpu().numpy()
    if mask.shape[0] == 1:
        mask = mask.squeeze(0)  # Remove the extra dimension
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    
    red_mask = np.zeros((height, width, 3), dtype=np.uint8)
    red_mask[mask == 1] = [0, 0, 255]  # Red color
    alpha = 0.5  # Transparency factor
    return cv2.addWeighted(frame, 1, red_mask, alpha, 0)

def main():
    # Configuration
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Initialize SAM2
    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    try:
        predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
    except Exception as e:
        print(f"Error loading SAM2 model: {e}")
        return

    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully.")
    print("Press 's' to start tracking (click center of image).")
    print("Press 'q' to quit.")

    is_tracking = False
    ann_frame_idx = 0
    ann_obj_id = 1
    
    # Default point: Center of the image. 
    # Logic will update this based on frame size once camera is open.
    points = None 
    labels = np.array([1], dtype=np.int32) # Positive click

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break

            display_frame = frame.copy()

            if is_tracking:
                try:
                    out_obj_ids, out_mask_logits = predictor.track(frame)
                    display_frame = add_mask_overlay(display_frame, out_obj_ids, out_mask_logits)
                except Exception as e:
                     print(f"Tracking error: {e}")
                     is_tracking = False

            cv2.imshow('SAM2 Webcam Demo', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if not is_tracking:
                    print("Starting tracking...")
                    # Update points to center of current frame
                    height, width = frame.shape[:2]
                    points = np.array([[width // 2, height // 2]], dtype=np.float32)
                    
                    predictor.load_first_frame(frame)
                    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        points=points,
                        labels=labels,
                    )
                    is_tracking = True
                else:
                    print("Already tracking. Restarting...")
                    is_tracking = False # Simple reset logic: stop then start again on next 's'
                    # Ideally we would reset state here properly if needed, 
                    # but for this demo, let's just allow re-initialization on next 's' press 
                    # or maybe just reset completely.
                    # For now, let's just re-trigger the start logic immediately:
                    height, width = frame.shape[:2]
                    points = np.array([[width // 2, height // 2]], dtype=np.float32)
                    predictor.load_first_frame(frame)
                    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        points=points,
                        labels=labels,
                    )
                    is_tracking = True


    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
