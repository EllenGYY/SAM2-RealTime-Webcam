import cv2
import os
import time
import glob
import numpy as np
from pygrabber.dshow_graph import FilterGraph

graph = FilterGraph()

print(graph.get_input_devices())# list of camera device 
device = graph.get_input_devices().index("OBS Virtual Camera")
print(device)

device = 0
# Create a folder to store the images
output_folder = os.path.join("temp_data", "camera_frames")
os.makedirs(output_folder, exist_ok=True)

# Folder to read PNG files from
input_folder = os.path.join("temp_data", "input_images")
os.makedirs(input_folder, exist_ok=True)
previous_jpg = None

# Function to get the latest JPG file
def get_latest_jpg(folder, previous_result=None):
    jpg_files = glob.glob(os.path.join(folder, "*[0-9][0-9][0-9][0-9].jpg"))
    if not jpg_files:
        return previous_result
    latest_file = max(jpg_files, key=lambda f: int(os.path.splitext(f)[0][-4:]))
    
    # Check if the file is complete and valid
    try:
        img = cv2.imread(latest_file)
        if img is None:
            return previous_result
        return latest_file
    except Exception:
        return previous_result

def cleanup_old_files(folder, max_age_seconds=10):
    current_time = time.time()
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                os.remove(file_path)

# Initialize the camera
cap = cv2.VideoCapture(device)  # 0 for default camera, change if using a different camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

frame_count = 0
saving_enabled = False

try:
    while True:
        key = cv2.waitKey(1) & 0xFF
        # Press 'q' to quit
        if key == ord('q'):
            print("Cleaning up files before exiting...")
            cleanup_old_files(output_folder, max_age_seconds=0)
            cleanup_old_files(input_folder, max_age_seconds=0)
            print("Cleanup complete.")
            break
        elif key == ord('s') and not saving_enabled:
            print("here")
            saving_enabled = True

        cleanup_old_files(output_folder)
        cleanup_old_files(input_folder)

        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame.")
            break
        if saving_enabled:
            # Generate a filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{output_folder}/frame_{timestamp}_{frame_count:04d}.jpg"

            # Write the frame as an image file
            cv2.imwrite(filename, frame)

            frame_count += 1

            # Calculate the time elapsed since the start of this iteration
            elapsed_time = time.time() - start_time

            # Calculate the time to wait to achieve 30 fps
            wait_time = max(1.0/30 - elapsed_time, 0)
            time.sleep(wait_time)

            # Get the latest PNG file from the input folder
            latest_jpg = get_latest_jpg(input_folder, previous_jpg)
            
            if latest_jpg:
                second_frame = cv2.imread(latest_jpg)
                # Resize second_frame to match the dimensions of the camera frame
                second_frame = cv2.resize(second_frame, (frame.shape[1], frame.shape[0]))
                previous_jpg = latest_jpg
            else:
                # If no PNG file found, use a blank frame
                second_frame = np.zeros_like(frame)
        else:
            center = (frame.shape[1] // 2, frame.shape[0] // 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            second_frame = np.zeros_like(frame)
        # Display the frames side by side
        combined_frame = cv2.hconcat([frame, second_frame])
        cv2.imshow('Camera Feed and Latest Result', combined_frame)

        # Adjust window size to fit both frames
        cv2.resizeWindow('Camera Feed and Latest Result', frame.shape[1] * 2, frame.shape[0])

except KeyboardInterrupt:
    print("Capturing stopped by user.")

finally:
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

print(f"Total frames captured: {frame_count}")
