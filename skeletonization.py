import cv2
import mediapipe as mp
import os
import pandas as pd

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Set input folder paths
augmentation_folder = 'augmented_dataset'
dataset_folder = 'dataset'
skeletonized_folder = 'skeletonized_image'
base_directory = os.getcwd()  # Base directory

# Create output folder if it doesn't exist
if not os.path.exists(skeletonized_folder):
    os.makedirs(skeletonized_folder)

# Only use specific landmarks
selected_landmarks = [0, 1, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Landmark connections and their colors
connections = {
    'pink': [(8, 5), (5, 0), (0, 2), (2, 7)],
    'green': [(16, 14), (14, 12), (15, 13), (13, 11)],
    'red': [(12, 11), (12, 24), (11, 23), (24, 23), (12, 23), (11, 24)],
    'blue': [(24, 26), (26, 28), (23, 25), (25, 27)]
}

# Function to get the color based on connection group
def get_color(group):
    if group == 'pink':
        return (147, 20, 255)  # Pink
    elif group == 'green':
        return (0, 255, 0)  # Green
    elif group == 'red':
        return (0, 0, 255)  # Red
    elif group == 'blue':
        return (255, 0, 0)  # Blue

# Function to process and skeletonize images
def process_images(input_folder, output_folder, label_data, prefix=''):
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        for root, dirs, files in os.walk(input_folder):
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)
                output_subdir = os.path.join(output_folder, subdir)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                for filename in os.listdir(subdir_path):
                    if filename.endswith(('.png')):
                        # Image file path
                        image_path = os.path.join(subdir_path, filename)
                        image = cv2.imread(image_path)

                        # Convert BGR image to RGB
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # Extract pose with Mediapipe
                        results = pose.process(image_rgb)

                        # If pose landmarks are detected
                        if results.pose_landmarks:
                            h, w, _ = image.shape
                            # Calculate line thickness proportional to person size
                            thickness = max(1, int(min(h, w) * 0.015))

                            for group, connection_list in connections.items():
                                color = get_color(group)
                                for connection in connection_list:
                                    start_idx, end_idx = connection
                                    if start_idx in selected_landmarks and end_idx in selected_landmarks:
                                        start = results.pose_landmarks.landmark[start_idx]
                                        end = results.pose_landmarks.landmark[end_idx]
                                        start_point = (int(start.x * w), int(start.y * h))
                                        end_point = (int(end.x * w), int(end.y * h))
                                        cv2.line(image, start_point, end_point, color, thickness)

                            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                                if idx in selected_landmarks:
                                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                                    radius = max(1, int(thickness * 0.8))  # Point size proportional to line thickness
                                    cv2.circle(image, (cx, cy), radius, (0, 255, 255), -1)  # Yellow circle

                            # Set output file path
                            output_filename = prefix + filename
                            output_path = os.path.join(output_subdir, output_filename)

                            # Save the result image
                            cv2.imwrite(output_path, image)
                            label_data.append([os.path.relpath(output_path, base_directory), subdir])

# Create label file
def create_label_file(label_data, output_label_file):
    df = pd.DataFrame(label_data, columns=['file_path', 'label'])
    df.to_csv(output_label_file, index=False)

# Initialize label data list
label_data = []

# Process images from augmentation_images and dataset folders, and save skeletonized images in preprocessing folder
process_images(augmentation_folder, skeletonized_folder, label_data)
process_images(dataset_folder, skeletonized_folder, label_data)

# Save the final label file in the base directory
skeletonized_label_file = os.path.join(base_directory, 'skeletonized_labels.csv')
create_label_file(label_data, skeletonized_label_file)
