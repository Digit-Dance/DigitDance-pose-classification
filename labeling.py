import os
import pandas as pd

def create_label_file(image_dir, output_csv):
    data = []
    # Get the absolute path of the image directory
    base_dir = os.path.abspath(image_dir)

    for label in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, label)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.endswith('.png'):
                    # Get the absolute path of each image file, then create a relative path based on the project directory
                    path =  os.path.join(class_dir, img_file)
                    data.append([path, label])

    df = pd.DataFrame(data, columns=['file_path', 'label'])
    
    # Create the utils folder if it does not exist
    if not os.path.exists(os.path.dirname(output_csv)):
        os.makedirs(os.path.dirname(output_csv))
    
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    # Use relative paths based on the current directory
    current_directory = os.path.dirname(__file__)
    image_directory = os.path.join(current_directory, 'dataset')
    output_csv_file = os.path.join(current_directory, 'labels.csv')  # Save labels.csv in the utils folder
    create_label_file(image_directory, output_csv_file)
