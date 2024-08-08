import os
import pandas as pd
import imgaug.augmenters as iaa
import imageio

def augment_image(image_path, output_dir, base_dir, label, num_augments=2):
    # Convert to absolute path
    absolute_image_path = os.path.join(base_dir, image_path)
    if not os.path.exists(absolute_image_path):
        return []
    
    image = imageio.imread(absolute_image_path)
    
    # Define augmentation sequence
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-30, 30)),
        iaa.Affine(scale=(0.8, 1.2)),
        iaa.SaltAndPepper(0.1)
    ])
    
    augmented_files = []
    for i in range(num_augments):
        aug_image = seq(image=image)
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        # Add label to the file name to make it unique
        aug_image_path = os.path.join(output_dir, f"{label}_{name}_aug_{i}{ext}")
        imageio.imwrite(aug_image_path, aug_image)
        augmented_files.append(aug_image_path)
    
    return augmented_files

def augment_dataset(label_file, output_base_dir, base_dir, num_augments=2):
    df = pd.read_csv(label_file)
    
    augmented_data = []
    for index, row in df.iterrows():
        # Extract the folder number from the file path
        folder_name = os.path.basename(os.path.dirname(row['file_path']))
        output_dir = os.path.join(output_base_dir, folder_name)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        augmented_files = augment_image(row['file_path'], output_dir, base_dir, row['label'], num_augments)
        for file in augmented_files:
            augmented_data.append([file, row['label']])
    
    augmented_df = pd.DataFrame(augmented_data, columns=['file_path', 'label'])
    # Save the augmented data to CSV in the current directory
    augmented_label_file = os.path.join(os.path.dirname(label_file), 'augmented_labels.csv')
    augmented_df.to_csv(augmented_label_file, index=False)

if __name__ == "__main__":
    # Use relative paths based on the current directory
    current_directory = os.path.dirname(__file__)
    label_file_path = os.path.join(current_directory, 'labels.csv')
    # Set image_base_directory to the dataset folder
    image_base_directory = os.path.join(current_directory, 'dataset')
    augmented_directory = os.path.join(current_directory, 'augmented_dataset')  # Store augmented images

    # Run the dataset augmentation
    augment_dataset(label_file_path, augmented_directory, image_base_directory)