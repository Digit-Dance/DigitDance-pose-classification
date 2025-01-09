import os
import pandas as pd
import albumentations as A
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
import imageio
from PIL import Image


def augment_image(image_path, output_dir, base_dir, label, num_augments=2):
    # Convert to absolute path
    absolute_image_path = os.path.join(base_dir, image_path)
    if not os.path.exists(absolute_image_path):
        return []

    try:
        image = imageio.v2.imread(absolute_image_path)
    except Exception as e:
        print(f"Error reading image {absolute_image_path}: {e}")
        return []

    # Define augmentation sequence
    augmentation = A.Compose([
        A.Rotate(limit=30, p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.SaltAndPepper(p=0.3)
    ])

    augmented_files = []
    for i in range(num_augments):
        augmented = augmentation(image=image)
        aug_image = augmented["image"]
        
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        aug_image_path = os.path.join(output_dir, f"{label}_{name}_aug_{i}{ext}")

        try:
            Image.fromarray(aug_image).save(aug_image_path)
            augmented_files.append(aug_image_path)
        except Exception as e:
            print(f"Error saving augmented image {aug_image_path}: {e}")

    return augmented_files


def augment_dataset(label_file, output_base_dir, base_dir, num_augments=2):
    try:
        df = pd.read_csv(label_file)
    except Exception as e:
        print(f"Error reading label file {label_file}: {e}")
        return

    augmented_data = []
    for index, row in df.iterrows():
        # Extract the folder name from the file path
        folder_name = os.path.basename(os.path.dirname(row['file_path']))
        output_dir = os.path.join(output_base_dir, folder_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        augmented_files = augment_image(row['file_path'], output_dir, base_dir, row['label'], num_augments)
        for file in augmented_files:
            augmented_data.append([file, row['label']])

    augmented_df = pd.DataFrame(augmented_data, columns=['file_path', 'label'])
    augmented_label_file = os.path.join(os.path.dirname(label_file), 'augmented_labels.csv')
    augmented_df.to_csv(augmented_label_file, index=False)
    print(f"Augmented labels saved to {augmented_label_file}")


if __name__ == "__main__":
    # Use relative paths based on the current directory
    current_directory = os.path.dirname(__file__)
    label_file_path = os.path.join(current_directory, 'labels.csv')
    image_base_directory = os.path.join(current_directory, 'dataset')
    augmented_directory = os.path.join(current_directory, 'augmented_dataset')

    # Run the dataset augmentation
    augment_dataset(label_file_path, augmented_directory, image_base_directory)
