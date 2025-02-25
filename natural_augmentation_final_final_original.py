import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from pathlib import Path
import re
import shutil
import random
import math

# Importing the excel sheet with diameter(mean) values
df = pd.read_excel(r'C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\toolweardiameter.xlsx') # Dataframe for excel sheet

diameter = df['mean'] # Diameter values

def categorize_diameter(value):
    if value < 10:
        return 'fine'
    elif value >10 and value < 15:
        return 'mild'
    else:
        return 'severe'

df['category'] = diameter.apply(categorize_diameter) # creating a new column for category

for index, row in df.iterrows():
    print(f"Row {index}: Mean = {row['mean']}, Category = {row['category']}") # prints for type of category

df.to_excel(r'C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\updated_toolweardiameter.xlsx', index=False) # This will create a new excel sheet with updated category

# Importing the excel sheet with toolname and category
df_2 = pd.read_excel(r'C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\updated_toolweardiameter.xlsx')

for index, row in df.iterrows():
    print(f"Row {index}: Toolname = {row['toolname']}, Category = {row['category']}")

df_2['tool_category'] = df_2['toolname'].astype(str) + ':' + df_2['category']
df_2.to_excel(r'C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\updated_toolweardiameter.xlsx')
# Updating the names of the original_images in the folder with tool:category
df_3 = pd.read_excel(r'C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\updated_toolweardiameter.xlsx')
original_dataset_images = r'C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\dataset_original'

image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

# Get a sorted list of image filenames in the folder
images = sorted([
    f for f in os.listdir(original_dataset_images)
    if Path(f).suffix.lower() in image_extensions
])

# Check if the number of images matches the number of tool_category entries
if len(images) != len(df_3['tool_category']):
    print(f"Error: Number of images ({len(images)}) does not match number of tool_category entries ({len(df_3['tool_category'])}).")
    exit(1)

for img, new_name in zip(images, df_3['tool_category']):
    sanitized_name = re.sub(r'[\\/*?:"<>|]', '_', str(new_name))# Replace illegal characters with an underscore
    ext = Path(img).suffix # Get the file extension
    new_filename = f"{sanitized_name}{ext}" # Create the new filename

    # Full paths
    src = os.path.join(original_dataset_images, img) # Source path
    dst = os.path.join(original_dataset_images, new_filename) # Destination path

    # Rename the file
    os.rename(src, dst)
    print(f"Renamed {img} to {new_filename}")

print("All images renamed successfully.")

def organize_files(directory):
    """
    Organizes files in the specified directory by moving them into subfolders
    based on the substring after the first underscore in their filenames.
    """
    # Ensure the provided directory exists
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return
    # Iterating over all the files in original_dataset_images   
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            name_part, separator, suffix_part = filename.partition("_") # Split the filename into two parts at the first underscore

            if separator:
                suffix, extension = os.path.splitext(suffix_part) # Split the suffix into the suffix and extension
                suffix_lower = suffix.lower() 

                if suffix_lower == 'fine':
                    category_folder = os.path.join(directory, 'fine_original') # new folder fine_original

                    # Create the category folder if it doesn't exist
                    if not os.path.exists(category_folder):
                        try:
                            os.makedirs(category_folder)
                            print(f"Created folder: {category_folder}")
                        except Exception as e:
                            print(f"Error creating folder '{category_folder}': {e}")
                            continue  # Skip moving this file
                    
                    # Set the destination path
                    destination = os.path.join(category_folder, filename)

                    # Move the file
                    try:
                        shutil.move(file_path, destination)
                        print(f"Moved '{filename}' to '{suffix_lower}/'")
                    except Exception as e:
                        print(f"Error moving file '{filename}': {e}")

                if suffix_lower == 'mild':
                    category_folder = os.path.join(directory, 'mild_original') # new folder mild_original

                    # Create the category folder if it doesn't exist
                    if not os.path.exists(category_folder):
                        try:
                            os.makedirs(category_folder)
                            print(f"Created folder: {category_folder}")
                        except Exception as e:
                            print(f"Error creating folder '{category_folder}': {e}")
                            continue  # Skip moving this file
                    
                    # Set the destination path
                    destination = os.path.join(category_folder, filename)

                    # Move the file
                    try:
                        shutil.move(file_path, destination)
                        print(f"Moved '{filename}' to '{suffix_lower}/'")
                    except Exception as e:
                        print(f"Error moving file '{filename}': {e}")

                if suffix_lower == 'severe':
                    category_folder = os.path.join(directory, 'severe_original') # new folder severe_original

                    # Create the target folder if it doesn't exist
                    if not os.path.exists(category_folder):
                        try:
                            os.makedirs(category_folder)
                            print(f"Created folder: {category_folder}")
                        except Exception as e:
                            print(f"Error creating folder '{category_folder}': {e}")
                            continue  # Skip moving this file
                    
                    # Set the destination path
                    destination = os.path.join(category_folder, filename)

                    # Move the file
                    try:
                        shutil.move(file_path, destination)
                        print(f"Moved '{filename}' to '{suffix_lower}/'")
                    except Exception as e:
                        print(f"Error moving file '{filename}': {e}")

# Calling the organize_files function
organize_files(r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\dataset_original")

# Dividing category_folder into training and test sets
def split_dataset(original_dir, train_dir, test_dir, train_ratio, seed=None):
    """
    Splits the dataset into training and test sets.
    
    original_dir: The directory containing the original dataset(eg: "dataset_original\fine_original")
    train_dir: The directory to save the training set (eg: "dataset_original\fine_original/train")
    test_dir: The directory to save the test set(eg: "dataset_original\fine_original/test")
    split_ratio: The ratio of the test set size to the original dataset size(0.75 in this case)
    seed: The random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        print(f"Random seed set to {seed}")
    
    # Ensure the original directory is present
    if not os.path.isdir(original_dir):
        print(f"Error: Source directory '{original_dir}' does not exist.")
        return
    
    # Create the training and test directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)  

    # List all files in the source directory
    all_files = [f for f in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, f))]
    total_files = len(all_files)
    print(f"Category '{os.path.basename(original_dir)}': Total files found: {total_files}")

    # Shuffle the files for randomness
    random.shuffle(all_files)

    # Calculate split index
    split_index = int(total_files * train_ratio) # int in python floors the decimal value

    # Split the files
    train_files = all_files[:split_index]
    test_files = all_files[split_index:]
    print(f"Category '{os.path.basename(original_dir)}': Training files count: {len(train_files)}")
    print(f"Category '{os.path.basename(original_dir)}': Testing files count: {len(test_files)}")

    # Move training files
    for file in train_files:
        src_path = os.path.join(original_dir, file)
        dest_path = os.path.join(train_dir, file)
        try:
            shutil.move(src_path, dest_path)
            print(f"Moved '{file}' to '{train_dir}'")
        except Exception as e:
            print(f"Error moving file '{file}' to '{train_dir}': {e}")

    # Move testing files
    for file in test_files:
        src_path = os.path.join(original_dir, file)
        dest_path = os.path.join(test_dir, file)
        try:
            shutil.move(src_path, dest_path)
            print(f"Moved '{file}' to '{test_dir}'")
        except Exception as e:
            print(f"Error moving file '{file}' to '{test_dir}': {e}")

    print(f"Category '{os.path.basename(original_dir)}': Split completed.\n")

# Calling the function split_dataset 3 times, once for each category
split_dataset(r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\dataset_original\fine_original", r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\dataset_original\fine_original/train", r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\dataset_original\fine_original/test", 0.75, seed=42)
split_dataset(r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\dataset_original\mild_original", r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\dataset_original\mild_original/train", r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\dataset_original\mild_original/test", 0.75, seed=42)
split_dataset(r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\dataset_original\severe_original", r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\dataset_original/severe_original/train", r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\dataset_original\severe_original/test", 0.75, seed=42)

# Creating new folders for augmented images inside each categories
def  create_augmented_folders(base_dir):

    subdirectories = ["fine_original", "mild_original", "severe_original"]
    new_folders = ["train_augmented", "test_augmented"]

    # Iterate through each subdirectory
    for subdir in subdirectories:
        current_dir = os.path.join(base_dir, subdir)
        
        if not os.path.isdir(current_dir):
            print(f"Warning: The directory '{current_dir}' does not exist. Skipping...")
            continue  # Skip to the next subdirectory if current one doesn't exist
        
        for folder in new_folders:
            new_folder_path = os.path.join(current_dir, folder)
            try:
                os.makedirs(new_folder_path, exist_ok=True)
                print(f"Folder created: {new_folder_path}")
            except Exception as e:
                print(f"Error creating folder '{new_folder_path}': {e}")

# calling the function create_augmented_folders
create_augmented_folders(r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\dataset_original")

def skip_create_destination_folder(destination_folder):
    if os.path.exists(destination_folder):
        if os.listdir(destination_folder):
            print(f"Destination folder '{destination_folder}' already exists and contains files.")
            print("Proceeding to add more images to it.")
        else:
            print(f"Destination folder '{destination_folder}' already exists and is empty.")
    else:
        print(f"Destination folder '{destination_folder}' does not exist. Please create it before running the script.")
        exit()

# Augmentation functions

# 1. Rotations (6 angles)
def rotate_image(image, angles):
    rotated_images = []
    h, w = image.shape[:2]
    for angle in angles:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h))
        rotated_images.append(rotated)
    return rotated_images

# 2. Horizontal flip
def flip_image(image):
    return cv2.flip(image, 1)  # Flip horizontally

# 3. Adding Gaussian noise (2 levels)
def add_gaussian_noise(image, noise_levels):
    noisy_images = []
    row, col, ch = image.shape
    mean = 0
    for sigma in noise_levels:
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy_image = image + gauss.reshape(row, col, ch)
        noisy_image = np.clip(noisy_image, 0, 255)
        noisy_images.append(noisy_image.astype(np.uint8))
    return noisy_images

# 4. Adjust brightness and contrast (3 variations)
def adjust_brightness_contrast(image, brightness_values, contrast_values):
    bright_contrast_images = []
    for brightness in brightness_values:
        for contrast in contrast_values:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            enhancer = ImageEnhance.Brightness(pil_image)
            image_bright = enhancer.enhance(brightness)

            enhancer = ImageEnhance.Contrast(image_bright)
            image_contrast = enhancer.enhance(contrast)

            bright_contrast_images.append(cv2.cvtColor(np.array(image_contrast), cv2.COLOR_RGB2BGR))
    return bright_contrast_images

# Augmentation pipeline
def augment_image_controlled(image_path, filename, destination_folder, max_augmentations):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    # Apply augmentations with multiple variations
    angles = [-20,-15, -10, -5, 5, 10, 15, 20]  # Different rotation angles
    noise_levels = [10, 15, 20, 25]  # Different levels of Gaussian noise
    brightness_values = [0.8, 0.9, 1.0, 1.1, 1.2]  # Brightness factors
    contrast_values = [0.8, 0.9, 1.0, 1.1, 1.2]  # Contrast factors

    # Perform augmentations
    rotated_images = rotate_image(image, angles) # 8 rotated images
    flipped_image = flip_image(image)  # 1 Flipped image
    noisy_images = add_gaussian_noise(image, noise_levels) # 4 noisy images
    bright_contrast_images = adjust_brightness_contrast(image, brightness_values, contrast_values) # 25 bright/contrast images

    # List to keep track of all augmented images
    augmented_images = []

    # Add rotated images
    for i, rotated in enumerate(rotated_images):
        if len(augmented_images) >= max_augmentations:
            break
        augmented_images.append((rotated, f'{filename}_rotated_{i+1}.bmp'))

    # Add flipped image
    if len(augmented_images) < max_augmentations:
        augmented_images.append((flipped_image, f'{filename}_flipped.bmp'))

    # Add noisy images
    for i, noisy in enumerate(noisy_images):
        if len(augmented_images) >= max_augmentations:
            break
        augmented_images.append((noisy, f'{filename}_noisy_{i+1}.bmp'))

    # Add brightness and contrast images
    for i, bright_contrast in enumerate(bright_contrast_images):
        if len(augmented_images) >= max_augmentations:
            break
        augmented_images.append((bright_contrast, f'{filename}_bright_contrast_{i+1}.bmp'))

    # Save augmented images
    for img, fname in augmented_images:
        save_path = os.path.join(destination_folder, fname)
        success = cv2.imwrite(save_path, img)
        if success:
            print(f"Saved augmented image: {fname}")
        else:
            print(f"Failed to save augmented image: {fname}")

def calculate_augmentations_needed(original_count, target_count):
    additional_images = target_count - original_count
    if additional_images <= 0:
        return 0
    return math.ceil(additional_images / original_count)

def remove_excess_images(destination_folder, target_count):
    """
    Removes excess images from the destination folder to match the target count.

    - destination_folder: Folder from which to remove excess images.
    - target_count: Desired total number of images.
    """
    all_images = [f for f in os.listdir(destination_folder) if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))]
    current_count = len(all_images)
    excess = current_count - target_count
    if excess > 0:
        print(f"Removing {excess} excess images from '{destination_folder}'.")
        for i in range(excess):
            image_to_remove = all_images.pop()  # Removes the last image in the list
            os.remove(os.path.join(destination_folder, image_to_remove))
            print(f"Removed: {image_to_remove}")
    else:
        print("No excess images to remove.")

def main():
    """
    Main function to perform image augmentations for multiple classes and datasets.
    """
    base_dir = r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\dataset_original"

    # Define the classes to process
    classes = ["mild_original", "fine_original", "severe_original"]

    # Define target counts per class and dataset
    target_counts = {
        "mild_original": {
            "train": {"original": 6, "target": 240},
            "test": {"original": 3, "target": 60}
        },
        "fine_original": {
            "train": {"original": 11, "target": 240},  
            "test": {"original": 4, "target": 60}      
        },
        "severe_original": {
            "train": {"original": 6, "target": 240},  
            "test": {"original": 2, "target": 60}      
        }
    }

    for cls in classes:
        print(f"\n--- Processing class: {cls} ---")
        class_dir = os.path.join(base_dir, cls)
        
        if not os.path.isdir(class_dir):
            print(f"Class directory '{class_dir}' does not exist. Skipping.")
            continue

        datasets = ["train", "test"]

        for dataset in datasets:
            print(f"\nProcessing '{dataset}' dataset for class '{cls}':")
            source_folder = os.path.join(class_dir, dataset)
            destination_folder = os.path.join(class_dir, f"{dataset}_augmented")

            # Check if source folder exists
            if not os.path.exists(source_folder):
                print(f"Source folder '{source_folder}' does not exist. Skipping.")
                continue

            # Skip creation of destination folder since it already exists
            skip_create_destination_folder(destination_folder)

            # Get original and target counts
            original_count = target_counts.get(cls, {}).get(dataset, {}).get("original", 0)
            target_count = target_counts.get(cls, {}).get(dataset, {}).get("target", 0)

            print(f"Source folder: {source_folder}")
            print(f"Destination folder: {destination_folder}")
            print(f"Original images: {original_count}, Target images: {target_count}")

            # List of original images
            original_images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))]
            actual_original_count = len(original_images)
            print(f"Number of original images found: {actual_original_count}")

            if actual_original_count == 0:
                print(f"No images found in the source folder '{source_folder}'. Skipping '{dataset}' dataset for class '{cls}'.")
                continue

            if original_count != actual_original_count:
                print(f"Warning: Expected {original_count} original images, but found {actual_original_count}. Proceeding with found images.")
                # Update original_count to actual
                original_count = actual_original_count

            # Calculate augmentations needed
            augmentations_needed = target_count - actual_original_count
            if augmentations_needed <= 0:
                print(f"Target count for '{dataset}' dataset in class '{cls}' is less than or equal to the number of original images. Skipping augmentation.")
                # Optionally, ensure originals are in the destination
                print(f"Copying original images to '{destination_folder}'...")
                for filename in original_images:
                    source_path = os.path.join(source_folder, filename)
                    destination_path = os.path.join(destination_folder, filename)
                    shutil.copy(source_path, destination_path)
                continue

            augmentations_per_image = calculate_augmentations_needed(original_count, target_count)
            print(f"Augmentations needed per image: {augmentations_per_image}")

            # Apply augmentations
            for idx, filename in enumerate(original_images, 1):
                if filename.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(source_folder, filename)
                    filename_without_ext = os.path.splitext(filename)[0]
                    print(f"Augmenting image {idx}/{actual_original_count}: {filename}")
                    augment_image_controlled(image_path, filename_without_ext, destination_folder, augmentations_per_image)

            # Copy original images to the destination folder
            print(f"Copying original images to '{destination_folder}'...")
            for filename in original_images:
                source_path = os.path.join(source_folder, filename)
                destination_path = os.path.join(destination_folder, filename)
                shutil.copy(source_path, destination_path)

            # Verify total images
            total_images = len([f for f in os.listdir(destination_folder) if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))])
            print(f"Total images in '{destination_folder}': {total_images} (Target: {target_count})")

            if total_images < target_count:
                print(f"Note: Only {total_images} images were created, which is less than the target of {target_count}.")
            
                # Calculate how many additional images are needed
                additional_needed = target_count - total_images
                print(f"Adding {additional_needed} additional augmented images.")
                
                # Calculate how many augmentations per original image are needed
                augmentations_per_image = math.ceil(additional_needed / actual_original_count)
                print(f"Additional augmentations needed per image: {augmentations_per_image}")
                
                # Apply additional augmentations
                for idx, filename in enumerate(original_images, 1):
                    if filename.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(source_folder, filename)
                        filename_without_ext = os.path.splitext(filename)[0]
                        print(f"Adding extra augmentations to image {idx}/{actual_original_count}: {filename}")
                        
                        # Apply augmentations with max_augmentations=augmentations_per_image
                        augment_image_controlled(image_path, f"{filename_without_ext}_extra_aug", destination_folder, augmentations_per_image)
                
                # Recalculate total images after additional augmentations
                total_images = len([f for f in os.listdir(destination_folder) if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))])
                print(f"Total images after additional augmentations: {total_images} (Target: {target_count})")
                
                if total_images < target_count:
                    remaining = target_count - total_images
                    print(f"Still {remaining} images short of the target. Adding more augmentations.")
                    
                    # Loop through original images again to add remaining augmentations
                    for idx, filename in enumerate(original_images, 1):
                        if remaining <= 0:
                            break
                        if filename.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(source_folder, filename)
                            filename_without_ext = os.path.splitext(filename)[0]
                            print(f"Adding final augmentations to image {idx}/{actual_original_count}: {filename}")
                            
                            # Apply one more augmentation per image until target is met
                            augment_image_controlled(image_path, f"{filename_without_ext}_final_aug", destination_folder, 1)
                            remaining -= 1
                
                # Final verification
                total_images = len([f for f in os.listdir(destination_folder) if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))])
                if total_images < target_count:
                    print(f"Warning: Only {total_images} images were created, which is still less than the target of {target_count}.")
                elif total_images > target_count:
                    print(f"Note: {total_images} images were created, which exceeds the target of {target_count}.")
                    remove_excess_images(destination_folder, target_count)
                else:
                    print(f"Successfully reached the target of {target_count} images for '{dataset}' dataset in class '{cls}'.")

            elif total_images > target_count:
                print(f"Note: {total_images} images were created, which exceeds the target of {target_count}.")
                remove_excess_images(destination_folder, target_count)
            else:
                print(f"Successfully reached the target of {target_count} images for '{dataset}' dataset in class '{cls}'.")

if __name__ == "__main__":
    main()

print("Data augmentation completed.")

# Reorganizing Folders for Compatibility with CNN
current_base_dir = r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\dataset_original"
target_base_dir = r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\original_dataset"

# Define mapping for reorganization
reorganization_map = {
    "fine_original": {
        "test": "test/fine_test",
        "test_augmented": "test/fine_test_augmented",
        "train": "train/fine_train",
        "train_augmented": "train/fine_train_augmented"
    },
    "mild_original": {
        "test": "test/mild_test",
        "test_augmented": "test/mild_test_augmented",
        "train": "train/mild_train",
        "train_augmented": "train/mild_train_augmented"
    },
    "severe_original": {
        "test": "test/severe_test",
        "test_augmented": "test/severe_test_augmented",
        "train": "train/severe_train",
        "train_augmented": "train/severe_train_augmented"
    }
}

# Reorganize files
for category, subdirs in reorganization_map.items():
    for current_subdir, target_subdir in subdirs.items():
        current_path = os.path.join(current_base_dir, category, current_subdir)
        target_path = os.path.join(target_base_dir, target_subdir)
        
        if os.path.exists(current_path):
            # Create target directory if it doesn't exist
            os.makedirs(target_path, exist_ok=True)
            
            # Move files from current_path to target_path
            for file in os.listdir(current_path):
                shutil.move(os.path.join(current_path, file), os.path.join(target_path, file))
            print(f"Moved files from {current_path} to {target_path}")

# Optional: Remove empty directories
for category in reorganization_map.keys():
    category_path = os.path.join(current_base_dir, category)
    if os.path.exists(category_path) and not os.listdir(category_path):
        shutil.rmtree(category_path)
        print(f"Removed empty directory: {category_path}")

print("Reorganization complete.")

# Reorganizing files in test and train folders

# Paths to the directories
test_directory = r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\original_dataset\test"
train_directory = r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\original_dataset\train"

# Folders to delete for test and train directories
test_folders_to_delete = ["fine_test", "mild_test", "severe_test"]
train_folders_to_delete = ["fine_train", "mild_train", "severe_train"]

# Function to delete specified folders
def delete_specified_folders(directory, folders):
    try:
        for folder_name in folders:
            folder_path = os.path.join(directory, folder_name)
            if os.path.isdir(folder_path):
                print(f"Deleting folder: {folder_path}")
                shutil.rmtree(folder_path)
            else:
                print(f"Folder not found: {folder_path}")
        print("Deletion process completed.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    print("Processing test directory...")
    delete_specified_folders(test_directory, test_folders_to_delete)

    print("Processing train directory...")
    delete_specified_folders(train_directory, train_folders_to_delete)
