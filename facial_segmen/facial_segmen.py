import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Define the base directory path
base_dir = '/Users/amierzuhri/code/JpLepeckiBR/PCA/raw_data/facial_recognition_data/raw_data'

def display_images_from_directory(base_dir, num_images_to_display=5):
    """
    Display a specified number of images from the given directory.

    Parameters:
    - base_dir (str): The directory containing the image files.
    - num_images_to_display (int): The number of images to display.
    """
    # List all image files in the directory
    image_files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f)) and f.lower().endswith(('.jpg'))]

    # Determine how many images to display
    num_images_to_display = min(num_images_to_display, len(image_files))

    # Create a figure to display the images
    plt.figure(figsize=(15, 15))

    # Loop through and display the images
    for i in range(num_images_to_display):
        img_path = os.path.join(base_dir, image_files[i])
        img = Image.open(img_path)

        plt.subplot(1, num_images_to_display, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Image {i + 1}')

    plt.show()

def preprocess_images(base_dir, image_files, image_size=(256, 256)):
    """
    Preprocess images: resize and normalize.

    Parameters:
    - base_dir (str): The directory containing the image files.
    - image_files (list): List of image file names to process.
    - image_size (tuple): The target size to resize images.

    Returns:
    - np.ndarray: Array of processed images.
    """
    images = []
    for file_name in image_files:
        img_path = os.path.join(base_dir, file_name)
        img = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB format
        img = img.resize(image_size)               # Resize to the target size
        img = np.array(img) / 255.0                # Normalize the image to [0, 1]
        images.append(img)
    return np.array(images)

# Display the images from the directory
display_images_from_directory(base_dir)

# List image files again for preprocessing
image_files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f)) and f.lower().endswith(('.jpg'))]

# Preprocess the images
processed_images = preprocess_images(base_dir, image_files)
print("Processed images shape:", processed_images.shape)
