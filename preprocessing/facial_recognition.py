from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import os
import face_recognition

image_directory = 'raw_data/train'
if not os.path.exists(image_directory):
    print(f"Directory not found: {image_directory}")
else:
    print(f"Directory exists: {image_directory}")

images = os.listdir(image_directory)
print(f"Files in directory: {images}")


def draw_boxes_on_faces(image_directory, save_directory=None, display=False):
    """
    Draws bounding boxes around faces for all images in the specified directory.

    Parameters:
        image_directory (str): The directory where the images are stored.
        save_directory (str, optional): The directory where images with boxes will be saved. If None, images are not saved.
        display (bool, optional): Whether to display the images with boxes using Matplotlib. Default is False.
    """
    # List all files in the directory
    images = os.listdir(image_directory)

    # Process each image in the directory
    for i, image_name in enumerate(images):
        # Create the full path to the image
        image_path = os.path.join(image_directory, image_name)

        # Load the image using face_recognition
        image = face_recognition.load_image_file(image_path)

        # Find all face locations in the image
        face_locations = face_recognition.face_locations(image)

        # Convert the image to a PIL Image object for drawing
        pil_image = Image.fromarray(image)

        # Create a drawing object
        draw = ImageDraw.Draw(pil_image)

        # Draw a box around each face
        for (top, right, bottom, left) in face_locations:
            draw.rectangle(((left, top), (right, bottom)), outline="red", width=5)

        # Optionally display the image with the drawn box using Matplotlib
        if display:
            plt.imshow(np.asarray(pil_image))
            plt.axis('off')  # Turn off axis labels
            plt.show()

        # Optionally, save the image with the bounding box
        if save_directory:
            # Ensure the save directory exists
            os.makedirs(save_directory, exist_ok=True)

            # Save the image with bounding boxes
            save_path = os.path.join(save_directory, f"boxed_{image_name}")
            pil_image.save(save_path)

# Example usage:
draw_boxes_on_faces('raw_data/train', save_directory='output_images', display=True)
