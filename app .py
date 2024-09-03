#######---------------------------------------------------#######
#######----------------FACE SEGMENTATION------------------#######
#######---------------------------------------------------#######

import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# convenience expression for automatically determining device
device = (
    "cuda"
    # Device for NVIDIA or AMD GPUs
    if torch.cuda.is_available()
    else "mps"
    # Device for Apple Silicon (Metal Performance Shaders)
    if torch.backends.mps.is_available()
    else "cpu"
)

# load models
image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
model.to(device)


image_dir = "raw_data/cool_selfie.jpg/"
image = Image.open(image_dir)


# run inference on image
inputs = image_processor(images=image, return_tensors="pt").to(device)
outputs = model(**inputs)
logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

# resize output to match input image dimensions
upsampled_logits = nn.functional.interpolate(logits,
                size=image.size[::-1], # H x W
                mode='bilinear',
                align_corners=False)

# get label masks
labels = upsampled_logits.argmax(dim=1)[0]



# move to CPU to visualize in matplotlib
labels_viz = labels.cpu().numpy()
plt.imshow(labels_viz)
plt.show()

plt.imshow(image)

foreground = labels_viz
background = Image.open(image_dir)
R = np.array(background)[:,:,0]
G = np.array(background)[:,:,1]
B = np.array(background)[:,:,2]

skin = np.dstack((R*(np.array(foreground)==1),G*(np.array(foreground)==1) ,B*(np.array(foreground)==1)))
plt.imshow(skin)
plt.show()
hair = np.dstack((R*(np.array(foreground)==13),G*(np.array(foreground)==13) ,B*(np.array(foreground)==13)))
plt.imshow(hair)
plt.show()

#######---------------------------------------------------#######
#######--------------------HEX VALUES---------------------#######
#######---------------------------------------------------#######
