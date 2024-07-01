"""
==================================
Video Color Quantization using K-Means
==================================

Performs a pixel-wise Vector Quantization (VQ) of the entire video file, reducing the number of colors required to show the video,
then saves the file.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
from PIL import Image
import skimage
from pathlib import Path
import shutil

def quantize_image(image, num_colors):
    pic = Image.open(image)
    pic = np.array(pic, dtype=np.float64) / 255
    w, h, d = original_shape = tuple(pic.shape)
    assert d == 3
    image_array = np.reshape(pic, (w * h, d))

    image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    kmeans = KMeans(n_clusters=num_colors, n_init="auto", random_state=0).fit(
    image_array_sample
    )

    labels = kmeans.predict(image_array)

    quantized_image = kmeans.cluster_centers_[labels].reshape(w, h, -1)
    quantized_image = skimage.img_as_ubyte(quantized_image)

    return quantized_image


# Settings
num_colors = 30 # Number of colors for quantization
video_path = 'vid1.mp4'
output_path = 'vid1_quantized_.mp4'

# Open the video file and get properties
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(5)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

counter=0
counter2=0

temp_dir = 'frames_temp'
Path(temp_dir).mkdir(exist_ok=True)

# Loop through each frame, apply quantization, and write to the output video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    counter=counter+1
    counter2=counter2+1
    if counter == 20:
        print("Processing frame num: ", counter2)
        counter=0

    temp_image_path = f'{temp_dir}/frame_{counter2:04d}.png'
    cv2.imwrite(temp_image_path, frame)

    # Apply quantization to the frame
    quantized_frame = quantize_image(temp_image_path, num_colors)

    # Write the quantized frame to the output video
    out.write(quantized_frame)

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

shutil.rmtree(temp_dir)

print("Finished")

