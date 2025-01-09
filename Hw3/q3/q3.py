# Lian Natour, 207300443
# Mohammad Mhneha, 315649814

import numpy as np
import cv2
import matplotlib.pyplot as plt


def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):

    # Ensure the image is in float64 format
    im = im.astype(np.float64)

    # Apply padding to the image
    padded_image = np.pad(im, radius, mode='reflect')

    # Get the dimensions of the padded image
    padded_height, padded_width = padded_image.shape

    # Initialize the output image
    cleanIm = np.zeros_like(im)

    # Create spatial Gaussian kernel
    x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    gs = np.exp(-(x ** 2 + y ** 2) / (2 * stdSpatial ** 2))

    # Iterate over each pixel in the original image
    for i in range(radius, padded_height - radius):
        for j in range(radius, padded_width - radius):
            # Extract the window centered around the pixel
            window = padded_image[i - radius:i + radius + 1, j - radius:j + radius + 1]

            # Compute intensity Gaussian
            gi = np.exp(-((window - padded_image[i, j]) ** 2) / (2 * stdIntensity ** 2))

            # Compute the bilateral filter response
            weights = gs * gi
            weights /= weights.sum()
            cleanIm[i - radius, j - radius] = np.sum(weights * window)

    # Convert the result back to uint8
    return cleanIm.astype(np.uint8)
from matplotlib import pyplot as plt

#Question 3 section a
# Define the paths
input_image_filename = "broken.jpg"
output_image_filename = "fixed_image_median_bilateral.jpg"

# Load the input image in grayscale
input_image = cv2.imread(input_image_filename, cv2.IMREAD_GRAYSCALE)
if input_image is None:
    raise FileNotFoundError("Input image not found.")

#Apply median blur to reduce noise
median_blurred = cv2.medianBlur(input_image, 5)  # Kernel size is 5

#Apply a bilateral filter
bilateral_filtered = clean_Gaussian_noise_bilateral(median_blurred, 5, 75, 75)

# Save the resulting image
cv2.imwrite(output_image_filename, bilateral_filtered)
print(f"Filtered image saved as: {output_image_filename}")

#Question 3 section b
# Define the paths
noised_images_file = "noised_images.npy"
output_image_filename = "cleaned_image.jpg"

# Load the noised images
try:
    noised_images = np.load(noised_images_file)
    print(f"Loaded {noised_images.shape[0]} noised images of size {noised_images.shape[1:]} each.")
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find the file {noised_images_file}.")
#  Initialize an array to store the processed images
processed_images = np.zeros_like(noised_images, dtype=np.float32)
# Step 3: Process each image
for i, image in enumerate(noised_images):
    # Apply median blur
    median_blurred = cv2.medianBlur(image, 5)  # Kernel size is 5
    # Apply bilateral filter
    bilateral_filtered = cv2.bilateralFilter(median_blurred, d=7, sigmaColor=25, sigmaSpace=25)
    # Accumulate the processed image
    processed_images[i] = bilateral_filtered
# Step 4: Compute the average of all processed images
final_cleaned_image = np.mean(processed_images, axis=0).astype(np.uint8)
# Step 5: Save the resulting image
cv2.imwrite(output_image_filename, final_cleaned_image)
print(f"Cleaned image saved as: {output_image_filename}")

