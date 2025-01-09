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


# ---------------- Apply Cleaning for Each Image ---------------- #

# Clean the first image
image_path_1 = "taj.jpg"
image_1 = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
filtered_image_1 = clean_Gaussian_noise_bilateral(image_1, radius=5, stdSpatial=50, stdIntensity=30)
cv2.imwrite("cleaned_taj.jpg", filtered_image_1)

# Clean the second image
image_path_2 = "NoisyGrayImage.png"
image_2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)
filtered_image_2 = clean_Gaussian_noise_bilateral(image_2, radius=7, stdSpatial=7, stdIntensity=100)
cv2.imwrite("cleaned_NoisyGrayImage.jpg", filtered_image_2)

# Clean the third image
image_path_3 = "balls.jpg"
image_3 = cv2.imread(image_path_3, cv2.IMREAD_GRAYSCALE)
filtered_image_3 = clean_Gaussian_noise_bilateral(image_3, radius=9, stdSpatial=10, stdIntensity=10)
cv2.imwrite("cleaned_balls.jpg", filtered_image_3)

# ---------------- Plot Results ---------------- #

# Display the results for each image
fig, axes = plt.subplots(3, 2, figsize=(10, 15))

# Original and cleaned image 1
axes[0, 0].imshow(image_1, cmap='gray')
axes[0, 0].set_title("Original Taj")
axes[0, 1].imshow(filtered_image_1, cmap='gray')
axes[0, 1].set_title("Cleaned Taj")

# Original and cleaned image 2
axes[1, 0].imshow(image_2, cmap='gray')
axes[1, 0].set_title("Original NoisyGrayImage")
axes[1, 1].imshow(filtered_image_2, cmap='gray')
axes[1, 1].set_title("Cleaned NoisyGrayImage")

# Original and cleaned image 3
axes[2, 0].imshow(image_3, cmap='gray')
axes[2, 0].set_title("Original Balls")
axes[2, 1].imshow(filtered_image_3, cmap='gray')
axes[2, 1].set_title("Cleaned Balls")

plt.tight_layout()
plt.show()
