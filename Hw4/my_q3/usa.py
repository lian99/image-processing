import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

def clean_USAflag(im):
    result = np.zeros_like(im)
    # Extract and process regions
    right_part = im[:, 140:]
    bottom_left = im[90:, :141]

    # Simple smoothing with median blur
    processed_right = cv2.medianBlur(right_part, 3)
    processed_right = uniform_filter1d(processed_right, size=15, axis=1)
    processed_bottom = cv2.medianBlur(bottom_left, 3)
    processed_bottom = uniform_filter1d(processed_bottom, size=15, axis=1)

    # Reconstruct image
    result[0:90, :141] = im[0:90, :141]  # Keep top-left unchanged
    result[:, 140:] = processed_right    # Add processed right part
    result[90:, :141] = processed_bottom  # Add processed bottom-left

    return result  # Ensure this is the last line in the function

# ------------------- Moved this OUTSIDE the function -------------------

print("-----------------------image 5----------------------\n")
im5 = cv2.imread(r'Images\USAflag.tif')
im5 = cv2.cvtColor(im5, cv2.COLOR_BGR2GRAY)
im5_clean = clean_USAflag(im5)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(im5, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Cleaned Image")
plt.imshow(im5_clean, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.tight_layout()
plt.show()
