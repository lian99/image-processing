import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy.fft import fft2, fftshift, ifftshift, ifft2

# Load the image
image_path = 'Images\windmill.tif'  # Update the path accordingly
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
output_image_filename='wew.png'
# Step 1: Apply Fourier transform
f = fft2(image)                # Compute the 2-dimensional discrete Fourier Transform
fshift = fftshift(f)           # Shift the zero-frequency component to the center of the spectrum

fshift[124, 100] = 0  # Set the DC component to zero
fshift[132, 156] = 0  # Set the DC component to zero

# Step 3: Perform the inverse Fourier transform to reconstruct the image
f_ishift = ifftshift(fshift)        # Shift back the zero frequency component
img_reconstructed = np.abs(ifft2(f_ishift))  # Apply the inverse FFT