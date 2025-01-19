# Lian Natour, 207300443
# Mohammad Mhneha, 315649814
import numpy as np
import cv2
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import matplotlib.pyplot as plt

def pad_fourier_transform(transform, new_height, new_width):
    height, width = transform.shape
    pad_transform = np.zeros((new_height, new_width), dtype=np.complex128)
    start_row = (new_height - height) // 2
    start_col = (new_width - width) // 2
    pad_transform[start_row:start_row + height, start_col:start_col + width] = transform
    return pad_transform

def create_four_copies(transform, new_height, new_width):
    padded_transform = np.zeros((new_height, new_width), dtype=np.complex128)
    for i in range(transform.shape[0]):
        for j in range(transform.shape[1]):
            padded_transform[2 * i, 2 * j] = transform[i, j]
    return padded_transform

# Load image and compute its Fourier transform
image_path = 'zebra.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
fourier_transform = fftshift(fft2(image))

# Apply padding to double the image size
doubled_transform = pad_fourier_transform(fourier_transform, image.shape[0] * 2, image.shape[1] * 2)
doubled_image = np.abs(ifft2(ifftshift(doubled_transform)))

# Create four copies
four_copies_transform = create_four_copies(fourier_transform, image.shape[0] * 2 - 1, image.shape[1] * 2 - 1)
four_copies_image = np.abs(ifft2(ifftshift(four_copies_transform)))

# Visualize results
plt.figure(figsize=(10, 10))
plt.subplot(321)
plt.title('Original Grayscale Image')
plt.imshow(image, cmap='gray')

plt.subplot(322)
plt.title('Fourier Spectrum')
plt.imshow(np.log(1 + np.abs(fourier_transform)), cmap='gray')

plt.subplot(323)
plt.title('Fourier Spectrum Zero Padding')
plt.imshow(np.log(1 + np.abs(doubled_transform)), cmap='gray')

plt.subplot(324)
plt.title('Two Times Larger Grayscale Image')
plt.imshow(doubled_image, cmap='gray')

plt.subplot(325)
plt.title('Fourier Spectrum Four Copies')
plt.imshow(np.log(1 + np.abs(four_copies_transform)), cmap='gray')

plt.subplot(326)
plt.title('Four Copies Grayscale Image')
plt.imshow(four_copies_image, cmap='gray')

plt.savefig('zebra_scaled.png')
plt.show()
