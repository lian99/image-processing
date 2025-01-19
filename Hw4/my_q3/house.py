import cv2
import numpy as np
from matplotlib import pyplot as plt


def clean_house(image):

    #Create and normalize the horizontal motion blur kernel
    motion_blur_kernel = np.zeros(image.shape, dtype=complex)
    motion_blur_kernel[0, :10] = 0.1  # Simulate horizontal motion blur
    motion_blur_frequency = np.fft.fft2(motion_blur_kernel)
    motion_blur_frequency[np.abs(motion_blur_frequency) < 0.01] = 1  # Prevent division by small values
    #Apply Fourier Transform to the input image
    image_frequency = np.fft.fft2(image)

    #Apply inverse filtering in the frequency domain
    restored_frequency = image_frequency / motion_blur_frequency

    #Perform inverse Fourier Transform to get the cleaned image
    restored_image = np.abs(np.fft.ifft2(restored_frequency))

    return restored_image
print("-----------------------image 6----------------------\n")
im6 = cv2.imread(r'Images\house.tif')
im6 = cv2.cvtColor(im6, cv2.COLOR_BGR2GRAY)
im6_clean = clean_house(im6)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(im6, cmap='gray', vmin=0, vmax=255)
plt.subplot(1, 2, 2)
plt.imshow(im6_clean, cmap='gray', vmin=0, vmax=255)

plt.show()