import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift



def clean_umbrella(im):
    transform = fft2(im)

    # Step 2: Create H with frequency components
    H = np.zeros(im.shape, dtype=complex)
    H[0, 0] = 0.5
    H[4, 79] = 0.5  # Flip correction frequency

    # Convert H to frequency domain
    H = fft2(H)

    # Step 3: Use H_conj in the correction formula
    H_conj = np.conj(H)

    # Following the formula: F = (H* * G) / (H*H + λ)
    numerator = H_conj * transform
    denominator = H_conj * H + 0.001  # lambda_param = 0.001

    # Apply the correction
    F = numerator / denominator

    # Step 4: Convert back to spatial domain
    corrected_image = np.abs(ifft2(F))

    return corrected_image
print("-----------------------image 4----------------------\n")
im4 = cv2.imread(r'Images\umbrella.tif')
im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)
im4_clean = clean_umbrella(im4)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(im4, cmap='gray', vmin=0, vmax=255)
plt.subplot(1, 2, 2)
plt.imshow(im4_clean, cmap='gray', vmin=0, vmax=255)
plt.show()