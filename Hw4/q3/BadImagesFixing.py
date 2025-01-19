# Lian Natour, 207300443
# Mohammad Mhneha, 315649814

# Please replace the above comments with your names and ID numbers in the same format.

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

def extract_image(image, src_points):

   # Apply perspective transform to extract a sub-image.
    dst_points = np.float32([[0, 0], [256, 0], [256, 256], [0, 256]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    extracted = cv2.warpPerspective(image, matrix, (256, 256), flags=cv2.INTER_CUBIC)
    return extracted

def denoise_image(image):
    #Apply median blur to denoise the image.
    return cv2.medianBlur(image, 3)

def clean_baby(im):
    # Apply median blur on the entire image
    denoised_image = denoise_image(im)
    # Define source points (using your provided points)
    src_pts1 = np.float32([[5, 20], [111, 20], [111, 130], [5, 130]])
    src_pts2 = np.float32([[181, 5], [249, 70], [177, 121], [121, 51]])
    src_pts3 = np.float32([[78, 162], [146, 116], [246, 159], [132, 245]])

    # Extract and clean each baby face
    face1 = denoise_image(extract_image(denoised_image, src_pts1))
    face2 = denoise_image(extract_image(denoised_image, src_pts2))
    face3 = denoise_image(extract_image(denoised_image, src_pts3))

    # Merge the three cleaned images using mean to remove noise
    merged_image = np.mean(np.array([face1, face2, face3]), axis=0)

    return merged_image.astype(np.uint8)

def clean_windmill(im):
    #Apply Fourier transform
    f = fft2(im)  # Compute the 2-dimensional discrete Fourier Transform
    fshift = fftshift(f)  # Shift the zero-frequency component to the center of the spectrum

    #Set the peaks to zero
    fshift[124, 100] = 0
    fshift[132, 156] = 0

    #Perform the inverse Fourier transform to reconstruct the image
    f_ishift = ifftshift(fshift)  # Shift back the zero frequency component
    img_reconstructed = np.abs(ifft2(f_ishift))  # Apply the inverse FFT

    return img_reconstructed

def clean_watermelon(im):
    # Define the sharpening kernel
    sharpening_kernel = np.array([[0, -6, 0],
                                  [-6, 24, -6],
                                  [0, -6, 0]], dtype=np.float32)

    # Apply the kernel to the image using filter2D
    sharpened_image = cv2.filter2D(src=im, ddepth=-1, kernel=sharpening_kernel)
    final_image = cv2.add(im, sharpened_image)
    return final_image

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
    result[:, 140:] = processed_right  # Add processed right part
    result[90:, :141] = processed_bottom  # Add processed bottom-left

    return result  # Ensure this is the last line in the function

def clean_house(im):
    # Create and normalize the horizontal motion blur kernel
    motion_blur_kernel = np.zeros(im.shape, dtype=complex)
    motion_blur_kernel[0, :10] = 0.1  # Simulate horizontal motion blur
    motion_blur_frequency = np.fft.fft2(motion_blur_kernel)
    motion_blur_frequency[np.abs(motion_blur_frequency) < 0.01] = 1  # Prevent division by small values
    # Apply Fourier Transform to the input image
    image_frequency = np.fft.fft2(im)

    # Apply inverse filtering in the frequency domain
    restored_frequency = image_frequency / motion_blur_frequency

    # Perform inverse Fourier Transform to get the cleaned image
    restored_image = np.abs(np.fft.ifft2(restored_frequency))

    return restored_image

def clean_bears(im):
    equalized_image = cv2.equalizeHist(im)
    gamma = 0.9
    equalized_image = equalized_image ** gamma
    return equalized_image


