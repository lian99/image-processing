import cv2
import numpy as np
from matplotlib import pyplot as plt


def extract_image(image, src_points):

   # Apply perspective transform to extract a sub-image.

    dst_points = np.float32([[0, 0], [256, 0], [256, 256], [0, 256]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    extracted = cv2.warpPerspective(image, matrix, (256, 256), flags=cv2.INTER_CUBIC)
    return extracted


def denoise_image(image):

    #Apply median blur to denoise the image.
    return cv2.medianBlur(image, 3)


def clean_baby_image(image):
    # Apply median blur on the entire image
    denoised_image = denoise_image(image)

    # Define source points (using your provided points)
    src_pts1 = np.float32([[5, 20], [111, 20], [111, 130], [5, 130]])
    src_pts2 = np.float32([[181, 5], [249, 70], [177, 121], [121, 51]])
    src_pts3 = np.float32([[78, 162], [146, 116], [246, 159], [132, 245]])

    # Extract and clean each baby face
    face1 = denoise_image(extract_image(denoised_image, src_pts1))
    face2 = denoise_image(extract_image(denoised_image, src_pts2))
    face3 = denoise_image(extract_image(denoised_image, src_pts3))

    # Merge the three cleaned images using pixel-wise **median** to remove noise
    merged_image = np.mean(np.array([face1, face2, face3]), axis=0)

    return merged_image.astype(np.uint8)


# Load the noisy baby image
baby_image = cv2.imread('Images/baby.tif', cv2.IMREAD_GRAYSCALE)

# Clean the image
cleaned_baby = clean_baby_image(baby_image)

# Display the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Noisy Image')
plt.imshow(baby_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Restored Baby Image')
plt.imshow(cleaned_baby, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
