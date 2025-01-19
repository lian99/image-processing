import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the blurry watermelon image
image_path = 'Images\watermelon.tif'  # Update with the correct image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the sharpening kernel
sharpening_kernel = np.array([[0, -6, 0],
                              [-6, 24, -6],
                              [0, -6, 0]], dtype=np.float32)

# Apply the kernel to the image using filter2D
sharpened_image = cv2.filter2D(src=image, ddepth=-1, kernel=sharpening_kernel)
final_image = cv2.add(image, sharpened_image)

# Display the original and sharpened images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Blurry Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Sharpened Image')
plt.imshow(final_image, cmap='gray')


plt.show()

# Save the sharpened image
cv2.imwrite('sharpened_watermelon.png', sharpened_image)
print("Sharpened image saved as 'sharpened_watermelon.png'")
