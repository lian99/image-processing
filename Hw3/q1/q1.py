# Lian Natour, 207300443
# Mohammad Mhneha, 315649814
import cv2
import numpy as np
import os

def calc_mse(image1, image2):
    """Calculate the Mean Squared Error (MSE) between two images."""
    mse = np.mean((image1 - image2) ** 2)
    print(f"MSE: {mse}")
    return mse

if __name__ == "__main__":

    # Image_1 Solution
    # Load the original image in grayscale
    input_image_filename = "1.jpg"
    comparison_image_filename = "image_1.jpg"
    output_image_filename = "filtered_image_1.jpg"
    # Read the input image
    input_image = cv2.imread(input_image_filename, cv2.IMREAD_GRAYSCALE)
    # Create a custom kernel with the middle row containing 1's
    kernel_height, kernel_width = input_image.shape
    kernel = np.zeros((kernel_height, kernel_width), dtype=np.float32)
    middle_row = kernel_height // 2
    kernel[middle_row, :] = 1.0
    # Normalize the kernel so that the sum of all its values equals 1
    kernel /= np.sum(kernel)
    # Apply the filter using OpenCV's filter2D function
    filtered_image = cv2.filter2D(input_image, -1, kernel, borderType=cv2.BORDER_WRAP)
    # Save the filtered image
    cv2.imwrite(output_image_filename, filtered_image)
    print(f"Filtered image saved as: {output_image_filename}")
    # Read the comparison image
    comparison_image = cv2.imread(comparison_image_filename, cv2.IMREAD_GRAYSCALE)
    # Calculate the Mean Squared Error between the filtered image and the comparison image
    calc_mse(comparison_image, filtered_image)

    # ---------------- Image_2 Solution ----------------
    # Load the original image in grayscale
    comparison_image_filename = "image_2.jpg"
    output_image_filename = "filtered_image_2.jpg"
    # Apply Gaussian Blur
    filtered_image = cv2.GaussianBlur(input_image, (11, 11), 15, borderType=cv2.BORDER_WRAP)
    # Save the filtered image
    cv2.imwrite(output_image_filename, filtered_image)
    print(f"Filtered image saved as: {output_image_filename}")
    # Read the comparison image
    comparison_image = cv2.imread(comparison_image_filename, cv2.IMREAD_GRAYSCALE)
    # Calculate the Mean Squared Error between the filtered image and the comparison image
    calc_mse(comparison_image, filtered_image)

    """ # Image_2 Solution
    # Load the original image in grayscale
    input_image_filename = "1.jpg"
    comparison_image_filename = "image_2.jpg"
    output_image_filename = "optimized_edited_image_2.jpg"

    input_image = cv2.imread(input_image_filename, cv2.IMREAD_GRAYSCALE)
    comparison_image = cv2.imread(comparison_image_filename, cv2.IMREAD_GRAYSCALE)

    # Initialize variables to track the minimum MSE and the corresponding parameters
    min_mse = float('inf')
    best_kernel = (0, 0)
    best_sigma = 0

    # Test different kernel sizes and sigma values
    for kernel_size in range(3, 16, 2):  # Kernel sizes: 3x3, 5x5, ..., 15x15
        for sigma in np.arange(0.5,11, 0.1):  # Sigma values from 0.5 to 3.0
            # Apply Gaussian Blur
            blurred_image = cv2.GaussianBlur(input_image, (kernel_size, kernel_size), sigma)
            # Calculate MSE
            mse = calc_mse(comparison_image, blurred_image)
            print(f"Kernel: ({kernel_size}, {kernel_size}), Sigma: {sigma:.1f}, MSE: {mse:.4f}")

            # Update the best parameters if a lower MSE is found
            if mse < min_mse:
                min_mse = mse
                best_kernel = (kernel_size, kernel_size)
                best_sigma = sigma

    # Apply the Gaussian Blur with the best parameters
    optimized_blurred_image = cv2.GaussianBlur(input_image, best_kernel, best_sigma)

    # Save the optimized blurred image
    cv2.imwrite(output_image_filename, optimized_blurred_image)
    print(f"Optimized filtered image saved as: {output_image_filename}")
    print(f"Best Kernel: {best_kernel}, Best Sigma: {best_sigma}, Minimum MSE: {min_mse:.4f}") """

# ---------------- Image_3 Solution ----------------
# Load the original image in grayscale
comparison_image_filename = "image_3.jpg"
output_image_filename = "filtered_image_3.jpg"
# Apply Median Filter
# Kernel size defines the size of the neighborhood for calculating the median
filtered_image = cv2.medianBlur(input_image, 11)  # Kernel size: 11
# Save the filtered image
cv2.imwrite(output_image_filename, filtered_image)
print(f"Filtered image saved as: {output_image_filename}")
# Read the comparison image
comparison_image = cv2.imread(comparison_image_filename, cv2.IMREAD_GRAYSCALE)
# Calculate the Mean Squared Error between the filtered image and the comparison image
calc_mse(comparison_image, filtered_image)

# ---------------- Image_4 Solution ----------------
# Load the original image in grayscale
comparison_image_filename = "image_4.jpg"
output_image_filename = "filtered_image_4.jpg"
# Define a vertical kernel for averaging along the y-axis
kernel_height = 15
kernel = np.ones((kernel_height, 1), dtype=np.float32) / kernel_height
# Handle border by padding the image
border_size = kernel_height // 2
padded_im = cv2.copyMakeBorder(input_image, border_size, border_size, 0, 0, cv2.BORDER_WRAP)
# Apply the vertical averaging filter
filtered_image = cv2.filter2D(padded_im


                              , -1, kernel)
# Crop the padded edges to match the original size
filtered_image = filtered_image[border_size:-border_size, :]
# Save the filtered image
cv2.imwrite(output_image_filename, filtered_image)
print(f"Filtered image saved as: {output_image_filename}")
# Read the comparison image
comparison_image = cv2.imread(comparison_image_filename, cv2.IMREAD_GRAYSCALE)
# Calculate the MSE
mse = calc_mse(comparison_image, filtered_image)

# ---------------- Image_5 Solution ----------------
comparison_image_filename = "image_5.jpg"
output_image_filename = "filtered_image_5.jpg"
# Load images
input_image = cv2.imread(input_image_filename, cv2.IMREAD_GRAYSCALE)
comparison_image = cv2.imread(comparison_image_filename, cv2.IMREAD_GRAYSCALE)
# Best parameters
kernel_size = (11, 11)
sigma = 5.9
add_constant = 128
# Apply Gaussian Blur and adjust with the constant
blurred_image = cv2.GaussianBlur(input_image, kernel_size, sigma)
processed_image = cv2.addWeighted(input_image.astype(np.float32), 1.0,
                                  blurred_image.astype(np.float32), -1.0, add_constant)
processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
# Save the processed image
cv2.imwrite(output_image_filename, processed_image)
print(f"Filtered image saved as: {output_image_filename}")
# Calculate and print MSE
mse = calc_mse(comparison_image, processed_image)
"""input_image_filename = "1.jpg"  # Original image
comparison_image_filename = "image_6.jpg"  # Target image
output_image_filename = "edited_image_6.jpg"

# Load images in grayscale
input_image = cv2.imread(input_image_filename, cv2.IMREAD_GRAYSCALE)
comparison_image = cv2.imread(comparison_image_filename, cv2.IMREAD_GRAYSCALE)

# Initialize variables to track the best MSE and kernel
min_mse = float('inf')
best_kernel = None
best_image = None
# Base kernel matrix (from your best results)
base_kernel = np.array([[-0.4, -0.4, -0.4], [0, 0, 0], [0.4, 0.4, 0.4]], dtype=np.float32)

# Iterate over different multipliers to scale the kernel
for multiplier in np.arange(0.8, 5, 0.05):  # Multipliers from 0.8 to 5
    # Scale the base kernel
    scaled_kernel = base_kernel * multiplier

    # Apply the kernel using filter2D
    filtered_image = cv2.filter2D(input_image, -1, scaled_kernel)
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    # Calculate the MSE between the filtered image and the target
    mse = calc_mse(comparison_image, filtered_image)

    # Update the best parameters if a lower MSE is found
    if mse < min_mse:
        min_mse = mse
        best_kernel = scaled_kernel
        best_image = filtered_image.copy()

# Save the best result
if best_image is not None:
    cv2.imwrite(output_image_filename, best_image)
    print(f"Filtered image saved as: {output_image_filename}")
    print(f"Best Kernel Matrix:\n{best_kernel}")
    print(f"Minimum MSE: {min_mse}")
else:
    print("No valid filtered image was generated.")"""

# ---------------- Image_6 Solution ----------------
comparison_image_filename = "image_6.jpg"
output_image_filename = "filtered_image_6.jpg"
# Load the input and comparison images in grayscale
input_image = cv2.imread(input_image_filename, cv2.IMREAD_GRAYSCALE)
comparison_image = cv2.imread(comparison_image_filename, cv2.IMREAD_GRAYSCALE)
 # Best kernel matrix obtained from the optimization
best_kernel = np.array([
        [-0.34, -0.34, -0.34],
        [ 0.0,   0.0,   0.0],
        [ 0.34,  0.34,  0.34]
    ], dtype=np.float32)
# Apply the custom kernel using filter2D
filtered_image = cv2.filter2D(input_image, -1, best_kernel)
filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
# Save the filtered image
cv2.imwrite(output_image_filename, filtered_image)
print(f"Filtered image saved as: {output_image_filename}")
# Calculate and print the MSE
mse = calc_mse(comparison_image, filtered_image)

# ---------------- Image_7 Solution ----------------
comparison_image_filename = "image_7.jpg"
output_image_filename = "filtered_image_7.jpg"
input_image = cv2.imread(input_image_filename, cv2.IMREAD_GRAYSCALE)
comparison_image = cv2.imread(comparison_image_filename, cv2.IMREAD_GRAYSCALE)
# Define a kernel for a vertical cyclic shift
kernel = np.zeros((input_image.shape[0], 1), dtype=np.float32)  # Column vector kernel
kernel[0, 0] = 1  # Set the first element to create the cyclic shift effect
# Apply the kernel using filter2D with BORDER_WRAP for cyclic behavior
filtered_image = cv2.filter2D(input_image, -1, kernel, borderType=cv2.BORDER_WRAP)
# Save the filtered image
cv2.imwrite(output_image_filename, filtered_image)
print(f"Filtered image saved as: {output_image_filename}")
# Calculate and print the MSE
calc_mse(comparison_image, filtered_image)

# ---------------- Image_8 Solution ----------------
comparison_image_filename = "image_8.jpg"  # Target image
output_image_filename = "filtered_image_8.jpg"  # Output image path
# Load the input image in grayscale
input_image = cv2.imread(input_image_filename, cv2.IMREAD_GRAYSCALE)
# Load the comparison image
comparison_image = cv2.imread(comparison_image_filename, cv2.IMREAD_GRAYSCALE)
# Define the identity kernel
kernel = np.zeros((3, 3), dtype=np.float32)
kernel[1, 1] = 1.0  # The identity kernel
# Apply the identity kernel using filter2D
filtered_image = cv2.filter2D(input_image, -1, kernel)
# Save the filtered image
cv2.imwrite(output_image_filename, filtered_image)
print(f"Filtered image saved as: {output_image_filename}")
# Calculate the Mean Squared Error (MSE) between the filtered image and the comparison image
mse = calc_mse(comparison_image, filtered_image)

# ---------------- Image_9 Solution ----------------
comparison_image_filename = "image_9.jpg"  # Target image
output_image_filename = "filtered_image_9.jpg"  # Output image path
# Load the input image in grayscale
input_image = cv2.imread(input_image_filename, cv2.IMREAD_GRAYSCALE)
# Load the comparison image
comparison_image = cv2.imread(comparison_image_filename, cv2.IMREAD_GRAYSCALE)
# sharpening kernel
sharpening_kernel = np.array([
    [-0.5, -0.5, -0.5],
    [-0.5, 6, -0.5],
    [-0.5, -0.5, -0.5]
], dtype=np.float32)/2
# Apply the sharpening kernel using filter2D
filtered_image = cv2.filter2D(input_image, -1, sharpening_kernel)
# Save the filtered image
cv2.imwrite(output_image_filename, filtered_image)
print(f"Filtered image saved as: {output_image_filename}")
# Calculate and print the MSE
mse = calc_mse(comparison_image, filtered_image)
