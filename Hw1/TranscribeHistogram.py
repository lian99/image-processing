# Lian Natour, 207300443
# Mohammad Mhneha, 315649814

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

# Input: numpy array of images and number of gray levels to quantize the images down to
# Output: numpy array of images, each with only n_colors gray levels
def quantization(imgs_arr, n_colors=4):
	img_size = imgs_arr[0].shape
	res = []

	for img in imgs_arr:
		X = img.reshape(img_size[0] * img_size[1], 1)
		km = KMeans(n_clusters=n_colors)
		km.fit(X)
		
		img_compressed = km.cluster_centers_[km.labels_]
		img_compressed = np.clip(img_compressed.astype('uint8'), 0, 255)

		res.append(img_compressed.reshape(img_size[0], img_size[1]))
	
	return np.array(res)

# Input: A path to a folder and formats of images to read
# Output: numpy array of grayscale versions of images read from input folder, and also a list of their names
def read_dir(folder, formats=(".jpg", ".png")):
	image_arrays = []
	lst = [file for file in os.listdir(folder) if file.endswith(formats)]
	for filename in lst:
		file_path = os.path.join(folder, filename)
		image = cv2.imread(file_path)
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image_arrays.append(gray_image)
	return np.array(image_arrays), lst

# Input: image object (as numpy array) and the index of the wanted bin (between 0 to 9)
# Output: the height of the idx bin in pixels
def get_bar_height(image, idx):
	# Assuming the image is of the same pixel proportions as images supplied in this exercise, the following values will work
	x_pos = 70 + 40 * idx
	y_pos = 274
	while image[y_pos, x_pos] == 0:
		y_pos-=1
	return 274 - y_pos

# Sections c, d
# Remember to uncomment compare_hist before using it!

def compare_hist(src_image, target):
	# Your code goes here
	emd_threshold = 260  # Threshold for EMD matching
	# Calculate the histogram of the target image
	target_hist = cv2.calcHist([target], [0], None, [256], [0, 256]).flatten()
	# Define the sliding window size (same as the size of the target image)
	window_size = target.shape
	# Restrict the region of the source image to speed up the search
	search_region = src_image[:130, 15:50]  # Adjust these numbers for your specific image
	# Create sliding windows using numpy's stride tricks
	windows = np.lib.stride_tricks.sliding_window_view(search_region, window_size)
	# Iterate over each window and compare its histogram to the target histogram
	for y in range(windows.shape[0]):  # Loop through rows of windows
		for x in range(windows.shape[1]):  # Loop through columns of windows
			# Calculate the histogram of the current window
			window_hist = cv2.calcHist([windows[y, x]], [0], None, [256], [0, 256]).flatten()
			# Calculate Earth Mover's Distance (EMD)
			emd = np.sum(np.abs(np.cumsum(window_hist) - np.cumsum(target_hist)))
			# If the EMD is below the threshold, a match is found
			if emd < emd_threshold:
				return True
	# If no match is found, return False
	return False

# Sections a, b

images, names = read_dir('data')
numbers, _ = read_dir('numbers')
#test problem 3 a and b
# #=====================================
#cv2.imshow(names[0], images[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#section b
# Load and display digit images

#digit_images, digit_names = read_dir('numbers')
#for i, digit_img in enumerate(digit_images):
    #cv2.imshow(f"Digit {digit_names[i]}", digit_img)
    #cv2.waitKey(0)  # Wait for a key press
#cv2.destroyAllWindows()  # Close all windows
#=====================================

#=====================================
#test for section c

# Test for the first histogram image and digit templates
src_image = images[0]  # First histogram image

# Loop through digit images
for digit_idx, target_image in enumerate(numbers):
    # Call compare_hist to check if the digit is present
    is_present = compare_hist(src_image, target_image)

    # Print the result
   # print(f"Is digit '{digit_idx}' present in the histogram? {is_present}")

#search_region = src_image[:130, 15:50]
#cv2.imshow("Search Region", search_region)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#=====================================

#=====================================
#section d
# Function to detect the topmost number for each histogram
def detect_topmost_numbers(images, numbers):
    heights_num_list = []
    for src_image in images:
        detected_number = None
        for digit_id in range(9, -1, -1):
            target_image = numbers[digit_id]
            if compare_hist(src_image, target_image):
                detected_number = digit_id
                break
        heights_num_list.append(detected_number)
    return heights_num_list

# Detect topmost numbers
heights_num_list = detect_topmost_numbers(images, numbers)
#for img_id, detected_number in enumerate(heights_num_list):
   # print(f"Histogram {names[img_id]} detected the number: {detected_number}")
#=====================================


#=====================================
#section e
# Function to quantize and convert images to binary
def quantize_and_binarize(images, n_colors=3, threshold_value=240):
	quantized_images = []  # To store quantized images
	binary_images = []  # To store binary images

	for img in images:
		#  Quantize the image
		quantized_image = quantization([img], n_colors=n_colors)[0]
		quantized_images.append(quantized_image)

		# Convert the quantized image to binary
		_, binary_image = cv2.threshold(quantized_image, threshold_value, 255, cv2.THRESH_BINARY)
		binary_images.append(binary_image)

	return np.array(quantized_images), np.array(binary_images)


# Process all images
quantized_images, binary_images = quantize_and_binarize(images, 3, 240)

# Test with the first image: display both quantized and binary versions
#cv2.imshow("Quantized Image (First)", quantized_images[0])
#cv2.imshow("Binary Image (First)", binary_images[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#=====================================

#=====================================
#section f
# Function to calculate bar heights for a single image
def get_bar_heights(image):
    bar_heights = []  # List to store heights of each bar (0-9)
    for bin_id in range(10):  # Loop over all bins (bars)
        height = get_bar_height(image, bin_id)  # Get height of the bar using the provided function
        bar_heights.append(height)
    return bar_heights

# Process all binary images to calculate bar heights
all_bar_heights = []  # To store bar heights for all images
for idx, binary_image in enumerate(binary_images):
    bar_heights = get_bar_heights(binary_image)  # Get bar heights for the current image
    all_bar_heights.append(bar_heights)
   # print(f"Bar heights for Histogram {names[idx]}: {bar_heights}")  # Print bar heights for verification
#=====================================
#=====================================
#section g
# Function to calculate students per bin for a single histogram
def calculate_students_per_bin(bar_heights, max_student_num):
    max_bin_height = max(bar_heights)  # Find the tallest bar
    # Apply the formula for each bin
    return [
        round(max_student_num * bin_height / max_bin_height) if max_bin_height > 0 else 0
        for bin_height in bar_heights
    ]

# function to transcribe all histograms
def transcribe_histograms(names, all_bar_heights, heights_num_list):
    transcriptions = []  # Store transcriptions for all histograms
    for img_id, bar_heights in enumerate(all_bar_heights):
        max_student_num = heights_num_list[img_id]  # Topmost number for this histogram
        # calculate students per bin
        students_per_bin = calculate_students_per_bin(bar_heights, max_student_num)
        # format the transcription
        transcription = f"Histogram {names[img_id]} gave {','.join(map(str, students_per_bin))}"
        transcriptions.append(transcription)
        # Print the transcription
        print(transcription)
    return transcriptions  # Return all transcriptions for further use
# call the function to transcribe all histograms
final_transcriptions = transcribe_histograms(names, all_bar_heights, heights_num_list)
#=====================================
exit()
# The following print line is what you should use when printing out the final result - the text version of each histogram, basically.
# print(f'Histogram {names[id]} gave {heights}')
