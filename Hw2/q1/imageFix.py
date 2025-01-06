# Lian Natour, 207300443
# Mohammad Mhneha, 315649814

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import matplotlib.pyplot as plt
import numpy as np


# histogram equalization
def histogram_equalization(image):
    # Applying histogram equalization to enhance the contrast
    return cv2.equalizeHist(image)
# gamma correction
def gamma_correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table_for_gamma = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table_for_gamma)

def apply_fix(image, id):
	if id == 1:
		# Apply histogram equalization for the first image
		return histogram_equalization(image)
	elif id == 2:
		# Apply gamma correction for the second image
		gamma = 2.1 # Example value, adjust based on image analysis
		return gamma_correction(image, gamma)
	elif id == 3:
		# leave image unchanged
		return image




for i in range(1,4):
	if i == 1:
		path = f'{i}.png'
	else:
		path = f'{i}.jpg'
	image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	fixed_image = apply_fix(image, i)
	plt.imsave(f'{i}_fixed.jpg', fixed_image, cmap='gray', vmin=0,vmax=255)
