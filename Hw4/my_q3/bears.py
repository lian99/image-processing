import cv2
from matplotlib import pyplot as plt


def clean_bears(im):
    equalized_image = cv2.equalizeHist(im)
    gamma = 0.9
    equalized_image = equalized_image ** gamma
    return equalized_image
    # Your code goes here
print("-----------------------image 7----------------------\n")
im7 = cv2.imread(r'Images\bears.tif')
im7 = cv2.cvtColor(im7, cv2.COLOR_BGR2GRAY)
im7_clean = clean_bears(im7)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(im7, cmap='gray', vmin=0, vmax=255)
plt.subplot(1, 2, 2)
plt.imshow(im7_clean, cmap='gray', vmin=0, vmax=255)

plt.show()