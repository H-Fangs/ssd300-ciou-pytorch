import cv2
import numpy as np

# img1 = cv2.imread("img_1.jpg")
img1 = cv2.imread("img_2.jpg")

img1_mirror = cv2.flip(img1, 0)
cv2.namedWindow("img_1", 0)
cv2.namedWindow("img1_mirror", 0)

cv2.imshow("img_1", img1)
cv2.imshow("img1_mirror", img1_mirror)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(1)