# 画像の読み込みと表示
import cv2

img = cv2.imread("Lenna.bmp")
img_gray = cv2.imread("Lenna.bmp", 0)
cv2.imshow("image", img)
cv2.imshow("image_gray", img_gray)

cv2.waitKey()