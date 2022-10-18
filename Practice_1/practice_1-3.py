import cv2
img_in = cv2.imread("Lenna.bmp", 0)
cv2.imshow("input", img_in)
ret, img_out = cv2.threshold(img_in, 100, 255, cv2.THRESH_BINARY)
cv2.imshow("output", img_out)
cv2.imwrite("output.bmp", img_out)
cv2.waitKey()