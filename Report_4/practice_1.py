import cv2

IMAGE_URL = '/Users/YaguchiYuzuki/PatternProcessing/Report_4/haarcascade_frontalface_alt.xml'

img = cv2.imread("images/input.jpeg", 0)
face_cascade_path = IMAGE_URL
face_cascade = cv2.CascadeClassifier(face_cascade_path)
faces = face_cascade.detectMultiScale(img)

for face in faces:
    x, y, width, height = face
    face_area = img[y: y + height, x: x + width]
    img[y: y + height, x: x + width] = cv2.medianBlur(face_area, 5)

cv2.imshow("result", img)
cv2.imwrite("out/practice_1/output.bmp", img)
cv2.waitKey(0)