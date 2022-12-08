import cv2
import numpy as np
from sklearn.datasets import fetch_openml


mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.targets


for i in range(100):
    cv2.imshow("sample", X[i].reshape(28,28)) print(y[i])
    cv2.waitKey()