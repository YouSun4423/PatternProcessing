import cv2
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

mnist = fetch_openml('mnist_784', as_frame=False)

X, y = mnist.data, mnist.target

# 主成分の個数
D = 10

# D個の主成分を返却
pca = PCA(n_components=D)

# 次元削減
X_trans = pca.fit_transform(X)
# 次元復元
X_inv = pca.inverse_transform(X_trans)

for i in range(5):
    cv2.imshow("sample", X_inv[i].reshape(28, 28))
    cv2.imwrite(
        "out/practice_3/output{}.bmp".format(i), 
        X_inv[i].reshape(28, 28)
    )
    cv2.waitKey()
