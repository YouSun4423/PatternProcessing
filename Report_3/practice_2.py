from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


mnist = fetch_openml('mnist_784', as_frame=False)

X, y = mnist.data, mnist.target

list = []
for i in range(100):
    if mnist['target'][i] == '1':
        list.append(mnist['data'][i])

pca = PCA()
pca.fit(list)

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(pca.components_[i].reshape(28, 28))
    plt.savefig(
        "out/practice_2/output{}.png".format(i)
    )
    plt.show()