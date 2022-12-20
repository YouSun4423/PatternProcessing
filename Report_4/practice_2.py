from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC


class MachineLearningMethod:
    def knearest_neighbor_method(X_train, X_test, y_train, y_test, i):
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X_train, y_train)
        print('k最近傍法')
        print('正答率(データ数:{}):{}'.format(i, clf.score(X_test, y_test)))
    
    def linear_discriminant_method(X_train, X_test, y_train, y_test, i):
        clf = LDA()
        clf.fit(X_train, y_train)
        print('線形判別分析')
        print('正答率(データ数:{}):{}'.format(i, clf.score(X_test, y_test)))
    
    def linear_SVM_method(X_train, X_test, y_train, y_test, i):
        clf = SVC(kernel="linear")
        clf.fit(X_train, y_train)
        print('線形SVM')
        print('正答率(データ数:{}):{}'.format(i, clf.score(X_test, y_test)))
    
    def nonlinear_SVM_method(X_train, X_test, y_train, y_test, i):
        clf = SVC(kernel="rbf")
        clf.fit(X_train, y_train)
        print('非線形SVM')
        print('正答率(データ数:{}):{}'.format(i, clf.score(X_test, y_test)))


def main():
    MLmethod = MachineLearningMethod
    mnist = fetch_openml('mnist_784', as_frame=False)

    for i in [100, 1000, 10000]:
        X, y = mnist.data[0:i], mnist.target[0:i]
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        MLmethod.knearest_neighbor_method(X_train, X_test, y_train, y_test, i)
        MLmethod.linear_discriminant_method(X_train, X_test, y_train, y_test, i)
        MLmethod.linear_SVM_method(X_train, X_test, y_train, y_test, i)
        MLmethod.nonlinear_SVM_method(X_train, X_test, y_train, y_test, i)
        print("\n")


if __name__ == "__main__":
    main()
