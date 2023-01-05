from sklearn import svm
from sklearn.model_selection import train_test_split
from PrincipalComponentAnalysis import GetDataPCA
import numpy as np

classes, _, data = GetDataPCA(0.99)

X = data.drop("Class", axis=1)
y = data["Class"]

kernals = ["linear", "poly", "rbf", "sigmoid"]

kernalScore = np.zeros(4)
for iteration in range(1, 101):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    for idx, ker in enumerate(kernals):
        clf = svm.SVC(kernel=ker).fit(X_train, y_train)
        kernalScore[idx] += clf.score(X_test, y_test)

    if iteration%10 == 0:
        print(kernalScore/iteration, iteration)