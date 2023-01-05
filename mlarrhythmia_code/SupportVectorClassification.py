from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from PrincipalComponentAnalysis import GetDataPCA

_, _, data = GetDataPCA(0.99)
#data['Class'] = data['Class'].apply(lambda x: 1 if x > 1 else 0)
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

kernals = ['linear', 'poly', 'rbf', 'sigmoid']
for ker in kernals:
    model = svm.SVC(kernel=ker)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #print(confusion_matrix(y_test, y_pred))
    print(ker)
    print(classification_report(y_test, y_pred))

'''
kernalScore = np.zeros(4)
for iteration in range(1, 101):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    for idx, ker in enumerate(kernals):
        clf = svm.SVC(kernel=ker).fit(X_train, y_train)
        kernalScore[idx] += clf.score(X_test, y_test)

    if iteration%10 == 0:
        print(kernalScore/iteration, iteration)'''