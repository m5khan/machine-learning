import Exercise3 as Ex3
import numpy as np
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class TBDataVisualizer(Ex3.RForestVisualizer):
    data = ""

    def __init__(self):
        super(TBDataVisualizer, self).__init__()

    def loadData(self):
        self.data = Ex3.Util().load_csv("TuberculosisData.csv")

    def classifyRF(self):
        print("-----------+RANDOM FOREST CLASSIFIER+-------------")
        forestClf = ensemble.RandomForestClassifier(20, max_depth=4)
        foldId = 1
        for dset in self.kFolds:
            dtrain, dtest = dset[0], dset[1]
            X_train, X_test = dtrain[:, 0:-1], dtest[:, 0:-1]
            y_train, y_test = dtrain[:, -1], dtest[:, -1]
            forestClf.fit(X_train, y_train)
            meanAccuracy = forestClf.score(X_test, y_test) * 100
            print("Mean Accuracy for fold {} = {}".format(foldId, meanAccuracy))
            prediction = forestClf.predict(X_test)
            cnfMat = confusion_matrix(y_train, prediction)
            print("confusion matrix for fold {} dataset".format(foldId))
            print(cnfMat)
            print("\r\n")
            foldId += 1


    def classifySVM(self):
        print("\r\n-----------+SVM CLASSIFIER+-------------")
        svc = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001,
            cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,
            random_state=None)
        foldId = 1
        for dset in self.kFolds:
            dtrain, dtest = dset[0], dset[1]
            X_train, X_test = dtrain[:, 0:-1], dtest[:, 0:-1]
            y_train, y_test = dtrain[:, -1], dtest[:, -1]
            svc.fit(X_train, y_train)
            meanAccuracy = svc.score(X_test, y_test) * 100
            print("Mean Accuracy for fold {} = {}".format(foldId, meanAccuracy))
            prediction = svc.predict(X_test)
            cnfMat = confusion_matrix(y_train, prediction)
            print("confusion matrix for fold {} dataset".format(foldId))
            print(cnfMat)
            print("\r\n")
            foldId += 1


    def classifyLogReg(self):
        print("\r\n-----------+LOGISTIC REGRESSION CLASSIFIER+-------------")
        logreg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                           class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
                           verbose=0, warm_start=False, n_jobs=1)
        foldId = 1
        for dset in self.kFolds:
            dtrain, dtest = dset[0], dset[1]
            X_train, X_test = dtrain[:, 0:-1], dtest[:, 0:-1]
            y_train, y_test = dtrain[:, -1], dtest[:, -1]
            logreg.fit(X_train, y_train)
            meanAccuracy = logreg.score(X_test, y_test) * 100
            print("Mean Accuracy for fold {} = {}".format(foldId, meanAccuracy))
            prediction = logreg.predict(X_test)
            cnfMat = confusion_matrix(y_train, prediction)
            print("confusion matrix for fold {} dataset".format(foldId))
            print(cnfMat)
            print("\r\n")
            foldId += 1


if __name__ == '__main__':
    dv = TBDataVisualizer()
    dv.loadData()
    dv.createFoldsSklearn(2)
    dv.classifyRF()
    dv.classifySVM()
    dv.classifyLogReg()

