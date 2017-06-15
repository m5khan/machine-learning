import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import model_selection
from matplotlib.colors import ListedColormap

class Util:

    def __init__(self):
        pass

    # load csv files
    @staticmethod
    def load_csv(fileName, path="./"):
        return np.loadtxt(path + fileName, delimiter=',', skiprows=1)


class DataLoader:
    data = 0        #holds data that is loaded i.e spiral data or twistData
    trainSet = 0
    testSet = 0
    kFolds = []     # contains tuples of k (train_k, test_k) data manifolds

    def __init__(self):
        pass


    def loadSpiralData(self):
        self.data = Util.load_csv("SpiralData.csv")


    def loadTwistData(self):
        self.data = Util.load_csv("TwistData.csv")


    # partitions the data into k folds
    # k :   Number of folds
    # preserveData  :   Default True, preserves the data as an instance variables
    def createFolds(self, k, preserveData=True):
        rows = self.data.shape[0]
        folds = math.floor(rows/k)
        trains = folds * (k-1)
        trainSet = self.data[0:trains, :]
        testSet = self.data[trains:, :]
        if preserveData == True:
            self.trainSet = trainSet
            self.testSet = testSet
        return (trainSet, testSet)


    def createFoldsSklearn(self, k, shuffle=True):
        kfolds = model_selection.KFold(k, shuffle)
        splitted = kfolds.split(self.data)
        for train_index, test_index in splitted:
            self.kFolds.append((self.data[train_index], self.data[test_index]))


    def visualizeData(self, X, Z, W=None, b=None):
        plt.scatter(X[:,0], X[:,1], c=Z, cmap=plt.cm.get_cmap("cool"))
        plt.show()


    def visualizeLearning(self, X, X_train, y_train, X_test, y_test, clf):
        h = 0.02
        plt.figure(figsize=(10, 5))
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        cm = plt.cm.RdBu
        #cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        cm_bright = plt.cm.cool
        ax = plt.subplot(1, 2, 1)
        ax.set_title("Random Forest")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        # ---- plotting diecision boundaries
        ax = plt.subplot(1, 2, 2)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title("Random Forest")
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')

        plt.show()

    def mapper(a):
        if a == 0.2:
            a = 2
        elif a>=0 and a < 1:
            a = 3
        else: a = 1
        return a

if __name__ == "__main__":
    dl = DataLoader()
    # Switch data from here
    #dl.loadSpiralData()
    dl.loadTwistData()

    # partition data into 2 folds
    dl.createFoldsSklearn(2)

    train1 = dl.kFolds[0][0]            #1st index: k in kfolds, 2nd index: 0 for train, 1 for test
    test1 = dl.kFolds[0][1]

    forest10 = ensemble.RandomForestClassifier(10, max_depth=3)
    print("visualizing")
    dl.visualizeLearning(dl.data, train1[:,0:2], train1[:,-1], test1[:,0:2], test1[:,-1], forest10)