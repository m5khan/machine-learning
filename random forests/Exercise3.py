import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import model_selection

class Util:

    def __init__(self):
        pass

    # load csv files
    @staticmethod
    def load_csv(fileName, path="./"):
        return np.loadtxt(path + fileName, delimiter=',', skiprows=1)


class RForestVisualizer:
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

    # visualize the data points and draw contour graph
    # X : data with 3 features X1, X2, Y
    # kFolds : array of length k of tuples (train_Data, test_Data)
    # clf : classifier
    def visualizeLearningKfold(self,X, kFolds, clf):
        h = 0.02
        no_of_folds = len(kFolds)
        plt.figure(figsize=(20, 10))
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        cm = plt.cm.RdBu
        #cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        cm_bright = plt.cm.cool

        plotId, kfoldNo = 1, 1
        for dset in kFolds:
            dtrain, dtest = dset[0], dset[1]
            X_train, X_test = dtrain[:, 0:2], dtest[:, 0:2]
            y_train, y_test = dtrain[:,-1], dtest[:, -1]

            ax = plt.subplot(no_of_folds, 2, plotId)
            ax.set_title("Fold : {}".format(kfoldNo))
            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())

            # ---- plotting diecision boundaries-------------
            ax = plt.subplot(no_of_folds, 2, plotId+1)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)       #mean accuracy

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
            ax.set_title("contour for fold {}".format(kfoldNo))
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            plotId += 2
            kfoldNo +=1

        plt.show()



if __name__ == "__main__":
    dl = RForestVisualizer()

    # Switch data from here
    dl.loadSpiralData()
    #dl.loadTwistData()

    # partition data into 2 folds
    dl.createFoldsSklearn(2)

    forest10 = ensemble.RandomForestClassifier(50, max_depth=100)
    print("visualizing")
    dl.visualizeLearningKfold(dl.data, dl.kFolds, forest10)