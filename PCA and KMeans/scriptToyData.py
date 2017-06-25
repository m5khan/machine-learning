import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import myPCA
import Kmeans

data = sp.loadmat("data/toydata.mat")
data2 = sp.loadmat("data/toydata2.mat")
HeartData = sp.loadmat("data/filtHeartDataSet.mat")


data = data['D']
data2 = data2['data']

def visualizeData(obj, label, svdTrans=False, components=2, colorLabel=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.text(0.3,0.03, label)

    Axes3D.scatter(ax, xs=data[0,:], ys=data[1,:], zs=data[2,:])

    # This block works for SVD Transformations
    if svdTrans == True:
        if components == 3:
            transD = obj['X_a'].T                                                   # X_a : all three components
        elif components == 2:
            transD = obj['X_b'].T                                                   # X_b : first two components
        elif components == 1:
            transD = obj['X_c'].T                                                   # X_c : first component only
        if colorLabel != None:
            Axes3D.scatter(ax, xs=transD[0,:], ys=transD[1,:], zs=transD[2,:], c=colorLabel, cmap=plt.cm.cool)
        else:
            Axes3D.scatter(ax, xs=transD[0,:], ys=transD[1,:], zs=transD[2,:], c="red")
    else:
        # This block works for poth SVD and COV
        transD = obj['projectedData'].T
        Axes3D.scatter(ax, xs=transD[0,:], ys=transD[1,:], c="red")

    kwargs = {'length':3.0, 'pivot':'tail'}

    soa = np.array([
                    np.concatenate(([0,0,0], obj['eigvecs'][:,0])),
                    np.concatenate(([0, 0, 0], obj['eigvecs'][:,1])),
                    np.concatenate(([0, 0, 0], obj['eigvecs'][:,2])),
                    ])
    X, Y, Z, U, V, W = zip(*soa)
    ax.quiver(X, Y, Z, U, V, W,**kwargs, color="green")

    soa2 = np.array([
                    np.concatenate((obj['meanDataMatrix'], obj['eigvecs'][:, 0])),
                    np.concatenate((obj['meanDataMatrix'], obj['eigvecs'][:, 1])),
                    np.concatenate((obj['meanDataMatrix'], obj['eigvecs'][:, 2])),
                    ])

    X2, Y2, Z2, U2, V2, W2 = zip(*soa2)
    ax.quiver(X2, Y2, Z2, U2, V2, W2, **kwargs, color="red")

    plt.show()


def runExercise1():
    # Exercise 1_A
    # PCA on toydata
    pcaObjSvd = myPCA.usingSVD(data, 1)
    pcaObjCov = myPCA.usingCOV(data,1)
    visualizeData(pcaObjSvd, "PCA with SVD decomposition")
    visualizeData(pcaObjSvd, "PCA with SVD decomposition and transformation", True)
    visualizeData(pcaObjCov, "PCA with Covariance matrix ")

    # Exercise 1_B
    # this is PCA of Heart data
    hDataVectors, hDataLabels = HeartData['dataMatrix'] , HeartData['labels']
    heartObj = myPCA.usingSVD(hDataVectors.T)
    heartObj['eigvecs'] = heartObj['eigvecs'][0:3,0:3]
    heartObj['meanDataMatrix'] = heartObj['meanDataMatrix'][0:3]
    #visualizeData(heartObj, "Heart Desease PCA", True, components=3, colorLabel=hDataLabels)
    visualizeData(heartObj, "Heart Desease PCA", True, components=3)


# Exercise 2
# K-MEANS
def runExercise2():
    toydata = data2.T
    centroids = Kmeans.initialize_centroids(toydata,3)
    Kmeans.visualizeData(toydata, centroids, "KMEANS Initial")

    newCentroids = np.zeros(centroids.shape)
    iteration = 1
    while (np.all(newCentroids - centroids != 0)):
        newCentroids = centroids
        closestCentroids = Kmeans.closest_centroid(toydata, newCentroids)
        centroids = Kmeans.move_centroids(toydata, closestCentroids, newCentroids)
        Kmeans.visualizeData(toydata,centroids, "KMEANS interation {}".format(iteration))
        iteration +=1

    Kmeans.visualizeData(toydata, centroids, "KMEANS Final positions")


if(__name__ == "__main__"):
    runExercise1()
    runExercise2()