import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import myPCA

data = sp.loadmat("data/toydata.mat")
data2 = sp.loadmat("data/toydata2.mat")


data = data['D']
data2 = data2['data']

def visualizeData(obj, label, svdTrans=False ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.set_label(label)
    fig.text(0.3,0.03, label)

    Axes3D.scatter(ax, xs=data[0,:], ys=data[1,:], zs=data[2,:])
    #Axes3D.scatter(ax, xs=data2[0, :], ys=data2[1, :], zs=data2[2, :])

    # This block works for SVD Transformations
    if svdTrans == True:
        #transD = obj['X_a'].T                                                   # X_a : all three components
        transD = obj['X_b'].T                                                   # X_b : first two components
        #transD = obj['X_c'].T                                                   # X_c : first component only
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



pcaObjSvd = myPCA.usingSVD(data, 1)
pcaObjCov = myPCA.usingCOV(data,1)
visualizeData(pcaObjSvd, "PCA with SVD decomposition")
visualizeData(pcaObjSvd, "PCA with SVD decomposition and transformation", True)
visualizeData(pcaObjCov, "PCA with Covariance matrix ")


