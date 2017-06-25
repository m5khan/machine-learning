""" 
% K-Means reference : http://flothesof.github.io/k-means-numpy.html
"""


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def initialize_centroids(points, k):
    """returns k centroids from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]


def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    print(np.array([points[closest == k]for k in range(centroids.shape[0])]))
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])


def visualizeData(data, centroids, label):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.text(0.3, 0.03, label)
    Axes3D.scatter(ax, xs=data[:, 0], ys=data[:, 1], zs=data[:, 2], alpha=0.1)
    Axes3D.scatter(ax, xs=centroids[:, 0], ys=centroids[:, 1], zs=centroids[:,2], c='r', s=100, alpha=1)
    plt.show()