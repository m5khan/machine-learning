import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

data = genfromtxt('01_homework_dataset.csv', delimiter=',', skip_header=1)

class DT:
    def __init__(self):
        pass

    def giniIndex(self, node, c1=0, c2=1, c3=2):
        total = np.shape(node)[0]
        if total == 0:
            return 0
        nc1, nc2, nc3 = 0,0,0
        for rows in node:
            if rows[3] == c1:
                nc1 += 1
            elif rows[3] == c2:
                nc2 += 1
            elif rows[3] == c3:
                nc3 += 1
        return 1 - (nc1/total)**2 - (nc2/total)**2 - (nc3/total)**2

    # split improvement
    def delta_i(self, splitValue, featureNo, node):
        i_t = self.giniIndex(node)
        lNode, rNode = self.splitMatrix(node, splitValue, featureNo)
        i_tL = self.giniIndex(lNode)
        i_tR = self.giniIndex(rNode)
        total = np.shape(node)[0]
        total_l = np.shape(lNode)[0]
        total_r = np.shape(rNode)[0]
        delta = i_t - ((total_l/total)*(i_tL)) - ((total_r/total)*(i_tR))
        return delta


    def splitMatrix(self, node, splitVal, cNumber):
        lNode, rNode = [], []
        for row in node:
            if (row[cNumber] <= splitVal):
                lNode.append(row)
            else:
                rNode.append(row)
        return np.array(lNode), np.array(rNode)

    def findBestSplit(self, node, feature):
        for row in node:
            splitVal = row[feature]
            gain = self.delta_i(splitVal, feature, node)
            print("split value: {}  | gain : {}".format(splitVal, gain))

if __name__ == '__main__':
    dT = DT()
    gini = dT.giniIndex(data)
    print("Gini index at root node: {}".format(gini))
    print("find best split value (gain) on feature 1")
    dT.findBestSplit(data, 0)
    print("Most gain on value 4.1")
    node1L, node1R = dT.splitMatrix(data, 4.1, 0)
    gini_1L = dT.giniIndex(node1L)
    print("Gini index at depth 1; left node: {}".format(gini_1L))
    gini_1R = dT.giniIndex(node1R)
    print("Gini index at depth 1; right node: {}".format(gini_1R))
    print("Left node at depth 1 is pure")
    print("splitting right node...")
    print("finding best value for right node at depth 1. Checking feature 1")
    dT.findBestSplit(node1R, 0)
    print("Most gain on value 6.9")
    node2L, node2R = dT.splitMatrix(node1R, 6.9, 0)
    gini_2L = dT.giniIndex(node2L)
    print("Gini index at depth 2; left node: {}".format(gini_2L))
    gini_2R = dT.giniIndex(node2R)
    print("Gini index at depth 2; right node: {}".format(gini_2R))