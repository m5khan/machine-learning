"""
    % K-Means Implementation
    % This is the outline of your first excercise in Machine Learning for
    % Medical Application (MLMI) practical course
    % --------------------------------------------------------------------------------------------
    % Author
    % Shadi Albarqouni, PhD Candidate @ CAMP-TUM.
    % Conatct:               shadi.albarqouni@tum.de
    % --------------------------------------------------------------------------------------------
    % Copyright (c) 2016 TU Munich.
    % All rights reserved.
    % This work should be used for nonprofit purposes only.
    % --------------------------------------------------------------------------------------------
"""

import numpy as np
import scipy.io as sio

#   % This function should assign the points in your dataMatrix to the
#    % closest centroid, the distance is computed using the L2 norm
#        %
#        % Input:
#        %       dataMatrix (nrDims x nrSamples)
#        %       centroids (nrDims x nrCentroids)
#        % Output
#        %       assigedPoints (1 x nrSamples) should have the centroid's
#        %       index

def assignPoints(dataMatrix, centroids):
    
    # return the assigned points
    return assignedPoints




#    % This function should assign the points in your dataMatrix to the
#    % closest centroid, the distance is computed using the L2 norm
#        %
#        % Input:
#        %       dataMatrix (nrDims x nrSamples)
#        %       assigedPoints (1 x nrSamples)
#        % Output
#        %       updatedCentroids (nrDims x nrCentroids)


def updateCentroids(dataMatrix, assignedPoints):

    
    # return the updated centroids
    return updatedCentroids




#   % This function should compute the cost function
#    %
#        % Input:
#        %       dataMatrix (nrDims x nrSamples)
#        %       centroids (nrDims x nrCentroids)
#        %       assignedPoints (1 x nrSamples)
#        % Output
#        %       cost


def computeCost(dataMatrix, centroids, assignedPoints)


return cost




#    % This function should run the K-Means algorithm
#    %
#        % Input:
#        %       dataMatrix (nrDims x nrSamples)
#        %       numberOfCluster
#        %       numberOfRuns
#        %       Tol
#        % Output
#        %       object


def runKmeans(dataMatrix, numberOfCluster, numberOfRuns, Tol)
    
#    % You can neglect the number of runs at the begining, once you
#    % are done, you can work on it. The idea is to run the Kmeans
#    % several times (runs), then choose the one giving you the min.
#    % cost.
            
            
#    % ....
#    Note that you need to change any necessary syntax for python

            for i = 1:numberOfRuns
                % initializa Centroids
                rndCen = randperm(nrSamples);
                centroids = dataMatrix(:,rndCen(1:objKmeans.numberOfCluster));
                
                if nargin <=3
                    Tol = 1e-6;
                end
                
                iter = 0; RE = 1;
                
                while abs(RE) >= Tol % convergance criterion
                    iter = iter + 1;
                    
                    % .....
                    
                    
                    cost(iter) =
                    
                    % check the convergance
                    if iter == 1;
                        RE = 1;
                    else
                        RE = (cost(iter) - cost(iter-1))/max(cost(iter-1), cost(iter));
                    end
                
                end
                
                
                % ......
        
            end
        
        
return objKmeans


#        % This function should run the K-Means algorithm for visualization
#        % Input:
#        %       dataMatrix (nrDims x nrSamples)
#        %       numberOfCluster
#        % Output
#        %       object
        
def runKmeansVis(dataMatrix, numberOfCluster)
        
        
return objKmeansVis

