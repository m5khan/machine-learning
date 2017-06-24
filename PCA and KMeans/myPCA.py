"""
    % This is the PCA implementation
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

# This function should implement the PCA using the Singular Value
#    % Decomposition (SVD) of the given dataMatrix
#        %
#        % Input:
#        %       dataMatrix (nrDims x nrSamples)
#        %       desiredVariancePercentage (%)
#        % Output is a structure
#        %       eigvecs: eigenvectors
#        %       eigvals: eigenvalues
#        %       meanDataMatrix
#        %       demeanedDataMatrix
#        %       projectedData


def usingSVD(dataMatrix, desiredVariancePercentage=1.0):
    # This function should implement the PCA using the Singular Value
    # Decomposition (SVD) of the given dataMatrix
    obj = {'meanDataMatrix':None, 'demeanedDataMatrix' : None, 'X_a':None, 'X_b': None, 'X_c': None, 'eigvecs':None, 'eigvals':None}
    # De-Meaning the feature space
    obj['meanDataMatrix'] = dataMatrix.mean(axis=1)
    obj['demeanedDataMatrix'] = dataMatrix - obj['meanDataMatrix'][:, np.newaxis]
    
    # SVD Decomposition
    # You need to transpose the data matrix

    transposedData = obj['demeanedDataMatrix'].T
    U, s, V = np.linalg.svd(transposedData, full_matrices=False)
    print(U.shape, V.shape , s.shape)
    S = np.diag(s)
    X_a = np.dot(np.dot(U, S), V)
    X_b = np.dot(np.dot(U[:,0:-1], S[0:2,0:2]), V[0:-1,:])

    X_c = np.dot(np.dot(U[:,0].reshape(U.shape[0],1), S[0,0].reshape(1,1)), V[0,:].reshape(1, V.shape[1]))

    print("standard deviation: original | transformed | difference")
    print(np.std(transposedData), np.std(X_a), np.std(transposedData - X_a))
    print(np.std(transposedData), np.std(X_b), np.std(transposedData - X_b))
    print("-------------------------------------------------------")

    # Enforce a sign convention on the coefficients -- the largest element (absolute) in each
    # column will have a positive sign.
    maxVals = np.amax(X_b, axis=0)
    print(maxVals)      # max vals are positive

    obj['X_a'] = X_a
    obj['X_b'] = X_b
    obj['X_c'] = X_c

    # Compute the accumelative Eigenvalues to finde the desired
    # Variance
    eigvals = s
    
    # Keep the eigenvectors and eigenvalues of the desired
    # variance, i.e. keep the first two eigenvectors and
    # eigenvalues if they have 90% of variance.
    eigvecs = V.T           #since svd is USV^T


    obj['eigvecs'] = eigvecs
    obj['eigvals'] = eigvals
    
    
    # Project the data
    obj['projectedData'] = np.dot(transposedData, eigvecs[:,0:2])
    
    # return the object
    return obj


# This function should implement the PCA using the EigenValue
#    % Decomposition of the given Covariance Matrix
#        %
#        % Input:
#        %       dataMatrix (nrDims x nrSamples)
#        %       desiredVariancePercentage (%)
#        % Output is a structure
#        %       eigvecs: eigenvectors
#        %       eigvals: eigenvalues
#        %       meanDataMatrix
#        %       demeanedDataMatrix
#        %       projectedData

 
def usingCOV(dataMatrix, desiredVariancePercentage=1.0):
    pass
    # This function should implement the PCA using the
    # EigenValue Decomposition of a given Covariance Matrix 
    
    # De-Meaning the feature space 
    #obj.meanDataMatrix =
    #obj.demeanedDataMatrix =
            
    # Computing the Covariance 
    #obj.covMatrix =
            
    # Eigen Value Decomposition
    
    
    
    # In COV, you need to order the eigevectors according to largest eigenvalues
    
    
    
    
    # Enforce a sign convention on the coefficients -- the largest element (absolute) in each
    # column will have a positive sign.


    # Compute the accumelative Eigenvalues to finde the desired
    # Variance 
    
    # Keep the eigenvectors and eigenvalues of the desired
    # variance, i.e. keep the first two eigenvectors and
    # eigenvalues if they have 90% of variance. 
    #obj.eigvecs =
    #obj.eigvals =
            
    # Project the data 
    #obj.projectedData =
    
    # return the object
    #return obj