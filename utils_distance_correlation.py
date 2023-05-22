"""Computes the distance correlation between two matrices.
https://en.wikipedia.org/wiki/Distance_correlation
"""

import dcor
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm


# __author__ = "Kailash Budhathoki"
# __email__ = "kbudhath@mpi-inf.mpg.de"
# __copyright__ = "Copyright (c) 2019"
# __license__ = "MIT"


# def dcov(X, Y):
#     """Computes the distance covariance between X and Y.
#     """
#     n = X.shape[0]
#     return np.sqrt(np.sum(X*Y)) / n


# def dvar(X):
#     """Computes the distance variance of a X.
#     """
#     n = X.shape[0]
#     return np.sqrt(np.sum(X ** 2 / n**2))


# def cent_dist(X):
#     """ Computes the pairwise euclidean distance between rows of X and centers
#         each cell of the distance matrix with row mean, column mean, and grand mean.
#     """
#     X = X.reshape(-1, 1)        # reshape to 2D array
#     M = squareform(pdist(X))    # distance matrix
#     rmean = M.mean(axis=1)
#     cmean = M.mean(axis=0)
#     gmean = rmean.mean()
#     R = np.tile(rmean, (M.shape[0], 1)).transpose()
#     C = np.tile(cmean, (M.shape[1], 1))
#     G = np.tile(gmean, M.shape)
#     CM = M - R - C + G
#     return CM


# def dcor(X, Y):
#     """ Computes the distance correlation between two matrices X and Y.
#         X and Y must have the same number of rows.
#     """
#     assert X.shape == Y.shape

#     A, B = cent_dist(X), cent_dist(Y)

#     dcov_AB = dcov(A, B)
#     dvar_A, dvar_B = dvar(A), dvar(B)

#     dcor = 0.0
#     if dvar_A > 0.0 and dvar_B > 0.0:
#         dcor = dcov_AB / np.sqrt(dvar_A * dvar_B)

#     return dcor


def df_distance_correlation(df_train, stocks):
    
    #initializes an empty DataFrame
    df_train_dcor = pd.DataFrame(index=stocks, columns=stocks)
    
    #initialzes a counter at zero
    k=0
    
    #iterates over the time series of each stock
    for i in tqdm(stocks):
        
        #stores the ith time series as a vector
        v_i = df_train.loc[:, i].values
        
        #iterates over the time series of each stock subect to the counter k
        for j in stocks[k:]:
            
            #stores the jth time series as a vector
            v_j = df_train.loc[:, j].values
            
            #computes the dcor coefficient between the ith and jth vectors
            dcor_val = dcor.distance_correlation(v_i, v_j)
            
            #appends the dcor value at every ij entry of the empty DataFrame
            df_train_dcor.at[i,j] = dcor_val
            
            #appends the dcor value at every ji entry of the empty DataFrame
            df_train_dcor.at[j,i] = dcor_val
        
        #increments counter by 1
        k+=1
    
    #returns a DataFrame of dcor values for every pair of stocks
    return df_train_dcor