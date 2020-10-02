import numpy as np
#import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
from numpy.random import randint
from random import sample
from sklearn import decomposition
from sklearn.datasets import load_breast_cancer

def GetData(pts = 100):
    mean = randint(-100,100, size=2)

    std = randint(10,500, size=2)
    cov = randint(10,std.min())
    cov = np.diag(std)+np.array([[0,cov],[cov,0]])

    X = np.random.multivariate_normal(mean, cov, pts)

    print("DATA")
    print("mean: ",mean)
    print("std: ",std)
    print("cov:\n",cov,"\n")

    return X

def PCA_skl(X,pts):
    print("####################################################################")

    # Covariance Matrix
    Cov = np.dot(X.T, X) / (pts - 1)
    print("Covariance Matrix:\n", Cov, "\n")

    # singular value descomposition of original covariance matrix
    v, s, vh = np.linalg.svd(Cov, full_matrices=True)
    print("V:\n", v)
    print("S:\n", s)
    print("Vh:\n", vh, "\n\n")

    # Plot Data & Main EigenVectors
    start = [[0, 0], [0, 0]]

    plt.figure()
    eigenv = np.dot(np.diag(s), v)
    ###############################
    plt.subplot(121)
    plt.title("Centered Data (sklearn)")
    plt.scatter(X[:, 0], X[:, 1], 5)
    plt.quiver(*start, eigenv[:, 0] / 10, eigenv[:, 1] / 10, color=['r', 'b'], angles='xy', scale_units='xy', scale=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    ###############################

    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)

    # eigenvectors of transform covariance
    Cov_pca = np.dot(X_pca.T, X_pca) / (pts - 1)
    print("Transform Covariance Matrix:\n", Cov_pca, "\n")
    v, s, vh = np.linalg.svd(Cov_pca, full_matrices=True)
    print("V:\n", v)
    print("S:\n", s)
    print("Vh:\n", vh, "\n")

    # Plot Rotated Data & Main EigenVectors
    eigenv = np.dot(np.diag(s), vh.T)
    ###############################
    plt.subplot(122)
    plt.title("PCA (sklearn)")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], 5)
    plt.quiver(*start, eigenv[:, 0] / 10, eigenv[:, 1] / 10, color=['r', 'b'], angles='xy', scale_units='xy', scale=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    ###############################
    print("####################################################################")


def PCA_svd(X,pts):
    print("####################################################################")
    '''
    Cov = X / np.sqrt(pts - 1)
    #print("Covariance Matrix:\n", Cov, "\n")
    u, s, vh = np.linalg.svd(Cov, full_matrices=True)
    print("U:\n", u)
    print("S:\n", s)
    print("Vh:\n", vh, "\n\n")
    '''

    #Covariance Matrix
    Cov = np.dot(X.T,X)/(pts-1)
    print("Covariance Matrix:\n", Cov,"\n")

    #singular value descomposition of original covariance matrix
    v , s, vh = np.linalg.svd(Cov, full_matrices=True)
    print("V:\n", v)
    print("S:\n", s)
    print("Vh:\n", vh,"\n\n")


    #Plot Data & Main EigenVectors
    start = [[0, 0], [0, 0]]

    plt.figure()
    eigenv = np.dot(v, np.diag(s)).T
    ###############################
    plt.subplot(121)
    plt.title("Centered Data (SVD)")
    plt.scatter(X[:,0], X[:,1], 5)
    plt.quiver(*start, eigenv[:,0]/10, eigenv[:,1]/10, color=['r','b'], angles='xy', scale_units='xy', scale=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    ###############################

    # Project X onto PC space
    X_pca = np.dot(X, vh.T)
    
    #eigenvectors of transform covariance
    Cov_pca = np.dot(X_pca.T, X_pca)/(pts - 1)
    print("Transform Covariance Matrix:\n", Cov_pca, "\n")
    v, s, vh = np.linalg.svd(Cov_pca, full_matrices=True)
    print("V:\n", v)
    print("S:\n", s)
    print("Vh:\n", vh,"\n")

    # Plot Rotated Data & Main EigenVectors
    eigenv = np.dot(v, np.diag(s)).T
    ###############################
    plt.subplot(122)
    plt.title("PCA (SVD)")
    plt.scatter(X_pca[:, 0], X_pca[:,1], 5)
    plt.quiver(*start, eigenv[:,0]/10, eigenv[:,1]/10, color=['r','b'], angles='xy', scale_units='xy', scale=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    ###############################
    print("####################################################################")



def PCA_eigen(X,pts):
    print("####################################################################")
    #Covariance Matrix
    Cov = np.dot(X.T, X) / (pts - 1)
    print("Covariance Matrix:\n", Cov)

    #eigen descomposicion of original covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eigh(Cov)
    print("eigen vec:\n", eigen_vecs)
    print("eigen val:\n", eigen_vals, "\n")

    vt = np.dot(eigen_vecs, np.diag(eigen_vals)).T

    plt.figure()

    start = [[0, 0], [0, 0]]
    ###############################
    plt.subplot(121)
    plt.title("Centered Data (EigenDecomposition)")
    plt.scatter(X[:, 0], X[:, 1], 5)
    plt.quiver(*start, vt[:,0]/10, vt[:,1]/10, color=['r','g'], angles='xy', scale_units='xy',
               scale=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    ###############################

    # Project X onto PC space
    X_pca = np.dot(X, eigen_vecs)

    # eigenvector of normalice covariance
    Cov_pca = np.dot(X_pca.T, X_pca) / (pts - 1)
    eigen_vals, eigen_vecs = np.linalg.eigh(Cov_pca)
    print("eigen vec_2:\n", eigen_vecs)
    print("eigen val_2:\n", eigen_vals)

    vt = np.dot(eigen_vecs, np.diag(eigen_vals)).T

    ###############################
    plt.subplot(122)
    plt.title("PCA (EigenDecomposition)")
    plt.scatter(X_pca[:,0], X_pca[:,1], 5)  # , c=np.sqrt(X[:,0] ** 2 + X[:, 1] ** 2))
    plt.quiver(*start, vt[:,0]/10, vt[:,1]/10, color=['r','g'], angles='xy', scale_units='xy',
               scale=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    ###############################
    print("####################################################################")


if __name__ == '__main__':
    #Number of sample data
    pts = 500

    X = GetData(pts)

    #PRE-PROC
    X = X - np.mean(X, axis=0)  #center data

    PCA_skl(X,pts)

    PCA_svd(X,pts)

    PCA_eigen(X, pts)

    plt.show()