import numpy as np
from numpy.linalg import eig, pinv
from scipy.spatial.distance import pdist, squareform

def diffusion_map(X, epsilon=0.01, t=50, alpha=0, dim=2):
    # Define diffusion kernel
    K = np.exp(-squareform(pdist(X))/epsilon)
    
    # Renormalize
    q = K.sum(axis=1)
    Q = np.diag(q)
    Ka = pinv(Q)**alpha @ K @ pinv(Q)**alpha

    # Compute row sums and define P
    da = Ka.sum(axis=1)
    Da = np.diag(da)
    P = pinv(Da) @ Ka

    # Compute eigendecomposition and embedding
    Lam, R = eig(P)
    if dim == 2:
        return np.array([Lam[0]**((1-alpha)*t) * R[0,:], 
                         Lam[1]**((1-alpha)*t) * R[1,:]]).T
    elif dim == 3:
        return np.array([Lam[0]**((1-alpha)*t) * R[0,:], 
                         Lam[1]**((1-alpha)*t) * R[1,:],
                         Lam[2]**((1-alpha)*t) * R[2,:]]).T
    else: raise RuntimeError(f'dim must be 2 or 3 (was {dim}')
