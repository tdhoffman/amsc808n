import numpy as np
from numpy.linalg import eig, solve, matrix_power
from sklearn.metrics.pairwise import euclidean_distances

def diffusion_map(X, eps=None, delta=0.2, alpha=0, dim=2):
    dists2 = euclidean_distances(X, X)**2
    
    # Choose epsilon
    if eps is None:
        np.fill_diagonal(dists2, np.inf)
        epsilon = dists2.min(axis=1).mean() 
        epsilon *= 1000
    else:
        epsilon = eps
        
    # Define diffusion kernel
    K = np.exp(-dists2/epsilon)

    # Renormalize
    q = K.sum(axis=1)
    Q = np.diag(q)
    Ka = matrix_power(Q, -alpha) @ K @ matrix_power(Q, -alpha)

    # Compute row sums and define P
    Da = np.diag(Ka.sum(axis=1))
    P = solve(Da, Ka)

    # Compute (sorted) eigendecomposition and t
    Lam, R = eig(P)
    idx = Lam.argsort()[::-1]
    Lam = Lam[idx]
    R = R[:,idx]

    t = np.ceil(np.log(delta)/(np.log(abs(Lam[dim])) - np.log(abs(Lam[1]))))
    
    # Compute and return embedding
    if dim == 2:
        return np.array([Lam[1]**((1-alpha)*t) * R[:,1], 
                         Lam[2]**((1-alpha)*t) * R[:,2]]).T
    elif dim == 3:
        return np.array([Lam[1]**((1-alpha)*t) * R[:,1], 
                         Lam[2]**((1-alpha)*t) * R[:,2],
                         Lam[3]**((1-alpha)*t) * R[:,3]]).T
    else: raise RuntimeError(f'dim must be 2 or 3 (was {dim}')
