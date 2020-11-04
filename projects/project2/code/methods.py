import numpy as np
from numpy.linalg import norm, svd, pinv, solve
from sklearn.utils.extmath import randomized_svd

def kmeans(A, k, tol=1e-5, maxiter=100000):
    n, d = A.shape
    L = np.zeros((n, k))
    R = A[np.random.choice(n, size=(k,), replace=False), :]  # get k representatives
    
    for _ in range(maxiter):
        # Attribute every observation to a cluster
        for i in range(n):
            dists = norm(A[i, :] - R, axis=1)**2
            L[i, :] = np.zeros((k,))
            L[i, np.argmin(dists)] = 1
        
        # Recompute representatives
        Rnew = np.zeros((k, d))
        for i in range(k): Rnew[i, :] = A[L[:, i] == 1, :].mean(axis=0)
        
        if norm(R - Rnew, ord='fro') < tol:
            R = Rnew
            break
        
        R = Rnew
    
    return L, R

def pGD(A, k, alpha=0.01, maxiter=1000):
    n, d = A.shape
    W = np.random.normal(loc=2.5, scale=2.5, size=(n, k))
    H = np.random.normal(loc=2.5, scale=2.5, size=(k, d))
    R = A - W @ H
    norms = np.zeros((maxiter,))
    norms[0] = norm(R, ord='fro')

    for itr in range(1, maxiter):
        # Update W and H
        Wnew = np.maximum(W + alpha * R @ H.T, np.zeros((n, k)))
        Hnew = np.maximum(H + alpha * W.T @ R, np.zeros((k, d)))
        W = Wnew; H = Hnew

        # Calculate residuals and norm
        R = A - W @ H
        norms[itr] = norm(R, ord='fro')
    
    return W, H, norms

def lee_seung(A, k, maxiter=1000):
    n, d = A.shape
    W = np.random.uniform(low=0, high=5, size=(n, k))
    H = np.random.uniform(low=0, high=5, size=(k, d))
    norms = np.zeros((maxiter,))
    norms[0] = norm(A - W @ H, ord='fro')

    for itr in range(1, maxiter):
        # Update W and H
        Wnew = (W * (A @ H.T))/(W @ H @ H.T)
        Hnew = (H * (Wnew.T @ A))/(Wnew.T @ Wnew @ H)
        W = Wnew; H = Hnew
        
        # Calculate residual norm
        norms[itr] = norm(A - W @ H, ord='fro')
    
    return W, H, norms

def low_rank(A, k, lbda=0.01, maxiter=1000):
    n, d = A.shape
    Omega = ~np.isnan(A)
    X = np.random.uniform(low=0, high=5, size=(n, k))
    Y = np.random.uniform(low=0, high=5, size=(d, k))
    
    for _ in range(maxiter):
        for i in range(n):
            axi = A[i, Omega[i, :]]
            X[i, :] = pinv(Y[Omega[i, :], :] + lbda * np.eye(Omega[i, :].sum(), k)) @ axi
        
        for i in range(d):
            ayi = A[Omega[:, i], i]
            Y[i, :] = pinv(X[Omega[:, i], :] + lbda * np.eye(Omega[:, i].sum(), k)) @ ayi

    return X @ Y.T

def nuclear_norm(A, lbda=0.01, maxiter=1000):
    _, d = A.shape
    Omega = ~np.isnan(A)
    M = np.random.uniform(low=0, high=5, size=A.shape)
    
    def P_O(X):
        # TODO vectorize this
        out = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if Omega[i, j]: out[i, j] = X[i, j]
        return out

    for _ in range(maxiter):
        U, Sig, Vt = svd(M + P_O(A - M))
        Mnew = (U[:, :d] * (Sig - lbda)) @ Vt
        M = Mnew

    return M

def CUR(A, k, c, r=None):
    def column_select(A, k, c):
        # Performs the ColumnSelect algorithm on A with rank parameter k
        # original idea (basically the same as the in class version but
        # translated into Python)
        _, d = A.shape
        _, _, Vt = randomized_svd(A, n_components=k)  # get SVD
        V = Vt.T  # select top k right singular vectors

        leverage = np.array([norm(V[j,:])**2/k for j in range(d)])  # compute leverage scores
        pjs = np.minimum(np.ones((d,)), c*leverage)  # compute probabilities

        S = np.where(np.random.uniform(size=(d,)) < pjs, True, False)
        
        return A[:, S]
    
    if r is None: r = c

    C = column_select(A, k, c)
    R = column_select(A.T, k, r).T
    U = pinv(C) @ A @ pinv(R)

    return C, U, R
