# -*- coding: utf-8 -*-
import numpy as np

from kahan_floats.kahan import K_sum
#from lu_decomposition.lu_decomp import LU_Decomp

def unit_diagonal(A):
    """
    Updates the diagonal values of A matrix with 1s
    """
    m = len(A)

    for i in range(m):
        A[i, i] = 1

    return A

def basis_arr(ks, n):
    """
    Creates an array of k'th standard basis vectors in R^n
    according to each k in ks.
    """

    b = np.zeros([n, n])
    for i, k in enumerate(ks):
        b[i, k] = 1
    return b

def norm(x, p):
    """
    Returns the p norm of a vector.
    """
    v = np.array(x).flatten()

    N = v.shape[0]

    summer = K_sum()
    for i in range(N):
        summer.add(np.power(np.abs(v[i]), p))

    return np.power(summer.current_sum(), 1./p)

def l_diagonal(A, diag = False):
    """
    Outputs the lower-diagonal elements of the
    square matrix A.
    """
    m = len(A)
    L = np.zeros_like(A)

    for i in range(m):
        u_b = i
        if diag:
            u_b = i + 1
        for j in range(0, u_b):
            L[i, j] = A[i, j]

    return L
   
def u_diagonal(A, diag=False):
    """
    Outputs the upper-diagonal elements of the
    square matrix A.
    """
    m = len(A)
    U = np.zeros_like(A)

    for i in range(m):
        l_b = i + 1
        if diag:
            l_b = i
        for j in range(l_b, m):
            U[i, j] = A[i, j]

    return U

def diag(A):
    """
    Outputs the diagonal elements of the square
    matrix A.
    """
    N = len(A)
    D = np.zeros([N, 1])

    for i in range(N):
        D[i] = A[i, i]

    return D


def create_diag(x):
    """
    Create a square matrix whose diagonal
    elements are the elements of x.
    """
    N = x.shape[0]
    D = np.zeros([N, N])

    for i in range(N):
        D[i, i] = x[i]

    return D


def permute(N, idx):
    """
    Permutes the rows of a square matrix A of shape (N, N)
    according to a list of indices stored in idx.
    Input
    ----
    - N: number of rows/columns of square matrix A.
    - idx: a list of integers or integer pairs where
      max(idx) < (N - 1).
    - todo: checking if idx is more than the size of A and displaying 
      error messages. 
    Output
    -------
    - P: permutation array of size (N, N).
    """

    # check if list is nested
    nested = any(isinstance(i, (tuple, list)) for i in idx)

    # convert to standard form
    if nested:

        before = [i[0] for i in idx]
        after = [i[1] for i in idx]

        idx = list(np.arange(N))
        for i in range(len(before)):
            idx[before[i]], idx[after[i]] = idx[after[i]], idx[before[i]]

    # construct permutation matrix
    P = basis_arr(idx, N)

    return P

