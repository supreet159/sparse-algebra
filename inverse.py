# -*- coding: utf-8 -*-
import numpy as np

from lu_decomposition.lu_decomp import LU_Decomp
from kahan_floats.kahan import K_sum

def inverse(A):
    """
    Outputs inverse of a matrix A with (N,N) dimension
    
    Input: 
        - A: a numpy array of shape (N,N)
    Output: 
        - a numpy arrau of shape(N,N)
    """
    
    N = A.shape[0]
    
    P, L, U = LU_Decomp(A).decomp()
    
    P = P.T
    
    y = np.zeros_like(L) # Ly = P for y
    for i in range(N):
        for j in range(N):
            sum_obj = K_sum()
            for k in range(i):
                sum_obj.add(L[i,k]*y[k,j])
            s = sum_obj.current_sum()
            numerator = P[i,j] - s
            y[i,j] = numerator / L[i,i]
    
    x = np.zeros_like(U) # Ux = y for x
    for i in range(N-1, -1, -1):
        for j in range(N):
            sum_obj = K_sum()
            for k in range(N-1, i, -1):
                sum_obj.add(U[i,k]*x[k,j])
            s = sum_obj.current_sum()
            numerator = y[i,j] - s
            x[i,j] = numerator / U[i,i]
    
    return x