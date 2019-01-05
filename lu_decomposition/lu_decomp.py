# -*- coding: utf-8 -*-
import numpy as np

from kahan_floats.kahan import K_sum
from utils.utils import unit_diagonal, l_diagonal, u_diagonal, permute

class LU_Decomp(object):
    """
    Computes LU decompostion of a matrix using partial pivoting.
    And solves Ax = b using three steps:
        Step 1: Decompose A matrix into L and U
        Step 2: Forward substitution (for y in Ly = Pb)
        Step 3: Backward substitution (for x in Ux = y)
    
    Input: 
        - A: a numpy array of size (N,N)
        - b: a column vector of shape (N,)
    Output: 
        - L: a lower traingular matrix of size (N,N)
        - U: a upper triangular matrix of size (N,N)
        - P: a permutation matrix if shape (N,N)
    
    Ref:
        - https://www.youtube.com/watch?v=smsE7iOlNj4
        - linalg library
    
    """
    
    def __init__(self, A):
        self._A = np.array(A)
        
    def decomp(self):
        
        N = len(self._A)
        self.A = np.array(self._A)
        self.P = np.eye(N)
        self.num_shuf = 0
        
        for i in range(N-1):
            
            flag = True
            if self.A[i,i] == 1:
                for j in range(i+1, N):
                    if self.A[j,i] != 0:
                        flag = False
            else:
                flag = False
                
            if flag:
                continue
            
            pivot_val = np.abs(self.A[i, i])
            pivot_row = i
            pivot_col = i

            # last row does not need partial pivoting
            if i != N - 1:

                # look underneath and find bigger pivot if it exists
                for j in range(i+1, N):
                    if np.abs(self.A[j, pivot_col]) > pivot_val:
                        pivot_val = np.abs(self.A[j, pivot_col])
                        pivot_row = j

                # switch current row with row containing max
                if pivot_row != i:
                    P = permute(N, [(i, pivot_row)])
                    self.A = np.dot(P, self.A)
                    self.P = np.dot(P, self.P)
                    self.num_shuf += 1

            self.pivot = self.A[i, pivot_col]
            
            for k in range(i+1, N):
                scaling = (self.A[k,i]/ self.pivot)
                self.A[k,i] = scaling
                for j in range(i+1,N):
                    self.A[k,j] -= scaling*self.A[i,j]
        
        self.P = self.P
        self.L = unit_diagonal(l_diagonal(self.A))
        self.U = u_diagonal(self.A, diag = True)
        
        return (self.P.T, self.L, self.U)
    
    def solve(self, b):
        """
        solve Ax = b using forward and backward substitution.
        """
        self.b = b
        self.decomp()
        self._forward_sub()
        self._backward_sub()
        return self.x
        
    def _forward_sub(self):
        """
        since we use partial pivoting, we will be solving: Ly = Pb
        """
        
        if self.b.ndim > 1:
            iters = self.b.shape[1]
        else: 
            iters = 1
        N = self.b.shape[0]
        
        self.y = np.zeros([N, iters])
        
        Pb = np.dot(self.P, self.b)
        
        for k in range(iters):
            for i in range(N):
                sum_obj = K_sum()
                for j in range(i):
                    sum_obj.add(self.L[i,j]*self.y[j,k])
                if self.b.ndim > 1:
                    self.y[i,k] = Pb[i,k] - sum_obj.current_sum()
                else:
                    self.y[i] = Pb[i] - sum_obj.current_sum()
        
    def _backward_sub(self):
        """
        we solve for Ux = y for x
        """
        if self.b.ndim > 1:
            iters = self.b.shape[1]
        else:
            iters = 1
        N = self.b.shape[0]
        
        
        self.x = np.zeros([N, iters])
        
        for k in range(iters):
            for i in range(N-1, -1, -1):
                sum_obj = K_sum()
                for j in range(N-1, i, -1):
                    sum_obj.add(self.U[i,j]*self.x[j,k])
                numerator = (self.y[i,k] - sum_obj.current_sum())
                self.x[i,k] = numerator / (self.U[i,i])
        
        if self.b.ndim == 1:
            self.x = self.x.squeeze()
        