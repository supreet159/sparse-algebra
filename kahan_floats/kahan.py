# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt

class K_sum(object):
    """
    Reduces error when summing floating point numbers 
    by storing a running compensation term to hold the
    lost low-order bits.
    
    ref: CS-205 numerical analysis class material and 
    lots of youtube videos
    """

    def __init__(self):
        """
        Initializes the sum and compensation variables
        """
        self.sum = 0
        self.comp = 0
    
    def add(self, x):
        """
        adds a floating point number x to the sum
        """
        # first add compensation term to the number
        x += self.comp
        
        # keep adding it to the sum
        sum = self.sum  + x
        
        # compensation and sum are updated here
        self.comp = x - (sum - self.sum)
        self.sum = sum
        
        #return the sum
        return self.sum
    
    def current_sum(self):
        return self.sum

    def kn_sum(x, axis = None):
        """
        computes the summation of a list of floating point numbers x
        
        axis : 
            None : sums all the floatint point numbers in x
            0: sums all the numbers in all the columns
            1: sums all the numbers in all the rows
        """
        
        x = np.asarray(x)
        
        #for 1D array
        if x.ndim == 1:
            N = len(x)
            
            summ = K_sum()
            
            for i in range(N):
                summ.add(x[i])
            
            return summ.current_sum()
        
        if x.ndim == 2:
            N_rows, N_cols = x.shape
            
            if axis == None:
                
                summ = K_sum()
                
                for i in range(N_rows):
                    for j in range(N_cols):
                        summ.add(x[i,j])
                
                return summ.current_sum()
            
            elif axis == 0:
                sum_temp = []
                
                for i in range(N_cols):
                    summ = K_sum()
                    
                    for j in range(N_rows):
                        summ.add(x[j,i])
                    
                    sum_temp.append(summ.current_sum())
                    
                summ = np.asarray(sum_temp)
                
                return summ
            
            elif axis == 1:
                sum_temp = []
                
                for i in range(N_rows):
                    summ = K_sum()
                    
                    for j in range(N_cols):
                        summ.add(x[i,j])
                    
                    sum_temp.append(summ.current_sum())
                
                summ = np.asarray(sum_temp)
                
                return summ