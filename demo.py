# -*- coding: utf-8 -*-
"""
Created on Wed Sept 5 11:17:51 2018

@author: kssupreet
"""
import numpy as np
from inverse import inverse
from condition import cond


def main():
    A = np.array([
                  [ 1, 2, 3, 4],
                  [ 5, 6, 7, 8],
                  [ 9,10,11,12],
                  [13,14,15,16]
                  ])
    
    Ainv = inverse(A)
    
    print('.............................................')
    print('........Testing with a small matrix..........')
    print('.............................................')
    
    print(A)
    print('........A inverse is.............')
    print(Ainv)
    
    print('.......condition number of A with 2-norm is........')
    print(cond(A,2))
    print('.......condition number of A with 1-norm is.........')
    print(cond(A,1))
    
    print('.............................................')
    print('........Testing with a large matrix..........')
    print('.............................................')
    
    A = np.random.randn(50,50)
    Ainv = inverse(A)
    print(A)
    print('........A inverse is.............')
    print(Ainv)
    
    print('.......condition number of A with 2-norm is........')
    print(cond(A,2))
    print('.......condition number of A with 1-norm is.........')
    print(cond(A,1))

if __name__ == '__main__':
    main()