# -*- coding: utf-8 -*-
#import numpy as np

#from ..lu_decomp import LU_Decomp
#from ...kahan_floats.kahan import K_sum
from utils.utils import norm
from inverse import inverse

def cond(A, p_norm):
    """
    Outputs the condition number of the square matrix
    cond = ||A||*||inv(A)||
    """
    
    # get the inverse of A matrix
    Ainv = inverse(A)
    
    return norm(A,p_norm)*norm(Ainv, p_norm)
    
