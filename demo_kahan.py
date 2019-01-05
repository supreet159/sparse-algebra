# -*- coding: utf-8 -*-
"""
Created on Wed Sept 5 11:17:51 2018

@author: kssupreet
"""

import numpy as np

from kahan_floats.kahan import K_sum

def analyse(x):
    normal_sum = 0.
    for i in range(len(x)):
        normal_sum += x[i]

    np_sum = x.sum()
    kahann_sum = K_sum.kn_sum(x)
    print("Numpy: {}".format(np_sum))
    print("Normal Code: {}".format(normal_sum))
    print("Kahan: {}".format(kahann_sum))
    print("Diff (Normal-Kahan): {}".format(np.abs(normal_sum-kahann_sum)))
    print("Diff (Numpy-Kahan): {}".format(np.abs(np_sum-kahann_sum)))
    
def main():
    print("............ for a small vector.....................")
    x = np.random.uniform(0, 1, int(1e2))
    analyse(x)
    print("............ for a large vector.....................")
    x = np.random.uniform(0, 1, int(1e8))
    analyse(x)
    
if __name__ == '__main__':
    main()