#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Created on Sun Nov 10 2024
    @author/student: STRH
    HW4 | Q5 | value iteration test
"""

import numpy as np
from GridWorld import q_grid
from Value_iteration import value_iteration

def print_values(V, g): 
    for i in range(g.rows): 
        print("------------------"*3)
        for j in range(g.cols): 
            s = (i,j)
            v = V.get(s,0)
            print("\t%.2f\t|"%v, end = "")
        print("")
            
def print_policy(P, g): 
    for i in range(g.rows): 
        print("-----------------"*3)
        for j in range(g.cols): 
            a = P.get((i,j), ' ')
            print(" %s |"%a, end = "")
        print("")

g = q_grid() 

V, policy = value_iteration(g)
print_policy(policy, g)
print_values(V, g)
