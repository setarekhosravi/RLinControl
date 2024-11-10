#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Created on Sun Nov 10 2024
    @author/student: STRH
    HW4 | Q5 | policy iteration test
"""

import numpy as np
from GridWorld import q_grid
from Policy_iteration import policy_iteration
from Iterative_policy_evaluation import iterative_policy_evaluation

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

policy = policy_iteration(g)

print_policy(policy, g)

V = iterative_policy_evaluation(policy, g)

# print_values(V, g)