#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Created on Sun Nov 10 2024
    @author/student: STRH
    HW4 | Q5 | policy evaluation
"""

import numpy as np 
from GridWorld import ACTION_SPACE

def iterative_policy_evaluation(policy, grid, threshold = 1.0e-3, gamma = 0.9): 
    # function for policy evaluation
    # gathering information form the grid
    transition_probs = {}
    rewards = {}
    for (s,a), v in grid.probs.items(): 
        for s2, p in v.items(): 
            transition_probs[(s,a,s2)] = p
            rewards[(s,a,s2)] = grid.rewards.get(s2,0)

    # Initialization
    V = {} 
    for s in grid.all_states(): 
        V[s] = 0
    
    # repeat
    it = 0 
    while True: 
        delta = 0 
        for s in grid.all_states(): 
            if not grid.is_terminal(s): 
                old_v = V[s]
                new_v = 0 
                for a in ACTION_SPACE: 
                    for s2 in grid.all_states(): 
                        t = transition_probs.get((s,a,s2), 0)
                        pi = policy[s].get(a, 0)
                        r = rewards.get((s,a,s2), 0)
                        new_v += pi*t*(r+gamma*V[s2])
                V[s] = new_v
                delta = max(delta, np.abs(old_v - V[s]))
        it += 1
        if delta < threshold: 
            return V