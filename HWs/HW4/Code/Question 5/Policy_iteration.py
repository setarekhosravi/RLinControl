#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Created on Sun Nov 10 11:11:23 2024
    @author/student: STRH
    HW4 | Q5 | Policy Iteraion
"""

import numpy as np
from GridWorld import q_grid, ACTION_SPACE

def random_policy(grid):
    # this function create a random policy
    policy = {}
    for s in grid.actions.keys(): 
        random_action = np.random.choice(ACTION_SPACE)
        policy[s] = {random_action: 1.0}
    return policy


def calculate_probs_and_rewards(grid): 
    # gathering information form the grid
    transition_probs = {}
    rewards = {}
    for (s,a), v in grid.probs.items(): 
        for s2, p in v.items(): 
            transition_probs[(s,a,s2)] = p
            rewards[(s,a,s2)] = grid.rewards.get(s2,0)
    return transition_probs, rewards

def iterative_policy_evaluation(policy, grid, threshold = 1.0e-3, gamma = 0.9): 
    # function for policy evaluation
    transition_probs, rewards = calculate_probs_and_rewards(grid)

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


def policy_iteration(grid, gamma = 0.9, threshold = 1.0e-3): 
    transition_probs, rewards = calculate_probs_and_rewards(grid)
    policy = random_policy(grid)

    while True: 
        # policy evaluation
        V = iterative_policy_evaluation(policy, grid)

        # policy improvement
        policy_stable = True
        for s in grid.actions.keys(): 
            old_action = list(policy[s].keys())[0]
            best_action = None 
            best_value = float("-inf")

            for a in ACTION_SPACE: 
                action_value = 0 
                for s2 in grid.all_states(): 
                    r = rewards.get((s,a,s2), 0)
                    t = transition_probs.get((s,a,s2), 0)

                    action_value += t*(r + gamma*V[s2])
                
                if action_value > best_value: 
                    best_action = a
                    best_value = action_value
            
            # update the policy
            policy[s] = {best_action: 1.0}
            if old_action != best_action: 
                policy_stable = False

        if policy_stable: 
            break
    return policy

