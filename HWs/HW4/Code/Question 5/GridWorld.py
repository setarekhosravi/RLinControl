#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Created on Sun Nov 10 10:10:26 2024
    @author/student: STRH
    HW4 | Q5 | GridWorld environment 
"""

import numpy as np

ACTION_SPACE = ['U', 'D', 'L', 'R']

class GridWorld:
    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]
        self.start = start

    def set(self, rewards, actions, probs):
        # rewards is a dictionary of {(row,col): reward} 
        # just define for states that have reward
        self.rewards = rewards

        # actions is a dictionary of actions that agent 
        # can do in a special state
        # actions: {(row,col): [list of actions]}
        self.actions = actions

        # probs is a dictionary of {((row,col),action): {(row',col'):p}}
        self.probs = probs

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)
    
    def is_terminal(self, s):
        return s not in self.actions
    
    def move(self, a):
        s = (self.i, self.j)
        next_states_probs = self.probs[(s,a)]
        next_states = list(next_states_probs.keys())
        next_probs = list(next_states_probs.values())

        idx = np.random.choice(len(next_states), p=next_probs)

        s2 = next_states[idx]

        self.i, self.j = s2

        return self.rewards.get(s2, 0)
    
    def game_over(self):
        return (self.i, self.j) not in self.actions
    
    def all_states(self):
        return set(self.actions.keys() | self.rewards.keys())
    
    def reset(self):
        self.i, self.j = self.start
        return self.start
    

def q_grid():
    grid = GridWorld(4,4,(0,0))
    rewards = {
        (3,3): 10,
        (1,2): -10
    }
    actions = {
        (0, 0): ('D', 'R'), 
        (0, 1): ('D', 'L', 'R'), 
        (0, 2): ('D', 'L', 'R'),
        (0, 3): ('D', 'L'),
        (1, 0): ('U', 'D', 'R'), 
        (1, 1): ('U', 'D', 'R', 'L'),
        (1, 2): ('U', 'D', 'R', 'L'),
        (1, 3): ('U', 'D', 'L'), 
        (2, 0): ('U', 'D', 'R'), 
        (2, 1): ('U', 'D', 'R', 'L'), 
        (2, 2): ('U', 'D', 'R', 'L'), 
        (2, 3): ('U', 'D', 'L'),
        (3, 0): ('U', 'R'),
        (3, 1): ('U', 'L', 'R'),
        (3, 2): ('U', 'L', 'R'),
    }

    probs = {
        ((0, 0), 'U'): {(0, 0): 1.0},   # (0, 0) 
        ((0, 0), 'D'): {(1, 0): 1.0},
        ((0, 0), 'L'): {(0, 0): 1.0},
        ((0, 0), 'R'): {(0, 1): 1.0}, 
        ((1, 0), 'U'): {(0, 0): 1.0},   # (1, 0)
        ((1, 0), 'D'): {(2, 0): 1.0},
        ((1, 0), 'L'): {(1, 0): 1.0},
        ((1, 0), 'R'): {(1, 1): 1.0},
        ((2, 0), 'U'): {(1, 0): 1.0},   # (2, 0)
        ((2, 0), 'D'): {(3, 0): 1.0},
        ((2, 0), 'L'): {(2, 0): 1.0},
        ((2, 0), 'R'): {(2, 1): 1.0}, 
        ((3, 0), 'U'): {(2, 0): 1.0},   # (3, 0)
        ((3, 0), 'D'): {(3, 0): 1.0},
        ((3, 0), 'L'): {(3, 0): 1.0},
        ((3, 0), 'R'): {(3, 1): 1.0}, 
        ((0, 1), 'U'): {(0, 1): 1.0},   # (0, 1) 
        ((0, 1), 'D'): {(1, 1): 1.0},
        ((0, 1), 'L'): {(0, 0): 1.0},
        ((0, 1), 'R'): {(0, 2): 1.0},
        ((0, 2), 'U'): {(0, 2): 1.0},   # (0, 2) 
        ((0, 2), 'D'): {(1, 2): 1.0},
        ((0, 2), 'L'): {(0, 1): 1.0},
        ((0, 2), 'R'): {(0, 3): 1.0},
        ((0, 3), 'U'): {(0, 3): 1.0},   # (0, 3) 
        ((0, 3), 'D'): {(1, 3): 1.0},
        ((0, 3), 'L'): {(0, 2): 1.0},
        ((0, 3), 'R'): {(0, 3): 1.0},
        ((1, 1), 'U'): {(0, 1): 1.0},   # (1, 1)
        ((1, 1), 'D'): {(2, 1): 1.0},
        ((1, 1), 'L'): {(1, 0): 1.0},
        ((1, 1), 'R'): {(1, 2): 1.0},
        ((1, 2), 'U'): {(0, 2): 1.0},    # (1, 2) 
        ((1, 2), 'D'): {(2, 2): 1.0},
        ((1, 2), 'L'): {(1, 1): 1.0},
        ((1, 2), 'R'): {(1, 3): 1.0},
        ((1, 3), 'U'): {(0, 3): 1.0},    # (1, 3) 
        ((1, 3), 'D'): {(2, 3): 1.0},
        ((1, 3), 'L'): {(1, 2): 1.0},
        ((1, 3), 'R'): {(1, 3): 1.0},
        ((2, 1), 'U'): {(1, 1): 1.0},   # (2, 1) 
        ((2, 1), 'D'): {(3, 1): 1.0},
        ((2, 1), 'L'): {(2, 0): 1.0},
        ((2, 1), 'R'): {(2, 2): 1.0},
        ((2, 2), 'U'): {(1, 2): 1.0},   # (2, 2) 
        ((2, 2), 'D'): {(3, 2): 1.0},
        ((2, 2), 'L'): {(2, 1): 1.0},
        ((2, 2), 'R'): {(2, 3): 1.0},
        ((2, 3), 'U'): {(1, 3): 1.0},   # (2, 3) 
        ((2, 3), 'D'): {(3, 3): 1.0},   # This leads to terminal state
        ((2, 3), 'L'): {(2, 2): 1.0},
        ((2, 3), 'R'): {(2, 3): 1.0},
        ((3, 1), 'U'): {(2, 1): 1.0},   # (3, 1) 
        ((3, 1), 'D'): {(3, 1): 1.0},
        ((3, 1), 'L'): {(3, 0): 1.0},
        ((3, 1), 'R'): {(3, 2): 1.0},
        ((3, 2), 'U'): {(2, 2): 1.0},   # (3, 2) 
        ((3, 2), 'D'): {(3, 2): 1.0},
        ((3, 2), 'L'): {(3, 1): 1.0},
        ((3, 2), 'R'): {(3, 3): 1.0},   # This leads to terminal state
        # Note: (3,3) is terminal, so it doesn't need transition probabilities
    }

    grid.set(rewards, actions, probs)
    return grid