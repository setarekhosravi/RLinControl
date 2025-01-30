#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Created on Wed Jan 23 20:55:29 2025
    @author: STRH/99411425
    RL in Control Comprehensive Project
    Q1: Choose algorithm via  commandd line
'''

# import libraries
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import time

from CartePoleOnMonteCarlo import OnMonteCarlo
from CartePoleQlearning import Qlearning
from CartePoleSARSA import Sarsa

def parse_args():
    parser = argparse.ArgumentParser(description="CartePole Training")
    parser.add_argument('--method', type=str, required=True, help="qlearning, sarsa or montecarlo")
    parser.add_argument('--train', type=str, required=True, help="Train or test")
    parser.add_argument('--render', type=str, required=True, help="yes or no")
    parser.add_argument('--episodes', type=int, required=True, default=15000, help="Number of episodes")
    parser.add_argument('--qtable', type=str, required=False, help="If you want to test, parse the qtable path")
    parser.add_argument('--learning_rate', type=float, required=False, default=0.01, help="Learning rate for training process")
    parser.add_argument('--discount_factor', type=float, required=False, default=0.8, help="Discount factor for training process")
    parser.add_argument('--epsilon', type=float, required=False, default=0.7, help="Epsilon factor")
    parser.add_argument('--epsilon_decay', type=float, required=False, default=0.00001, help="Epsilon decay factor")
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    if args.method.lower() == "qlearning":
        Qlearning(args)
    elif args.method.lower() == "sarsa":
        Sarsa(args)
    elif args.method.lower() == "montecarlo":
        OnMonteCarlo(args)
    else: 
        raise ValueError("Invalid algorithm, choose from qlearning, sarsa or montecarlo")