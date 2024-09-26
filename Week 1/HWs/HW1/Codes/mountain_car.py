#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Course: Reinforcement Learning in Control 
Semester: 4031 | Fall 2024
Student: STRH 
Created on Thu, 26 Sept 2024, 20:49:00
    HW1: Test Mountain Car Environment
"""
import gymnasium as gym
from alive_progress import alive_bar

env = gym.make("MountainCar-v0", render_mode="human")
observation, info = env.reset(seed=42)
with alive_bar(10000) as bar:
    for _ in range(10000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        bar()

    env.close()
