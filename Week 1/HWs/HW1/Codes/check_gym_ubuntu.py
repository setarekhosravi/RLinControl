#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Course: Reinforcement Learning in Control 
Semester: 4031 | Fall 2024
Student: STRH 
Created on Thu, 26 Sept 2024, 20:45:00
    HW1: Gym Installation on Ubuntu
"""
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()
