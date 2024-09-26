"""
Course: Reinforcement Learning in Control 
Semester: 4031 | Fall 2024
Student: STRH 
Created on Thu, 26 Sept 2024, 19:01:00
    HW1: Gym Installation on windows
"""
import gymnasium as gym

env = gym.make("Humanoid-v4", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()
