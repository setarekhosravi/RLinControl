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
import matplotlib.pyplot as plt

class ENV:
    def __init__(self, env_name, render_mode, episodes, seed, limit_range, save_path):
        self.env_name = env_name
        self.render_mode = render_mode
        self.episode_number = episodes
        self.seed = seed
        self.limit_range = limit_range
        self.output = save_path
        
    def env_creator(self):
        self.env = gym.make(self.env_name, render_mode=self.render_mode)
        self.initial_observation, info = self.env.reset(seed=self.seed)

    def active_plot_creator(self):
        plt.ion()
        fig, self.ax = plt.subplots()
        self.x_data = []
        self.y_data = []  # Data for plotting
        self.line, = self.ax.plot(self.x_data, self.y_data, 'r-')  # Plot an empty line (will be updated)

        # Plot configuration
        self.ax.set_xlim(0, self.episode_number)
        self.ax.set_ylim(self.limit_range[0], self.limit_range[1])  # Ex: For MountainCar position range is (-1.2 to 0.6)

    def run(self):
        with alive_bar(self.episode_number) as bar:
            for step in range(self.episode_number):
                # Sample random action from the environment
                action = self.env.action_space.sample()
                observation, reward, terminated, truncated, info = self.env.step(action)

                # Update the plot every 100 steps
                if step % 10 == 0:
                    self.x_data.append(step)
                    self.y_data.append(observation[0])  # observation space: Box([-1.2 -0.07], [0.6 0.07], (2,), float32)
                                                        # the zero param is position of the car along the x-axis
                    self.line.set_xdata(self.x_data)
                    self.line.set_ydata(self.y_data)
                    self.ax.relim()
                    self.ax.autoscale_view()
                    plt.draw()
                    plt.pause(0.001)

                if terminated or truncated:
                    observation, info = self.env.reset()

                bar()  # Update the progress bar

            self.env.close()
            plt.ioff()  # Turn off interactive mode
            self.ax.figure.savefig(self.output)  # Display the final plot


if __name__ == "__main__":
    mountain_car = ENV("MountainCar-v0", "human", 100, 42, [-1.2, 0.6], "/home/setare/Reinforcement Learning/Week 1/HWs/HW1/Images/output/output.png")
    mountain_car.env_creator()
    mountain_car.active_plot_creator()
    mountain_car.run()