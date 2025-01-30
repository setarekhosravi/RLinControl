#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Created on Wed Jan 23 20:33:29 2025
    @author: STRH/99411425
    RL in Control Comprehensive Project
    Q1 part A: On-Policy Monte Carlo (GLIE Monte Carlo Control) on CartPole
'''

# import libraries
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import time

def parse_args():
    parser = argparse.ArgumentParser(description="CartePole Training")
    parser.add_argument('--train', type=str, required=True, help="Train or test")
    parser.add_argument('--render', type=str, required=True, help="yes or no")
    parser.add_argument('--episodes', type=int, required=True, default=15000, help="Number of episodes")
    parser.add_argument('--qtable', type=str, required=False, help="If you want to test, parse the qtable path")
    parser.add_argument('--learning_rate', type=float, required=False, default=0.01, help="Learning rate for training process")
    parser.add_argument('--discount_factor', type=float, required=False, default=0.8, help="Discount factor for training process")
    parser.add_argument('--epsilon', type=float, required=False, default=0.7, help="Epsilon factor")
    parser.add_argument('--epsilon_decay', type=float, required=False, default=0.00001, help="Epsilon decay factor")
    return parser.parse_args()

# def main function
def OnMonteCarlo(args):
    if args.train.lower() == 'train':
        is_training = True
        print("Training Started...")
    else:
        is_training = False
        print("Testing Started...")

    if args.render.lower() == 'yes':
        render = True
    else:
        render = False
    # creating the environment
    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    # convert the continuous state space to discrete state space
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    # Initialize Q-table
    if is_training:
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n))
    else:
        with open(args.qtable, 'rb') as f:
            q = pickle.load(f)

    # Setting the parameters for training
    learning_rate_a = args.learning_rate  # alpha or learning rate
    discount_factor_g = args.discount_factor  # gamma or discount factor
    episodes = args.episodes
    epsilon = args.epsilon
    epsilon_decay_rate = args.epsilon_decay
    rng = np.random.default_rng()  # random number generator

    rewards_per_episode = []

    for i in range(episodes):
        state = env.reset()[0]  # Initial state
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        trajectory = []  # Store (state, action, reward) tuples
        terminated = False
        rewards = 0

        # Generate an episode
        while not terminated and rewards < 10000:
            if is_training and rng.random() < epsilon:
                # Choose random action (exploration)
                action = env.action_space.sample()
            else:
                # Choose greedy action (exploitation)
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            # Apply the action and observe the result
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            # Append the experience to the trajectory
            trajectory.append((state_p, state_v, state_a, state_av, action, reward))

            # Update state variables
            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av

            rewards += reward

        # Monte Carlo updates after the episode
        if is_training:
            G = 0  # Initialize return
            for state_p, state_v, state_a, state_av, action, reward in reversed(trajectory):
                G = reward + discount_factor_g * G  # Calculate return
                # Update Q-value using incremental mean
                q[state_p, state_v, state_a, state_av, action] = q[state_p, state_v, state_a, state_av, action] + learning_rate_a * (
                    G - q[state_p, state_v, state_a, state_av, action]
                )

        # Record rewards and print progress
        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[-100:])

        if not is_training and rewards % 100 == 0:
            print(f'Episode: {i}  Rewards: {rewards}')

        if is_training and i % 100 == 0:
            print(f'Episode: {i} {rewards}  Epsilon: {epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')

        if mean_rewards > 1000:
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)

    env.close()

    # Save Q table to file
    if is_training:
        with open(f'cartpole_OnMC_{episodes}.pkl', 'wb') as f:
            pickle.dump(q, f)

    # plot rewards convergence graph
    mean_rewards = []
    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t-100):(t+1)]))
    plt.plot(mean_rewards)
    plt.savefig(f'cartpole_On-Policy_MonteCarlo.png')

if __name__ == '__main__':
    args = parse_args()
    t1 = time.time()
    OnMonteCarlo(args)
    t2 = time.time()

    processing_time = t2 - t1
    print(f"Training Time: {processing_time}")
