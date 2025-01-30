#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Created on Wed Jan 23 20:04:07 2025
    @author: STRH/99411425
    RL in Control Comperehensive Project
    Q1 part C: SARSA on CartePole
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
def Sarsa(args):
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

    # convert the continues state space to discrete state space
    # the obsservation space has the shape of (4, )
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    # checking the process
    if(is_training):
        # creating qtable
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n)) # init a 11x11x11x11x2 array
    # if is not training weights will be load for testing
    else:
        f = open(args.qtable,'rb')
        q = pickle.load(f)
        f.close()

    # setting the parameters for training
    learning_rate_a = args.learning_rate # alpha or learning rate
    discount_factor_g = args.discount_factor # gamma or discount factor.
    episodes = args.episodes
    epsilon = args.epsilon         # 1 = 100% random actions
    epsilon_decay_rate = args.epsilon_decay # epsilon decay rate
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = []

    i = 0

    for i in range(episodes):
    # while(True):

        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False          # True when reached goal

        rewards=0

        while(not terminated and rewards < 10000):

            if is_training and rng.random() < epsilon:
                # Choose random action  (0=go left, 1=go right)
                # NOTE: the action space is discrete
                # if the random number generated is lower than 
                # epsilon we should choose random action
                action = env.action_space.sample()
            else:
                # if it is greater than epsilon the policy is greedy
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            # apply the action to environment and get state
            new_state,reward,terminated,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av= np.digitize(new_state[3], ang_vel_space)

            # now update the q function with SARSA (on-policy strategy)
            if is_training:
                # Choose next action based on current policy (ε-greedy)
                next_action = None
                if rng.random() < epsilon:
                    next_action = env.action_space.sample()  # Random action
                else:
                    next_action = np.argmax(q[new_state_p, new_state_v, new_state_a, new_state_av, :])  # Greedy action
                
                q[state_p, state_v, state_a, state_av, action] = q[state_p, state_v, state_a, state_av, action] + learning_rate_a * (
                    reward + discount_factor_g * q[new_state_p, new_state_v, new_state_a, new_state_av, next_action] - q[state_p, state_v, state_a, state_av, action]
                )

            # store new states into previous variables
            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av= new_state_av
            
            # calculate reward
            rewards+=reward
            
            # print reward per 100 episodes
            if not is_training and rewards%100==0:
                print(f'Episode: {i}  Rewards: {rewards}')

        # calculate mean reward
        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])

        if is_training and i%100==0:
            print(f'Episode: {i} {rewards}  Epsilon: {epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')

        if mean_rewards>1000:
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        i+=1

    env.close()

    # Save Q table to file
    if is_training:
        f = open(f'cartpole_sarsa_{episodes}.pkl','wb')
        pickle.dump(q, f)
        f.close()

    # plot rewards convergence graph
    mean_rewards = []
    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t-100):(t+1)]))
    plt.plot(mean_rewards)
    plt.savefig(f'cartpole.png')

if __name__ == '__main__':
    args = parse_args()
    t1 = time.time()
    Sarsa(args)
    t2 = time.time()

    processing_time = t2-t1
    print(f"Process Time: {processing_time}")