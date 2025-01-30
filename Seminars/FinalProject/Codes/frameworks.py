import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class KalmanFilter:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.theta = np.zeros((feature_dim, 1))  # Weight vector as column
        self.P = np.eye(feature_dim)  # Covariance matrix

    def update(self, h, r, R):
        """Kalman filter update for MAK-TD"""
        h = h.reshape(-1, 1)  # Ensure h is a column vector
        # Compute Kalman Gain
        K = self.P @ h / (h.T @ self.P @ h + R)
        # Update weights
        self.theta += K * (r - h.T @ self.theta)
        # Update covariance
        self.P = (np.eye(self.feature_dim) - K @ h.T) @ self.P
        return self.theta

class MAKTD:
    def __init__(self, state_dim, action_dim, gamma=0.99, R=0.1):
        self.feature_dim = state_dim + action_dim
        self.kf = KalmanFilter(self.feature_dim)
        self.gamma = gamma
        self.R = R

    def step(self, state, action, reward, next_state, phi):
        """One step update for MAK-TD"""
        h = phi(state, action, self.feature_dim)
        q_values = [self.kf.theta.T @ phi(next_state, a, self.feature_dim) for a in range(self.feature_dim - state_dim)]
        next_action = np.argmax(q_values)  # Select the action with the highest Q-value
        h_next = phi(next_state, next_action, self.feature_dim)
        h_td = h - self.gamma * h_next
        self.kf.update(h_td, reward, self.R)
        return self.kf.theta

class MAKSR:
    def __init__(self, state_dim, action_dim, gamma=0.99, alpha=0.1):
        self.gamma = gamma
        self.alpha = alpha
        self.M = np.zeros((state_dim, state_dim, action_dim))  # Successor representation

    def update_sr(self, state, next_state, action):
        """Update successor representation"""
        self.M[state, :, action] += self.alpha * (
            (np.eye(self.M.shape[1])[state]) + self.gamma * self.M[next_state, :, action] - self.M[state, :, action]
        )
        return self.M

    def compute_q(self, reward, state, action):
        """Compute Q-value using SR"""
        return np.sum(self.M[state, :, action] * reward)

def phi(state, action, feature_dim):
    """Feature extraction function"""
    features = np.zeros((feature_dim,))
    features[state] = 1  # One-hot encoding for state
    features[state + action] = 1  # One-hot encoding for action
    return features

# DQN implementation
class DQN:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Dense(24, input_dim=self.state_dim, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_dim, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= 0.1:  # Exploration rate
            return random.randrange(self.action_dim)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

# Simulation and comparison
def simulate_and_compare(state_dim, action_dim, episodes=100, steps=20):
    gamma = 0.99
    R = 0.1
    alpha = 0.1

    # Initialize MAK-TD, MAK-SR, and DQN
    maktd = MAKTD(state_dim, action_dim, gamma, R)
    maksr = MAKSR(state_dim, action_dim, gamma, alpha)
    dqn = DQN(state_dim, action_dim, gamma)

    feature_dim = state_dim + action_dim
    
    maktd_rewards = []
    maksr_rewards = []
    dqn_rewards = []

    total_maktd_reward = 0
    total_maksr_reward = 0
    total_dqn_reward = 0

    for episode in range(episodes):
        state = np.random.randint(0, state_dim)
        total_reward_td = 0
        total_reward_sr = 0
        total_reward_dqn = 0

        state_dqn = np.reshape(np.identity(state_dim)[state], (1, state_dim))

        for t in range(steps):
            # MAK-TD
            td_action = np.random.randint(0, action_dim)  # Replace with MAK-TD policy if available
            td_next_state = np.random.randint(0, state_dim)
            td_reward = np.random.random()  # Replace with proper reward model if available
            maktd.step(state, td_action, td_reward, td_next_state, phi)
            total_reward_td += td_reward

            # MAK-SR
            sr_action = np.random.randint(0, action_dim)  # Replace with MAK-SR policy if available
            sr_next_state = np.random.randint(0, state_dim)
            sr_reward = np.random.random()  # Replace with proper reward model if available
            maksr.update_sr(state, sr_next_state, sr_action)
            total_reward_sr += sr_reward

            # DQN
            dqn_action = dqn.act(state_dqn)  # Use DQN policy
            dqn_next_state = np.random.randint(0, state_dim)
            dqn_reward = np.random.random()  # Replace with proper reward model if available
            dqn.remember(state_dqn, dqn_action, dqn_reward, np.reshape(np.identity(state_dim)[dqn_next_state], (1, state_dim)), False)
            total_reward_dqn += dqn_reward

            # Update states
            state = dqn_next_state
            state_dqn = np.reshape(np.identity(state_dim)[state], (1, state_dim))

        # Replay for DQN
        dqn.replay()

        maktd_rewards.append(total_reward_td)
        maksr_rewards.append(total_reward_sr)
        dqn_rewards.append(total_reward_dqn)

        total_maktd_reward += total_reward_td
        total_maksr_reward += total_reward_sr
        total_dqn_reward += total_reward_dqn

        print(f"Episode {episode + 1}: MAK-TD Reward = {total_reward_td}, MAK-SR Reward = {total_reward_sr}, DQN Reward = {total_reward_dqn}")

    print("\nFinal Total Rewards after 100 Episodes:")
    print(f"MAK-TD Total Reward: {total_maktd_reward}")
    print(f"MAK-SR Total Reward: {total_maksr_reward}")
    print(f"DQN Total Reward: {total_dqn_reward}")

    # Plot results
    plt.plot(range(episodes), maktd_rewards, label="MAK-TD Total Reward", marker='o')
    plt.plot(range(episodes), maksr_rewards, label="MAK-SR Total Reward", marker='x')
    plt.plot(range(episodes), dqn_rewards, label="DQN Total Reward", marker='s')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Comparison of Total Rewards for MAK-TD, MAK-SR, and DQN")
    plt.legend()
    plt.show()

# Parameters
state_dim = 10
action_dim = 3
simulate_and_compare(state_dim, action_dim)