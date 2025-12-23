# Contents of /NeuroPause/NeuroPause/Personalization/rl_policy.py

"""
This file implements the reinforcement learning policy for the nudge system.

Expected input: user interaction data.
Expected output: policy decisions for nudges.
"""

import numpy as np

class RLPolicy:
    def __init__(self, state_size, action_size):
        """
        Initialize the reinforcement learning policy.

        Parameters:
        state_size (int): The size of the state space.
        action_size (int): The size of the action space.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))  # Initialize Q-table

    def choose_action(self, state, epsilon):
        """
        Choose an action based on the epsilon-greedy policy.

        Parameters:
        state (int): The current state.
        epsilon (float): The probability of choosing a random action.

        Returns:
        int: The chosen action.
        """
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_policy(self, state, action, reward, next_state, alpha, gamma):
        """
        Update the Q-table based on the action taken and the reward received.

        Parameters:
        state (int): The current state.
        action (int): The action taken.
        reward (float): The reward received.
        next_state (int): The next state after taking the action.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        """
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + gamma * self.q_table[next_state][best_next_action]
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += alpha * td_delta  # Update Q-value

    def save_policy(self, filepath):
        """
        Save the Q-table to a file.

        Parameters:
        filepath (str): The path to save the Q-table.
        """
        np.save(filepath, self.q_table)

    def load_policy(self, filepath):
        """
        Load the Q-table from a file.

        Parameters:
        filepath (str): The path to load the Q-table from.
        """
        self.q_table = np.load(filepath)