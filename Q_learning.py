import sys
import time
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from campus_gui import *

BOLD = "\033[1m"  # ANSI escape sequence for bold text
RESET = "\033[0m"  # ANSI escape sequence to reset text formatting

train_flag = "train" in sys.argv
gui_flag = "gui" in sys.argv

env = game

def hash_obs(obs):
    key = (
        obs['position'][0],
        obs['position'][1],
        obs['weather'],
        obs['layer'],
        obs['can_toggle_layer'],
        obs['nearest_crowd'][0],
        obs['nearest_crowd'][1],
    )

    return key

def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
    Q_table = {}
    num_updates = defaultdict(lambda: np.zeros(env.action_space.n))
    rewards = []
    
    for episode in tqdm(range(num_episodes)):
        total_reward = 0
        obs, reward, done, info = env.reset()
        
        total_reward += reward
        
        while not done:
            state = hash_obs(obs)
            if state not in Q_table.keys():
                Q_table[state] = np.zeros(env.action_space.n)

            current_prob = random.random()
            if current_prob > epsilon:
                action = np.argmax(Q_table[state])
            else:
                action = env.action_space.sample()
            
            eta = 1 / (1 + num_updates[state][action])
        
            obs, reward, done, info = env.step(action)
            total_reward +=reward
            state_next = hash_obs(obs)
            if state_next not in Q_table:
                Q_table[state_next] = np.zeros(env.action_space.n)
                
            # Values for Q_table update
            Q_old = Q_table[state][action]
            V_old = np.max(Q_table[state_next])
            
            # Update Q value for current (state, action, reward, next state) as well as number of updates
            Q_table[state][action] = (1 - eta) * Q_old + eta * (reward + (gamma * V_old))
            num_updates[state][action] += 1
        # decay epsilon after each episode    
        epsilon *= decay_rate
        rewards.append((episode, total_reward))
        
    x, y = zip(*rewards)
    avg_rewards = np.cumsum(y) / np.arange(1, len(y) + 1) 

    plt.figure(figsize=(10, 6), dpi=300)

    plt.plot(x, y, marker='o')
    plt.plot(x, avg_rewards, color='red', label='Average reward')

    plt.title("Episode Total Rewards", fontsize=16)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Total Reward", fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"training_plot_{num_episodes}episodes_{decay_rate}_decay.png", dpi=300,)
    
    return Q_table
            
"""
Specify number of episodes and decay rate for training and evaluation.
"""
num_episodes = 100000
decay_rate = 0.99999

"""
Run training if train_flag is set; otherwise, run evaluation using saved Q-table.
"""
if train_flag:
    Q_table = Q_learning(
        num_episodes=num_episodes, gamma=0.9, epsilon=1, decay_rate=decay_rate
    )  # Run Q-learning

    # Save the Q-table dict to a file
    with open(
        "Q_table_" + str(num_episodes) + "_" + str(decay_rate) + ".pickle", "wb"
    ) as handle:
        pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)


"""
Evaluation mode: play episodes using the saved Q-table. Useful for debugging/visualization.
Based on autograder logic used to execute actions using uploaded Q-tables.
"""


def softmax(x, temp=1.0):
    e_x = np.exp((x - np.max(x)) / temp)
    return e_x / e_x.sum(axis=0)


if not train_flag:
    rewards = []
    len_episodes = []

    filename = "Q_table_" + str(num_episodes) + "_" + str(decay_rate) + ".pickle"
    input(
        f"\n{BOLD}Currently loading Q-table from "
        + filename
        + f"{RESET}.  \n\nPress Enter to confirm, or Ctrl+C to cancel and load a different Q-table file.\n(set num_episodes and decay_rate in Q_learning.py)."
    )
    Q_table = np.load(filename, allow_pickle=True)
    start_time_total = time.perf_counter()
    new_states_reached = []
    actions_from_q_table = 0
    actions_from_random = 0
    for episode in tqdm(range(10000)):
        obs, reward, done, info = env.reset()
        total_reward = 0
        episode_len = 0

        while not done:
            state = hash_obs(obs)
            if state not in Q_table and state not in new_states_reached:
                new_states_reached.append(state)
            try:
                action = np.random.choice(
                    env.action_space.n, p=softmax(Q_table[state])
                )  # Select action using softmax over Q-values
                actions_from_q_table += 1
            except KeyError:
                action = (
                    env.action_space.sample()
                )  # Fallback to random action if state not in Q-table
                actions_from_random += 1
            
            obs, reward, done, info = env.step(action)

            total_reward += reward
            episode_len += 1
            # if gui_flag:
            #     refresh(
            #         obs, reward, done, info, delay=0.1
            #     )  # Update the game screen [GUI only]

        # print("Total reward:", total_reward)
        rewards.append(total_reward)
        len_episodes.append(episode_len)
    end_time_total = time.perf_counter()
    elapsed_time_total = end_time_total - start_time_total
    
    # Number of unique states in Q_table
    print(f"Number of unique states in {filename}: {len(Q_table)}")
    
    # Average reward after 10,000 episodes
    avg_reward = sum(rewards) / len(rewards)
    print(f"Average reward after 10,000 episodes: {avg_reward}")
    
    # Average episode length
    avg_episode_len = sum(len_episodes) / len(len_episodes)
    print(f"Average episode length after 10,000 episodes: {avg_episode_len}")
    
    # Time taken to run the 10,000 episode evaluation
    print(f"Total time for the 10,000 episode evaluation: {elapsed_time_total:.6f} seconds")
    
    # Number of new unique states reached during evalutation
    print(f"Number of unique states reached in evaluation not seen in training: {len(new_states_reached)}")
    
    # Percentage of actions taken from Q_table vs total actions taken
    print(f"Percentage of actions taken from Q_table vs total actions taken: {actions_from_q_table / (actions_from_q_table + actions_from_random)}")
    
    # Percentage of actions taken from randomly vs total actions taken
    print(f"Percentage of actions taken randomly vs total actions taken: {actions_from_random / (actions_from_q_table + actions_from_random)}")