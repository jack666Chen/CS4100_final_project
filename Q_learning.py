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
    episodes = []
    crowd_ratios = []
    
    for episode in tqdm(range(num_episodes)):
        total_reward = 0
        obs, reward, done, info = env.reset()
        
        total_reward += reward

        # Track crowd for this episode
        episode_steps = 0
        crowd_steps = 0
        
        while not done:
            state = hash_obs(obs)

            # Check if agent is in crowd
            if obs.get('is_crowd', 0) == 1:
                crowd_steps += 1
            episode_steps += 1
            
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
            
        # Calculate crowd contact ratio for this episode
        if episode_steps > 0:
            crowd_ratio = crowd_steps / episode_steps
        else:
            crowd_ratio = 0
        crowd_ratios.append(crowd_ratio)
        
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

    # Generate crowd contact ratio plot
    crowd_x = x
    crowd_y = crowd_ratios
    avg_crowd_ratios = np.cumsum(crowd_y) / np.arange(1, len(crowd_y) + 1)
    
    max_crowd = max(crowd_y) if crowd_y else 0
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(crowd_x, crowd_y, alpha=0.3, color='blue', label='Episode Crowd Ratio', linewidth=0.5)
    plt.plot(crowd_x, avg_crowd_ratios, color='red', label='Average Crowd Ratio', linewidth=2)
    plt.xlabel('Episode', fontsize=14, fontweight='bold')
    plt.ylabel('Crowd Contact Ratio', fontsize=14, fontweight='bold')
    plt.title('Agent Crowd Contact Ratio Over Training Episodes', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    if max_crowd > 0:
        plt.ylim([0, max_crowd * 1.1])
    else:
        plt.ylim([0, 0.1])
    plt.tight_layout()
    plt.savefig('crowd_contact_ratio_over_episodes.png', dpi=300, bbox_inches='tight')
    print("Crowd contact ratio plot saved as 'crowd_contact_ratio_over_episodes.png'")
    
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

    # Data collection
    weather_conditions = ["clear", "rain", "snow"]
    layer_usage_data = {}
    toggle_usage_data = {}
    time_data = {}
    for weather in weather_conditions:
        layer_usage_data[weather] = {"surface": 0, "tunnel": 0}
        toggle_usage_data[weather] = {"can_toggle_opportunities": 0, "toggle_choices": 0}
        time_data[weather] = []
    
    for episode in tqdm(range(10000)):
        obs, reward, done, info = env.reset()
        total_reward = 0
        episode_len = 0
        weather = obs['weather']

        while not done:
            state = hash_obs(obs)
            # Collect layer usage data
            if weather in layer_usage_data:
                if obs['layer'] == 0:
                    layer_usage_data[weather]["surface"] += 1
                else:
                    layer_usage_data[weather]["tunnel"] += 1
            
            # Check if agent is at a toggle building before action
            can_toggle = obs.get('can_toggle_layer', 0)
            current_layer = obs.get('layer', 0) 
            
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

            # Collect toggle choice data
            # Only count toggles from surface (layer 0) to tunnel (layer 1)
            if weather in toggle_usage_data and can_toggle == 1 and current_layer == 0:
                toggle_usage_data[weather]["can_toggle_opportunities"] += 1
                if action == 8:  # TOGGLE_LAYER action
                    toggle_usage_data[weather]["toggle_choices"] += 1
            
            obs, reward, done, info = env.step(action)

            total_reward += reward
            episode_len += 1
            # if gui_flag:
            #     refresh(
            #         obs, reward, done, info, delay=0.1
            #     )  # Update the game screen [GUI only]
            if gui_flag:
                refresh(obs, reward, done, info, delay=0.1)

        # Collect final time for this episode
        final_time = obs.get('time', 0)
        if weather in time_data:
            time_data[weather].append(final_time)
            
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

    # Visualize data to analysis agent performance
    # Generate layer usage visualization
    weather_labels = ["Clear", "Rain", "Snow"]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(3)
    width = 0.6
    
    tunnel_values = []
    surface_values = []
    for weather in weather_conditions:
        surface_count = layer_usage_data[weather]["surface"]
        tunnel_count = layer_usage_data[weather]["tunnel"]
        total = surface_count + tunnel_count
        if total > 0:
            tunnel_ratio = tunnel_count / total
            surface_ratio = surface_count / total
        else:
            tunnel_ratio = 0
            surface_ratio = 0
        tunnel_values.append(tunnel_ratio)
        surface_values.append(surface_ratio)
    
    ax.bar(x, tunnel_values, width, label='Tunnel', color='red', 
    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.bar(x, surface_values, width, bottom=tunnel_values, label='Surface', color='blue',
    alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for i in range(3):
        t_val = tunnel_values[i]
        s_val = surface_values[i]
        ax.text(i, t_val / 2, f'{t_val:.2%}', ha='center', va='center', fontweight='bold', fontsize=11, color='white')
        ax.text(i, t_val + s_val / 2, f'{s_val:.2%}', ha='center', va='center', fontweight='bold', fontsize=11, color='black')
    
    ax.set_ylabel('Usage Rate', fontsize=14, fontweight='bold')
    ax.set_xlabel('Weather Condition', fontsize=14, fontweight='bold')
    ax.set_title('Surface vs Tunnel Usage Rate Comparison Across Weather Conditions', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(weather_labels, fontsize=12)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('layer_usage_comparison.png', dpi=300, bbox_inches='tight')
    print("Layer usage comparison chart saved as 'layer_usage_comparison.png'")
    
    # Generate toggle probability visualization
    weather_labels = ["Clear", "Rain", "Snow"]
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(3)
    width = 0.6
    
    toggle_probabilities = []
    for weather in weather_conditions:
        opportunities = toggle_usage_data[weather]["can_toggle_opportunities"]
        toggle_choices = toggle_usage_data[weather]["toggle_choices"]
        if opportunities > 0:
            probability = toggle_choices / opportunities
        else:
            probability = 0
        toggle_probabilities.append(probability)
    
    colors = ['lightgreen', 'skyblue', 'lightgray']
    
    bars = ax.bar(x, toggle_probabilities, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for i in range(3):
        val = toggle_probabilities[i]
        ax.text(i, val * 1.05, f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Toggle Probability', fontsize=14, fontweight='bold')
    ax.set_xlabel('Weather Condition', fontsize=14, fontweight='bold')
    ax.set_title('TOGGLE_LAYER Probability at Toggle Buildings Across Weather Conditions', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(weather_labels, fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    max_prob = max(toggle_probabilities) if toggle_probabilities else 0.01
    ax.set_ylim([0, max_prob * 1.2])
    
    plt.tight_layout()
    plt.savefig('toggle_probability_comparison.png', dpi=300, bbox_inches='tight')
    print("Toggle probability comparison chart saved as 'toggle_probability_comparison.png'")
    
    # Generate average time comparison
    weather_labels = ["Clear", "Rain", "Snow"]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    avg_times = []
    for weather in weather_conditions:
        times = time_data[weather]
        if len(times) > 0:
            avg_time = np.mean(times)
        else:
            avg_time = 0
        avg_times.append(avg_time)
    
    x = np.arange(3)
    width = 0.6
    colors = ['lightgreen', 'skyblue', 'lightgray']
    
    bars = ax.bar(x, avg_times, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for i in range(3):
        val = avg_times[i]
        ax.text(i, val + max(avg_times) * 1.05, f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Average Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Weather Condition', fontsize=14, fontweight='bold')
    ax.set_title('Average Completion Time Across Weather Conditions', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(weather_labels, fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    max_time = max(avg_times) if avg_times else 100
    ax.set_ylim([0, max_time * 1.2])
    
    plt.tight_layout()
    plt.savefig('average_time_comparison.png', dpi=300, bbox_inches='tight')
    print("Average time comparison chart saved as 'average_time_comparison.png'")
    
