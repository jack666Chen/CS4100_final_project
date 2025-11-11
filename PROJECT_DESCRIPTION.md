# Northeastern Smart Campus Navigator Agent

**CS4100: Foundations of Artificial Intelligence**  
**Team Members:** Keelan Donovan, Francisco Chen, Ziang (Jack) Chen  

---

## 1. Project Overview

The **Smart Campus Navigator** is a reinforcement learning project that trains an agent to find efficient walking routes across Northeastern University’s campus under dynamic conditions.  


The environment is modeled as a **two-layer grid**, representing both **outdoor walkways** and the **underground tunnel network** that runs beneath parts of campus.



The agent uses **tabular Q-learning** to learn optimal movement policies across this environment.  


In each episode, the agent starts from a random valid outdoor location and must reach a designated **goal building** such as Snell Library, ISEC, or Curry.


Movement is **deterministic**: valid actions transition the agent to the neighboring cell or toggle between surface and tunnel layers, while **invalid actions** (off-grid or blocked) leave the agent in place and apply a penalty.


Environmental conditions—**weather (clear, rain, snow)** and **crowd level (low, medium, high)**—change each episode, influencing traversal costs. Across training, the agent learns to minimize expected travel time by adapting its route to these dynamic conditions and by leveraging the tunnel network when beneficial.

At the end of training, the project produces a **set of Q-tables**, each corresponding to a different goal destination on campus.
When a user selects a destination, the system loads the pre-trained Q-table for that goal to generate an optimal route in real time.

---

## 2. Project Definition

### Inputs
- **Campus Grid Representation (15×15×2):**
  - Two layers:  
    - **Layer 0 (Surface):** outdoor paths between major landmarks (e.g., ISEC, Marino, Curry, West Village, Snell).  
    - **Layer 1 (Tunnels):** underground network connecting select buildings (e.g., Richards ↔ Curry ↔ Snell).  
  - Each cell stores:  
    - Base traversal time (normalized 1–10)  
    - Path type (`outdoor`, `tunnel`, `invalid`)  
  - **Permanently blocked cells** (`invalid`) represent non-walkable terrain.  
  - **Tunnel entrances/exits** enable transitions between layers via the `toggle_layer` action.

- **Environmental Parameters (per episode):**
  - `weather_condition ∈ {clear, rain, snow}`
  - `crowd_level ∈ {low, medium, high}`

- **Learning Parameters:**
  - Learning rate (α)  
  - Discount factor (γ)  
  - Exploration rate (ε), exponentially decaying

- **Simulation Settings:**
  - Number of episodes: varies by experiment  
  - Step limit: 1000  
  - Randomized start coordinates (surface layer)  
  - Fixed goal location per training run (varies across Q-tables)

---

### State, Actions, and Transitions

- **State:** `(x, y, layer, weather, crowd)`  
  - Captures the agent’s position, environment layer, and conditions.  
  - Each state is encoded into a **unique integer index** for efficient Q-table access
- **Actions:** `{up, down, left, right, move_inside, move_outside}`  

- **Transitions (deterministic):**
  - **Valid move:** advance to a neighboring cell or switch layers at tunnel entrances/exits.  
  - **Invalid move:** remain in place and receive a penalty.  

---

### Reward Function

The reward reflects **deterministic traversal cost** under current environmental conditions and layer.

**Reward Equation:**  
R(s, a, s') = -[ t_base(s, a, layer) * (1 + α_w(layer) * weather + α_c(layer) * crowd) ] - P_invalid(s, a)


**Parameters:**
- `t_base(s, a, layer)`: base traversal time (lower in tunnels).  
- `α_w(layer)`: weather weight (0 for tunnels; >0 for surface).  
- `α_c(layer)`: crowd weight (smaller for tunnels).  
- `P_invalid(s, a)`: −20 for invalid moves.  
- Terminal reward: +1000 on reaching the goal.  
- Timeout penalty: −50 if the step limit is reached.  

This setup encourages the agent to prefer tunnels during harsh conditions and to optimize total travel time across varying weather and crowd levels.

---

## 3. Outputs

- **Multiple Learned Q-Tables:**  
  One per destination goal (e.g., Snell, ISEC, Curry, WVH).  
  Each table maps every `(x, y, layer, weather, crowd)` state to optimal actions minimizing travel time to that goal.

- **Runtime Route Selection:**  
  At runtime, when a user selects a goal, the system loads the corresponding Q-table and uses greedy policy evaluation (ε ≈ 0) to compute the optimal path.

- **Visualization and Analysis:**  
  - Q-value heatmaps across layers  
  - Policy maps for representative weather/crowd settings  
  - Learning curves showing cumulative reward and episode length convergence  
  - Metrics summarizing success rate, average travel time, and invalid-action rate

---

## 4. Evaluation Plan

### Performance Evaluation
- Plot cumulative reward, episode length, and success rate per goal Q-table.  
- Compare surface vs tunnel route utilization under different environmental conditions.  
- Track decline in invalid-action frequency and improvement in reward over episodes.  

### Convergence and Generalization
- Convergence identified by stable cumulative reward and minimal ΔQ ≤ 1e-3.  
- Generalization evaluated by testing trained Q-tables on unseen weather/crowd combinations.  
- Compare relative difficulty across destinations based on convergence speed and final performance.

### Visualization Tools
Implemented in **Python** using:
- **Matplotlib** for grid and route visualizations  
- **Seaborn** for heatmaps and learning curves  
- **NumPy/Pandas** for metrics logging and data analysis  

---

## 5. To-Do Items

1. **Environment Finalization**
   - Encode surface and tunnel layers with valid/invalid cells.  
   - Define tunnel entrance/exit coordinates and toggle-layer transitions.  
   - Implement deterministic traversal cost computation with layer-specific multipliers.  

2. **Reward and Learning Integration**
   - Implement reward calculation with layer-dependent costs and penalties.  
   - Implement and test the **state-encoding function** for Q-table indexing.  
   - Integrate Q-learning with ε-greedy exploration.  
   - Log cumulative rewards, episode lengths, and invalid-action rates.  

3. **Training and Evaluation**
   - Train separate Q-tables for each selected goal location.  
   - Evaluate convergence and performance for each goal individually.  
   - Compare learned behaviors across environmental conditions.  

4. **Visualization and Reporting**
   - Generate policy and Q-value heatmaps for both layers.  
   - Produce route visualizations and comparative metrics per goal.  
   - Compile findings and analysis for the final report and presentation.  

---

## 6. Team Member Roles and Contributions

Each team member contributes equally to environment setup, Q-learning implementation, and evaluation/visualization.

| Member | Responsibilities |
|---|---|
| **Keelan** | Designs grid structure and layer logic (surface/tunnel), implements cost and reward modeling, and produces policy/Q-value visualizations. |
| **Francisco** | Develops transition logic including tunnel toggles, integrates Q-learning exploration and update rules, and generates learning/convergence plots. |
| **Jack** | Builds data structures and metrics tracking, analyzes performance (success rate, invalid actions, tunnel usage), and leads visualization of route outcomes. |

All members collaborate on code integration, report writing, and final presentation preparation.

---

## 7. Development Environment

All development for the Smart Campus Navigator Agent is done using **Python 3.11.5** in a virtual environment to ensure consistent results across systems.

### Setup Instructions

```bash
# Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate   # (use venv\Scripts\activate on Windows)

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

