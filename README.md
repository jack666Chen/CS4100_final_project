# Northeastern Campus Navigator Guide
The **Smart Campus Navigator** is a reinforcement learning project that trains an agent to find efficient walking routes across Northeastern University’s campus under dynamic conditions.  


The environment is modeled as a **two-layer grid**, representing both **outdoor walkways** and the **underground tunnel network** that runs beneath parts of campus.



The agent uses **tabular Q-learning** to learn optimal movement policies across this environment.  


In each episode, the agent starts from a random valid outdoor location and must reach a designated **goal building** such as Snell Library, ISEC, or Curry.


Movement is **deterministic**: valid actions transition the agent to the neighboring cell or toggle between surface and tunnel layers, while **invalid actions** (off-grid or blocked) leave the agent in place and apply a penalty.


Environmental conditions—**weather (clear, rain, snow)** and **crowd level (low, medium, high)**—change each episode, influencing traversal costs. Across training, the agent learns to minimize expected travel time by adapting its route to these dynamic conditions and by leveraging the tunnel network when beneficial.

At the end of training, the project produces a **set of Q-tables**, each corresponding to a different goal destination on campus.
When a user selects a destination, the system loads the pre-trained Q-table for that goal to generate an optimal route in real time.


## How to Run the Agent

### Step 1: Clone the Repository

```bash
git clone <repo_url>
cd <repo_name>
```

### Step 2: Install Dependencies

Make sure you have Python 3.8+ installed on your system. Then, install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 3: Change code configuration

As long as you have our code, you are able to move into the `Q_learning.py` file and find the line:

```bash
"""
Specify number of episodes and decay rate for training and evaluation.
"""
num_episodes = <num you want>
decay_rate = <num you want>
```

You can modify the episodes and the decay rate to the number you want.

### Optional Step 4: Train

Use python to run the file with train flag, it will train data based on you setted episodes and decay_rate:

```bash
python Q_learning.py train
````

(depends on your system, you have to use the correct command, but either way you should have train flag)


### Step 5: Evaluation

Similar to the above, but you have two choices right now, if you want see the agent moving, then:

```bash
python Q_learning.py gui
````

If you don't care how it moves, but only want it finish moving and see the evaluation quick, then:

```bash
python Q_learning.py gui
````

In either way, you should have your episodes and decay_rate set properly, so that there is a pickle file named from it.
By default, we offer a 1000000 episodes and 0.999999 decay trained data uploaded. You may directly use it, but remember to set it as step 3.

## How to play with the environment
### Step 1: Clone the Repository

```bash
git clone <repo_url>
cd <repo_name>
```

### Step 2: Install Dependencies

Make sure you have Python 3.8+ installed on your system. Then, install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 3: Run the GUI

```bash
python campus_gui.py
```

You should be able to see what ever is on the gui.
You should be able to move the agent around,
or change layers,
or wait,
or hide the line between grid (better looking, not affecting the model)

## Evaluation Plan

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
