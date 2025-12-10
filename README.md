# Northeastern Campus Navigator Guide
Northeastern Campus Navigator is a Python based model, Python coded Reinforcement AI system of route guiding. This an AI-based navigator that aims to help student at Northeastern guide a good path to the goal destination.


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
