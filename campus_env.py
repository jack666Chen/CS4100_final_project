import gym
from gym import spaces
import numpy as np
import random

# Grid dimensions
GRID_WIDTH = 20
GRID_HEIGHT = 15

# Cell types
EMPTY = 0
WALL = 1
TUNNEL = 2

# Building codes
BUILDINGS = {
    'Richards': 10,
    'Ell': 11,
    'Snell Library': 12,
    'Dodge': 13,
    'Ryder': 14,
    'Hayden': 15,
    'Churchill': 16,
    'Shillman': 17,
    'Forsyth': 18,
    'West Village H': 19,
    'Curry Student Center': 20,
    'Marino': 21,
    'ISEC': 22,
    'Cabot': 23,
    'Mugar': 24,
    'Snell Engineering': 25,
}

# Base traversal times
BASE_TRAVERSAL_TIMES = {
    EMPTY: 2.0,
    TUNNEL: 1.5,
    WALL: float('inf'),
}

# Buildings are accessible waypoints with entry/exit cost
for building_code in BUILDINGS.values():
    BASE_TRAVERSAL_TIMES[building_code] = 2.5

class CampusRouteEnv(gym.Env):
    """Campus route-finding simulation environment"""
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=20, max_steps=1500):
        super(CampusRouteEnv, self).__init__()

        # Grid and episode parameters
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps = 0

        # Grid setup (placeholder: all free)
        # Later we can load this from a map file
        self.map = np.zeros((grid_size, grid_size), dtype=int)
        self.goal = (grid_size - 1, grid_size - 1)
        self.tunnels = {}  # dict: entrance -> exit
        self.crowd_cells = set()  # slower movement cost

        # Define 10 actions
        self.actions = [
            "UP", "DOWN", "LEFT", "RIGHT",
            "UPLEFT", "UPRIGHT", "DOWNLEFT", "DOWNRIGHT",
            "ENTER_TUNNEL", "EXIT_TUNNEL"
        ]
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation space: player position + current time cost
        self.observation_space = spaces.Dict({
            'position': spaces.Tuple((
                spaces.Discrete(self.grid_size),
                spaces.Discrete(self.grid_size)
            )),
            'time_elapsed': spaces.Box(0, np.inf, shape=())
        })

        # Reset for initial state
        self.reset()

    def reset(self):
        self.steps = 0
        self.time_elapsed = 0.0
        self.player_position = (0, 0)
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            'position': self.player_position,
            'time_elapsed': self.time_elapsed
        }

    def step(self, action):
        if isinstance(action, str):
            action = self.actions.index(action)
        action_name = self.actions[action]

        x, y = self.player_position
        new_pos = (x, y)

        # Movement mapping
        moves = {
            "UP": (-1, 0), "DOWN": (1, 0),
            "LEFT": (0, -1), "RIGHT": (0, 1),
            "UPLEFT": (-1, -1), "UPRIGHT": (-1, 1),
            "DOWNLEFT": (1, -1), "DOWNRIGHT": (1, 1)
        }

        # Move logic
        if action_name in moves:
            dx, dy = moves[action_name]
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                # Check for wall
                if self.map[nx, ny] != 1:  # 1 will be wall
                    new_pos = (nx, ny)
            else:
                pass  # Out of bounds, stay still

        elif action_name == "ENTER_TUNNEL":
            if self.player_position in self.tunnels:
                new_pos = self.tunnels[self.player_position]
        elif action_name == "EXIT_TUNNEL":
            # optional: maybe surface to a random nearby exit
            pass

        self.player_position = new_pos
        self.steps += 1

        # Compute time cost (crowd slows agent)
        base_cost = random.uniform(0.5, 1.0)
        if new_pos in self.crowd_cells:
            base_cost *= random.uniform(2.0, 3.0)
        self.time_elapsed += base_cost

        # Rewards / termination
        done = False
        reward = -base_cost  # minimize time
        if new_pos == self.goal:
            done = True
            reward = 1000 - self.time_elapsed  # fast arrival bonus
        elif self.steps >= self.max_steps:
            done = True

        info = {
            "action": action_name,
            "cost": base_cost,
            "goal": self.goal,
            "steps": self.steps
        }

        return self._get_obs(), reward, done, False, info

    def render(self, mode='human'):
        grid = np.full((self.grid_size, self.grid_size), '.')
        gx, gy = self.goal
        grid[gx, gy] = 'G'
        x, y = self.player_position
        grid[x, y] = 'A'
        print("\n".join("".join(row) for row in grid))
        print(f"Time: {self.time_elapsed:.2f} | Steps: {self.steps}")

