import gym
from gym import spaces
import numpy as np
import random

# Grid dimensions - reduced for tabular Q-learning
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

# Base traversal times (normalized 1-10 scale)
BASE_TRAVERSAL_TIMES = {
    EMPTY: 2.0,
    TUNNEL: 1.5,
    WALL: float('inf'),
}

# Buildings are accessible waypoints with entry/exit cost
for building_code in BUILDINGS.values():
    BASE_TRAVERSAL_TIMES[building_code] = 0.5
    
class CampusEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, goal_building: str = 'Snell Library'):
        super(CampusEnv, self).__init__()
        
        # Grid and episode parameters
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        self.max_steps = 1000
        self.steps = 0
        self.goal_building = goal_building
        
        # Create map layers
        self.surface_map = self._create_surface_grid()
        self.tunnel_map = self._create_tunnel_grid()
        
        # Environmental conditions
        self.weather_conditions = ['clear', 'rain', 'snow']
        self.crowd_levels = ['low', 'medium', 'high']
        
        # Weather impact multipliers (layer-specific)
        self.weather_multipliers = {
            'surface': {'clear': 1.0, 'rain': 1.3, 'snow': 1.6},
            'tunnel': {'clear': 1.0, 'rain': 1.0, 'snow': 1.0}
        }
        
        # Crowd impact multipliers (layer-specific)
        self.crowd_multipliers = {
            'surface': {'low': 1.0, 'medium': 1.2, 'high': 1.5},
            'tunnel': {'low': 1.0, 'medium': 1.05, 'high': 1.15}
        }
        
        # Buildings with tunnel access
        self.tunnel_buildings = [
            'Richards', 'Ell', 'Dodge', 'Cabot', 'Mugar',
            'Curry Student Center', 'Snell Library',
            'Churchill', 'Hayden', 'Forsyth', 'Snell Engineering'
        ]
        
        # Rewards
        self.rewards = {
            'goal': 1000,
            'invalid_action': -20,
            'timeout': -50,
        }
        
        # Actions
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'TOGGLE_LAYER', 'WAIT']
        self.action_space = spaces.Discrete(len(self.actions))
        
        # Observation space
        self.observation_space = spaces.Dict({
            'position': spaces.Tuple((
                spaces.Discrete(self.grid_width),
                spaces.Discrete(self.grid_height)
            )),
            'layer': spaces.Discrete(2),
            'weather': spaces.Discrete(len(self.weather_conditions)),
            'crowd': spaces.Discrete(len(self.crowd_levels)),
            'at_goal': spaces.Discrete(2)
        })
        
        # Set initial state
        self.reset()
    
    def create_surface_grid(self) -> np.ndarray:
      grid = np.zeros((self.grid_height, self.grid_width))
      return grid
    
    def create_tunnel_grid(self) -> np.ndarray:
      grid = np.full((self.grid_height, self.grid_width), WALL)
      return grid
    
    def reset(self):
      return self.get_observation, 0, False, {}
    
    def get_observation(self):
      obs = {
        'position': "",
        'layer': "",
        'weather': "",
        'crowd': "",
        'at_goal': "",
        'current_building': ""
      }
      return obs
    
    def is_terminal(self):
      return False
    
    def move_player(self, action):
      return ""
    
    def move_player_to_random_adjacent(self):
      return ""
    
    def try_toggle_layer(self):
      return ""
    
    def play_turn(self, action):
      return ""
    
    def step(self, actions):
      observation = ""
      reward = ""
      done = ""
      info = "" # empty strings are placeholders
      return observation, reward, done, info
    
    def render(self, mode='human'):
      print("")
      
    

