import gym
from gym import spaces
import numpy as np
import random

# Grid dimensions - reduced for tabular Q-learning
GRID_WIDTH = 20
GRID_HEIGHT = 20

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
      
      building_mappings = [
        # (x, y, building_code)
        (9, 1, BUILDINGS['Marino']),
        (10, 4, BUILDINGS['Cabot']),
        (9, 8, BUILDINGS['Forsyth']),
        (2, 4, BUILDINGS['West Village H']),
        (11, 9, BUILDINGS['Churchill']),
        (13, 8, BUILDINGS['Hayden']),
        (13, 4, BUILDINGS['Richards']),
        (15, 6, BUILDINGS['Ell']),
        (16, 4, BUILDINGS['Dodge']),
        (18, 7, BUILDINGS['Mugar']),
        (12, 11, BUILDINGS['Snell Library']),
        (9, 11, BUILDINGS['Snell Engineering']),
        (15, 10, BUILDINGS['Curry Student Center']),
        (2, 15, BUILDINGS['Ryder']),
        (5, 12, BUILDINGS['Shillman']),
        (13, 16, BUILDINGS['ISEC'])
      ]
      
      for x, y, code in building_mappings:
            if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
                grid[y, x] = code
                
      return grid
    
    def create_tunnel_grid(self) -> np.ndarray:
      grid = np.full((self.grid_height, self.grid_width), WALL, dtype=np.int32)
        
      # Accurate tunnel corridor coordinates
      tunnel_coordinates = [
          (13, 4), (13, 5), (13, 6), (13, 7), (13, 8),  # Richards to Hayden
          (12, 7), (11, 7), (10, 7), (10, 5), (10, 4),  # Richards to Cabot
          (12, 8), (11, 8), (11, 9),                    # Hayden to Churchill
          (10, 9), (9, 9), (9, 8),                      # Churchill to Forsyth
          (9, 10), (9, 11),                             # Between Churchill/Forsyth to Snell Eng
          (12, 8), (12, 9), (12, 10), (12, 11),        # Hayden to Snell Library
          (14, 7), (14, 6), (15, 6),                   # Richards to Ell
          (16, 6), (16, 5), (16, 4),                   # Ell to Dodge
          (17, 6), (18, 6), (18, 7)                    # Dodge to Mugar
      ]
      
      # Mark tunnel corridors as walkable
      for x, y in tunnel_coordinates:
          if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
              grid[y, x] = TUNNEL
      
      # Building tunnel entrances (11 buildings with tunnel access)
      tunnel_building_entrances = [
          (13, 4, BUILDINGS['Richards']),
          (15, 6, BUILDINGS['Ell']),
          (16, 4, BUILDINGS['Dodge']),
          (10, 4, BUILDINGS['Cabot']),
          (18, 7, BUILDINGS['Mugar']),
          (15, 10, BUILDINGS['Curry Student Center']),
          (12, 11, BUILDINGS['Snell Library']),
          (9, 11, BUILDINGS['Snell Engineering']),
          (11, 9, BUILDINGS['Churchill']),
          (13, 8, BUILDINGS['Hayden']),
          (9, 8, BUILDINGS['Forsyth']),
      ]
      
      # Place building entrances in tunnel grid
      for x, y, code in tunnel_building_entrances:
          if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
              grid[y, x] = code
      
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
      
    

