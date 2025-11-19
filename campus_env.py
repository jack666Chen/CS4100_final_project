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
TUNNEL = 3

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
