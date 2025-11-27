import sys
import time
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from campus_gui import *


def hash_obs(obs):
    key = (
        obs['position'][0],
        obs['position'][1],
        obs['weather'],
        obs['layer'],
        obs['can_toggle_layer'],
        obs['nearest_crowd'][0],
        bs['nearest_crowd'][1],
    )
    return key
