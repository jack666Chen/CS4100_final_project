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
    "Richards": 10,
    "Ell": 11,
    "Snell Library": 12,
    "Dodge": 13,
    "Ryder": 14,
    "Hayden": 15,
    "Churchill": 16,
    "Shillman": 17,
    "Forsyth": 18,
    "West Village H": 19,
    "Curry Student Center": 20,
    "Marino": 21,
    "ISEC": 22,
    "Cabot": 23,
    "Mugar": 24,
    "Snell Engineering": 25,
}

BUILDING_MAPPINGS = [
    # (x, y, building_code)
    ([(9, 1), (10, 1), (9, 0), (10, 0)], BUILDINGS["Marino"]),
    ([(10, 4), (9, 4), (10, 5), (9, 5), (10, 6), (9, 6)], BUILDINGS["Cabot"]),
    ([(9, 8), (9, 9), (8, 8), (8, 9)], BUILDINGS["Forsyth"]),
    ([(2, 4), (1, 4), (1, 5)], BUILDINGS["West Village H"]),
    ([(11, 9)], BUILDINGS["Churchill"]),
    (
        [(13, 8), (13, 6), (13, 7), (12, 6), (12, 7), (12, 8)],
        BUILDINGS["Hayden"],
    ),
    ([(13, 4), (12, 3), (13, 3), (13, 5), (12, 5)], BUILDINGS["Richards"]),
    ([(15, 6), (16, 6), (16, 7), (15, 7)], BUILDINGS["Ell"]),
    ([(17, 4), (17, 3), (17, 5)], BUILDINGS["Dodge"]),
    ([(19, 7), (18, 7)], BUILDINGS["Mugar"]),
    ([(12, 11), (13, 12), (12, 12), (13, 11)], BUILDINGS["Snell Library"]),
    ([(9, 11), (8, 11), (10, 11)], BUILDINGS["Snell Engineering"]),
    (
        [(15, 10), (16, 10), (16, 9), (16, 8), (15, 8), (15, 9)],
        BUILDINGS["Curry Student Center"],
    ),
    ([(2, 15), (2, 16), (1, 15), (1, 16)], BUILDINGS["Ryder"]),
    ([(5, 11), (4, 11), (3, 11)], BUILDINGS["Shillman"]),
    ([(13, 16), (14, 16), (13, 17), (14, 17)], BUILDINGS["ISEC"]),
]

WALL_MAPPINGS = [  # (y, [x0, x1, ..., ])
    (0, [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
    (2, [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]),
    (3, [18, 19]),
    (4, [4, 5, 7, 18, 19]),
    (5, [3, 7, 18, 19]),
    (6, [2, 3, 5]),
    (7, [0, 1, 2, 6]),
    (8, [0, 1, 3, 4, 6]),
    (9, [0, 1, 3, 4]),
    (10, [18, 19]),
    (11, [0, 1, 18, 19]),
    (12, [0, 1, 17, 18, 19]),
    (13, [8, 9, 10, 15, 16, 17, 18, 19]),
    (14, [8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]),
    (15, [0, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]),
    (16, [0, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19]),
    (17, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19]),
    (18, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19]),
    (19, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19]),
]

TUNNEL_COORDINATES = [
    (10, 8),
    (9, 10),
    (11, 6),
    (11, 8),
    (13, 9),
    (13, 10),
    (14, 6),
    (17, 6),
    (17, 7),
    (17, 9),
    (18, 8),
    (18, 9),
]

TUNNEL_BUILDING_ENTRANCES = [
    ([(13, 4), (12, 3), (13, 3), (13, 5), (12, 5)], BUILDINGS["Richards"]),
    ([(15, 6), (16, 6), (16, 7), (15, 7)], BUILDINGS["Ell"]),
    ([(17, 4), (17, 3), (17, 5)], BUILDINGS["Dodge"]),
    ([(10, 4), (9, 4), (10, 5), (9, 5), (10, 6), (9, 6)], BUILDINGS["Cabot"]),
    ([(19, 7), (18, 7)], BUILDINGS["Mugar"]),
    (
        [(15, 10), (16, 10), (16, 9), (16, 8), (15, 8), (15, 9)],
        BUILDINGS["Curry Student Center"],
    ),
    ([(12, 11), (13, 12), (12, 12), (13, 11)], BUILDINGS["Snell Library"]),
    ([(9, 11), (8, 11), (10, 11)], BUILDINGS["Snell Engineering"]),
    ([(11, 9)], BUILDINGS["Churchill"]),
    (
        [(13, 8), (13, 6), (13, 7), (12, 6), (12, 7), (12, 8)],
        BUILDINGS["Hayden"],
    ),
    ([(9, 8), (9, 9), (8, 8), (8, 9)], BUILDINGS["Forsyth"]),
]

# Base traversal times (normalized 1-10 scale)
BASE_TRAVERSAL_TIMES = {
    EMPTY: 2.0,
    TUNNEL: 1.5,
    WALL: float("inf"),
}

# Buildings are accessible waypoints with entry/exit cost
for building_code in BUILDINGS.values():
    BASE_TRAVERSAL_TIMES[building_code] = 0.5


class CampusEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, goal_building: str = "Snell Library"):
        super(CampusEnv, self).__init__()

        # Grid and episode parameters
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT

        self.max_steps = 1000
        self.steps = 0

        self.max_time = 100
        self.time = 0.0

        self.goal_building = goal_building
        self.goal_building_code = BUILDINGS[goal_building]

        # Create map layers
        self.surface_map = self.create_surface_grid()
        self.tunnel_map = self.create_tunnel_grid()

        # Environmental conditions
        self.weather_conditions = ["clear", "rain", "snow"]
        self.crowd_levels = ["low", "medium", "high"]

        # Weather impact multipliers (layer-specific)
        self.weather_multipliers = {
            "surface": {"clear": 1.0, "rain": 1.3, "snow": 1.6},
            "tunnel": {"clear": 1.0, "rain": 1.0, "snow": 1.0},
        }

        # Crowd impact multipliers (layer-specific)
        self.crowd_multipliers = {
            "surface": {"low": 1.0, "medium": 1.2, "high": 1.5},
            "tunnel": {"low": 1.0, "medium": 1.05, "high": 1.15},
        }

        self.tunnel_building_codes = {code for pos, code in TUNNEL_BUILDING_ENTRANCES}

        # Rewards
        self.rewards = {
            "goal": 1000,
            "invalid_action": -20,
            "timeout": -50,
        }

        # Actions
        self.actions = [
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT",
            "UP - L",
            "UP - R",
            "DOWN - L",
            "DOWN - R",
            "TOGGLE_LAYER",
            "WAIT",
        ]
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Tuple(
                    (
                        spaces.Discrete(self.grid_width),
                        spaces.Discrete(self.grid_height),
                    )
                ),
                "layer": spaces.Discrete(2),
                "weather": spaces.Discrete(len(self.weather_conditions)),
                "crowd": spaces.Discrete(len(self.crowd_levels)),
            }
        )

        # Set initial state
        self.reset()

    def create_surface_grid(self) -> np.ndarray:
        grid = np.zeros((self.grid_height, self.grid_width))
        for pos, code in BUILDING_MAPPINGS:
            for x, y in pos:
                if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
                    grid[y, x] = code

        for y, x_values in WALL_MAPPINGS:
            for x in x_values:
                if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
                    grid[y, x] = WALL

        return grid

    def create_tunnel_grid(self) -> np.ndarray:
        grid = np.full((self.grid_height, self.grid_width), WALL, dtype=np.int32)
        # Mark tunnel corridors as walkable
        for x, y in TUNNEL_COORDINATES:
            if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
                grid[y, x] = TUNNEL
        # Place building entrances in tunnel grid
        for pos, code in TUNNEL_BUILDING_ENTRANCES:
            for x, y in pos:
                if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
                    grid[y, x] = code
        return grid

    def reset(self):
        """
        Reset the player and map(weather, crowd) to the initial state
        """
        # Reset step counter
        self.steps = 0

        # Reset time
        self.time = 0.0
    
        # Randomly choose starting layer (0 = surface, 1 = tunnel)
        self.layer = random.randint(0, 1)
    
        # Reset the weather
        self.weather = random.choice(self.weather_conditions)
    
        # Choose a random starting position for the player that is not the goal
        goal_building_code = BUILDINGS.get(self.goal_building)
        valid_positions = []
    
        # Select the appropriate map based on layer
        current_map = self.surface_map if self.layer == 0 else self.tunnel_map
    
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell_value = current_map[y, x]
            
                if cell_value != WALL and cell_value != goal_building_code:
                    valid_positions.append((x, y))
    
        if not valid_positions:
            self.position = (0, 0)
            crowd_positions = {}
        else:
            self.position = random.choice(valid_positions)
            # Randomly select 10 unique crowded positions (no duplicates, excluding player position)
            available_positions = [pos for pos in valid_positions if pos != self.position]
            num_crowded = min(10, len(available_positions))
            if num_crowded > 0:
                selected_positions = random.sample(available_positions, num_crowded)
                # we assign a random crowd level to each selected position
                crowd_positions = {pos: random.choice(self.crowd_levels) for pos in selected_positions}
            else:
                crowd_positions = {}

        self.current_state = {
            'time': self.time,
            'position': self.position,
            'layer': self.layer,
            'weather': self.weather,
            'crowd_positions': crowd_positions,
        }
      
        return self.get_observation(), 0, False, {}

    def get_observation(self):
        player_position = self.current_state['position']
        px, py = player_position
        crowd_positions = self.current_state['crowd_positions']

        layer = self.current_state['layer']
        current_map = self.surface_map if self.layer == 0 else self.tunnel_map
        
        goal_building_code = BUILDINGS.get(self.goal_building)
        
        # Get current cell information
        current_building = 0
        is_wall = 1
        can_toggle_layer = 0
        is_crowd = 0
        is_goal = 0
        
        if 0 <= py < self.grid_height and 0 <= px < self.grid_width:
            cell_value = current_map[py, px]
            
            # Get building code
            current_building = cell_value if cell_value in BUILDINGS.values() else 0
            
            # Check if it's a wall
            is_wall = 1 if cell_value == WALL else 0
            
            # Check if can toggle layer
            if current_building > 0:
                can_toggle_layer = 1 if current_building in self.tunnel_building_codes else 0
            
            # Check if the current position is in the crowded positions
            # and get the crowd level for this position (default to "low")
            if (px, py) in crowd_positions:
                is_crowd = 1
                crowd = crowd_positions[(px, py)]
            else:
                crowd = "low"
            
            # Check if at goal
            is_goal = 1 if cell_value == goal_building_code else 0
        else:
            crowd = "low"
        
        weather = self.current_state['weather']
        time = self.current_state['time']

        obs = {
            'position': (px, py),
            'layer': layer,
            'weather': weather,
            'is_crowd': is_crowd,
            'crowd': crowd,
            'current_building': current_building,
            'is_wall': is_wall,
            'can_toggle_layer': can_toggle_layer,
            'is_goal': is_goal,
            'time': time
        }

        return obs

    def is_terminal(self):
        x, y = self.current_state["position"]
        current_map = self.surface_map if self.current_state["layer"] == 0 else self.tunnel_map
        if current_map[y, x] == self.goal_building_code:
            return "goal"
        if self.steps >= self.max_steps:
            return "truncated"
        return False
    
    
    def move_player(self, action):
        x, y = self.current_state["position"]
        directions = {
            0: (x, y - 1),  # UP
            1: (x, y + 1),  # DOWN
            2: (x - 1, y),  # LEFT
            3: (x + 1, y),  # RIGHT
            4: (x - 1, y - 1),  # UP - L
            5: (x + 1, y - 1),  # UP - R
            6: (x - 1, y + 1),  # DOWN - L
            7: (x + 1, y + 1),  # DOWN - R
        }
        new_position = directions.get(action, self.current_state["position"])
        current_map = (
            self.surface_map if self.current_state["layer"] == 0 else self.tunnel_map
        )

        # Ensure new position is within bounds
        if (
            0 <= new_position[0] < self.grid_width
            and 0 <= new_position[1] < self.grid_height
        ):
            if random.random() <= 0.99:
                self.current_state["position"] = new_position
            else:
                # 1% chance to move to a random adjacent cell
                adjacent_positions = [
                    directions[act] for act in directions if act != action
                ]
                adjacent_positions = [
                    pos
                    for pos in adjacent_positions
                    if 0 <= pos[0] < self.grid_width and 0 <= pos[1] < self.grid_height
                ]
                if adjacent_positions:
                    self.current_state["position"] = random.choice(adjacent_positions)
            new_x, new_y = self.current_state["position"]

            if current_map[new_y, new_x] == WALL:
                return "Hit and moved into wall!", self.rewards.get("enter_wall", -10)
            else:
                return f"Moved to {self.current_state['position']}", 0
        else:
            return "Out of bounds!", self.rewards.get("oob", -5)

    def move_crowd_random(self):
        """
        Move each crowd to a random adjacent position, and confirm that each crowd has a unique position.
        Each crowd maintains its own crowd level when moving.
        """
        current_map = self.surface_map if self.current_state['layer'] == 0 else self.tunnel_map
        crowd_positions = self.current_state['crowd_positions']
        player_position = self.current_state['position']
        
        new_crowd_positions = {}
        # Include all old crowd positions and player position as occupied initially
        occupied_positions = {player_position}
        occupied_positions.update(crowd_positions.keys())
        
        for old_position, crowd_level in crowd_positions.items():
            x, y = old_position
            directions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x, y)]
            
            adjacent_positions = [
                pos for pos in directions
                if 0 <= pos[0] < self.grid_width and 0 <= pos[1] < self.grid_height 
                and current_map[pos[1], pos[0]] != WALL
            ]
            
            # Remove old_position from occupied temporarily to allow staying in place
            # but exclude other occupied positions (other crowds' old positions and newly assigned positions)
            available_positions = [
                pos for pos in adjacent_positions
                if pos not in (occupied_positions - {old_position})
            ]
            
            if available_positions:
                new_position = random.choice(available_positions)
            elif adjacent_positions:
                new_position = old_position
            else:
                new_position = old_position
            
            # we keep the same crowd level when moving to new position
            new_crowd_positions[new_position] = crowd_level
            occupied_positions.discard(old_position)
            occupied_positions.add(new_position)
        
        self.current_state['crowd_positions'] = new_crowd_positions


    def try_toggle_layer(self):
        x, y = self.current_state["position"]
        current_map = (
            self.surface_map if self.current_state["layer"] == 0 else self.tunnel_map
        )
        cell_value = current_map[y, x]

        if cell_value in BUILDINGS.values():
            if cell_value in self.tunnel_building_codes:
                # Toggle layer
                new_layer = 1 - self.current_state["layer"]
                self.current_state["layer"] = new_layer
                self.layer = new_layer
                return "Toggled layer successfully.", 0
            else:
                return "Building has no tunnel access.", self.rewards.get(
                    "invalid_action", -20
                )
        else:
            return "Not in a building.", self.rewards.get("invalid_action", -20)

    def play_turn(self, action):
        if action in range(0, 8):  # Movement actions
            return self.move_player(action)
        elif action == 8:  # TOGGLE_LAYER
            return self.try_toggle_layer()
        elif action == 9:  # WAIT
            return "Waited for a turn.", 0
        else:
            return "Invalid action.", self.rewards.get("invalid_action", -20)

    def step(self, action: int):
        self.steps += 1
        result, reward = self.play_turn(action)
        terminal_status = self.is_terminal()

        done = False
        if terminal_status == "goal":
            reward += self.rewards.get("goal", 1000)
            done = True
        elif terminal_status == "truncated":
            reward += self.rewards.get("timeout", -50)
            done = True

        observation = self.get_observation()
        info = {
            "result": result,
            "action_taken": self.actions[action],
            "truncated": done and terminal_status == "truncated",
        }
        return observation, reward, done, info

    def render(self, mode="human"):
        print(f"Current state: {self.current_state}")
