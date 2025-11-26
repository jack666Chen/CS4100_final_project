import sys
import pygame
import math

from campus_env import CampusEnv, BUILDINGS, EMPTY, WALL, TUNNEL

CODE2BUILDING = {code: name for name, code in BUILDINGS.items()}

CONSOLE_WIDTH = 380
PADDING = 4

# Colors
WHITE      = (255, 255, 255)
BLACK      = (0, 0, 0)
GRAY       = (220, 220, 220)
DARK_GRAY  = (60, 60, 60)
BLUE       = (0, 120, 255)
CYAN       = (0, 200, 200)
GREEN      = (0, 200, 0)
YELLOW     = (255, 230, 0)
ORANGE     = (255, 165, 0)
RED        = (220, 60, 60)
PURPLE     = (140, 60, 200)
LIGHT_PURP = (200, 170, 255)

# The keys (action) can take
KEY2ACTION_IDX = {
    pygame.K_w: 0,  # UP
    pygame.K_s: 1,  # DOWN
    pygame.K_a: 2,  # LEFT
    pygame.K_d: 3,  # RIGHT
    pygame.K_q: 4,  # UP - L
    pygame.K_e: 5,  # UP - R
    pygame.K_z: 6,  # DOWN - L
    pygame.K_c: 7,  # Down - R

    pygame.K_SPACE: 8,  # TOGGLE_LAYER
    pygame.K_x: 9,  # WAIT
}

HELP_LINES = [
    "Controls:",
    "  Move: W/A/S/D",
    "  Move: Q/E/Z/C"
    "  Toggle layer: SPACE",
    "  Wait: X",
    "",
    "  R = reset",
    "  G = toggle grid",
]

LAYER_LABELS = {
    0: "Surface",
    1: "Tunnel",
}


class CampusGUI:
    """
      - attributes: grid_width, grid_height, surface_map, tunnel_map,
        weather_conditions, crowd_levels, goal_building, steps, max_steps
      - methods: reset(), step(action), (optionally) render()
      - observation dict with keys:
          'position' -> (x, y)
          'layer'    -> 0 or 1
          'weather'  -> index into env.weather_conditions
          'crowd'    -> index into env.crowd_levels
          'at_goal'  -> 0 or 1
          'current_building' -> name or None
    """

    def __init__(self, env: CampusEnv, cell_size=None):
        pygame.init()
        pygame.display.set_caption("Campus Route – GUI")

        self.env = env

        # dynamic cell size based on max dimension
        max_dim = max(env.grid_width, env.grid_height)
        self.cell = cell_size or max(16, min(48, int(720 / max(8, max_dim))))

        self.grid_w = self.cell * env.grid_width
        self.grid_h = self.cell * env.grid_height

        self.width = self.grid_w + CONSOLE_WIDTH
        self.height = self.grid_h
        self.screen = pygame.display.set_mode((self.width, self.height))

        self.clock = pygame.time.Clock()
        self.fps = 30
        self.show_grid = True

        self.font  = pygame.font.Font(None, 22)
        self.font2 = pygame.font.Font(None, 26)
        self.fontH = pygame.font.Font(None, 34)

        self.recent = []
        self.max_recent = 10

        self.hover_cell = None

        # state
        self.obs = self._safe_reset()

    # ------------------- env wrappers -------------------

    def _safe_reset(self):
        # Handles when env.reset() finish.
        obs, reward, done, info = self.env.reset()
        return obs
    

    def _safe_step(self, action_idx: int):

        out = self.env.step(action_idx)

        # Current intended spec: (obs, reward, done, info)
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            return obs, reward, bool(done), info

        # Fallback if they temporarily return something else
        return out, 0.0, False, {}

    # ------------------- main loop -------------------

    def run(self):
        running = True
        end_text = None

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in KEY2ACTION_IDX and not end_text:
                        a_idx = KEY2ACTION_IDX[event.key]
                        self.obs, r, done, info = self._safe_step(a_idx)
                        self._push_recent(a_idx, r, info)
                        if done:
                            end_text = "Episode finished. Press R to reset."
                    elif event.key == pygame.K_r:
                        self.obs = self._safe_reset()
                        self.recent.clear()
                        end_text = None
                    elif event.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.cell = min(64, self.cell + 2)
                        self._resize()
                    elif event.key == pygame.K_MINUS:
                        self.cell = max(8, self.cell - 2)
                        self._resize()
                    elif event.key == pygame.K_LEFTBRACKET:
                        self.fps = max(5, self.fps - 5)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        self.fps = min(120, self.fps + 5)

            # drawing
            self.screen.fill(WHITE)
            self._draw_map()
            self._draw_console(end_text)
            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()
        sys.exit()

    # ------------------- drawing helpers -------------------

    def _resize(self):
        self.grid_w = self.cell * self.env.grid_width
        self.grid_h = self.cell * self.env.grid_height
        self.width = self.grid_w + CONSOLE_WIDTH
        self.height = self.grid_h
        self.screen = pygame.display.set_mode((self.width, self.height))

    def _cell_rect(self, x, y):
        """Note: x is column, y is row (like (col, row))."""
        return pygame.Rect(x * self.cell, y * self.cell, self.cell, self.cell)

    def _draw_grid(self):
        for y in range(self.env.grid_height):
            for x in range(self.env.grid_width):
                pygame.draw.rect(self.screen, BLACK, self._cell_rect(x, y), 1)

    def _get_active_layer_grid(self):
        layer_idx = self.obs.get('layer', 0)
        if layer_idx == 1:
            return getattr(self.env, "tunnel_map", None)
        return getattr(self.env, "surface_map", None)

    def _draw_map(self):
        grid = self._get_active_layer_grid()
        if grid is None:
            # If maps not yet initialized, just draw a blank background.
            pygame.draw.rect(
                self.screen,
                GRAY,
                pygame.Rect(0, 0, self.grid_w, self.grid_h),
            )
        else:
            # draw cells based on type
            for y in range(self.env.grid_height):
                for x in range(self.env.grid_width):
                    val = grid[y, x]
                    rect = self._cell_rect(x, y)

                    if val == WALL:
                        pygame.draw.rect(self.screen, DARK_GRAY, rect)
                    elif val == EMPTY:
                        pygame.draw.rect(self.screen, GRAY, rect)
                    elif val == TUNNEL:
                        pygame.draw.rect(self.screen, LIGHT_PURP, rect)
                    elif val in BUILDINGS.values():
                        # Draw buildings. Goal building highlighted later.
                        pygame.draw.rect(self.screen, CYAN, rect)
                    else:
                        # any other numeric codes
                        pygame.draw.rect(self.screen, GRAY, rect)

        # highlight goal building cells
        self._draw_goal_cells()

        # draw agent
        self._draw_agent()
        
        # draw the crowdsr
        self._draw_crowd_cells()

        if self.show_grid:
            self._draw_grid()

        self._draw_hover_label()

    # Draw the goal cell in this case
    def _draw_goal_cells(self):
        goal_name = getattr(self.env, "goal_building", None)
        if goal_name is None:
            return
        code = BUILDINGS.get(goal_name)
        if code is None:
            return
        grid = self._get_active_layer_grid()
        if grid is None:
            return
        for y in range(self.env.grid_height):
            for x in range(self.env.grid_width):
                if grid[y, x] == code:
                    rect = self._cell_rect(x, y).inflate(-2, -2)
                    pygame.draw.rect(self.screen, YELLOW, rect, 3)

    # Draw where the agent is.
    def _draw_agent(self):
        pos = self.obs.get('position', None)
        if pos is None or len(pos) != 2:
            return
        x, y = pos
        if not (0 <= x < self.env.grid_width and 0 <= y < self.env.grid_height):
            return
        rect = self._cell_rect(x, y)
        cx, cy = rect.center
        pygame.draw.circle(self.screen, BLUE, (cx, cy), int(self.cell * 0.35))
        pygame.draw.circle(self.screen, WHITE, (cx, cy), int(self.cell * 0.35), 2)

    # Draw the output console
    def _draw_console(self, end_text):
        x0 = self.grid_w + PADDING
        y = 10

        hdr = self.fontH.render("Campus Route", True, BLUE)
        self.screen.blit(hdr, (x0, y))
        y += 40

        # Pull info out of obs/env, with safe defaults
        layer_idx = self.obs.get('layer', 0)
        layer_name = LAYER_LABELS.get(layer_idx, f"Layer {layer_idx}")

        weather_name = self.obs.get('weather')
        crowd_name = self.obs.get('crowd')


        at_goal = self.obs.get('at_goal', 0)
        at_goal_str = "YES" if at_goal else "no"

        curr_bldg = CODE2BUILDING.get(self.obs.get('current_building', None))
        if not curr_bldg:
            curr_bldg = "-"

        steps = getattr(self.env, "steps", 0)
        max_steps = getattr(self.env, "max_steps", 0)
        goal_building = getattr(self.env, "goal_building", "Snell Library")

        lines = [
            f"Grid: {self.env.grid_width}x{self.env.grid_height} | FPS: {self.fps}",
            f"Layer: {layer_name}",
            f"Weather: {weather_name}",
            f"Crowd: {crowd_name}",
            f"Goal building: {goal_building}",
            f"At goal?: {at_goal_str}",
            f"Current building: {curr_bldg}",
            f"Steps: {steps}/{max_steps}",
            "",
        ] + HELP_LINES + ["", "Recent:"]

        for s in lines:
            txt = self.font.render(s, True, BLACK)
            self.screen.blit(txt, (x0, y))
            y += 22

        # Recent events
        for r in self.recent[-10:]:
            txt = self.font.render(r, True, BLACK)
            self.screen.blit(txt, (x0, y))
            y += 18

        if end_text:
            y += 10
            txt = self.font2.render(end_text, True, RED)
            self.screen.blit(txt, (x0, y))

    # Like PA2, this is where we track recent actions with reward.
    def _push_recent(self, action_idx, reward, info):
        try:
            action_name = self.env.actions[action_idx]
        except Exception:
            action_name = str(action_idx)

        msg = f"{action_name}  R: {reward}"
        self.recent.append(msg)
        if len(self.recent) > self.max_recent:
            self.recent.pop(0)


    def _draw_crowd_cells(self):
        """Draw the crowded cell, should be red circles"""
        state = getattr(self.env, "current_state", {})
        crowd_positions = state.get("crowd_positions", [])

        for (x, y) in crowd_positions:
            # safety: ensure within bounds
            if not (0 <= x < self.env.grid_width and 0 <= y < self.env.grid_height):
                continue

            rect = self._cell_rect(x, y)
            cx, cy = rect.center
            radius = int(self.cell * 0.25)
            pygame.draw.circle(self.screen, RED, (cx, cy), radius)



    def _draw_hover_label(self):
        # Mouse position in pixels
        mx, my = pygame.mouse.get_pos()

        # Only show label if mouse is over the grid area
        if not (0 <= mx < self.grid_w and 0 <= my < self.grid_h):
            return

        # Convert pixels -> grid coordinates
        x = mx // self.cell
        y = my // self.cell

        if not (0 <= x < self.env.grid_width and 0 <= y < self.env.grid_height):
            return

        grid = self._get_active_layer_grid()
        if grid is None:
            return

        val = grid[y, x]
        name = CODE2BUILDING.get(val)
        if not name:
            return  # not a building cell

        # Render text surface
        text_surf = self.font.render(name, True, BLACK)
        text_rect = text_surf.get_rect()

        # Position tooltip near the mouse
        text_rect.topleft = (mx + 10, my - text_rect.height - 6)

        # Keep it inside the grid window
        if text_rect.right > self.grid_w:
            text_rect.right = self.grid_w - 2
        if text_rect.top < 0:
            text_rect.top = 2

        # Background box
        bg_rect = text_rect.inflate(6, 4)
        pygame.draw.rect(self.screen, YELLOW, bg_rect)
        pygame.draw.rect(self.screen, BLACK, bg_rect, 1)

        # Draw text
        self.screen.blit(text_surf, text_rect.topleft)


def main():
    # You can parameterize goal_building or other env settings here.
    env = CampusEnv(goal_building='Snell Library')

    # Once the campus env finished, this should work correctly
    gui = CampusGUI(env)
    gui.run()


if __name__ == "__main__":
    main()
