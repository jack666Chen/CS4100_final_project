# vis_campus.py
import sys, time, math, random
import pygame

# Import your new environment
from campus_env_ai_test_model import CampusRouteEnv   # make sure campus_env.py is alongside this file

# ------------------- Display constants -------------------
CONSOLE_WIDTH = 380
PADDING = 4

# Colors
WHITE      = (255,255,255)
BLACK      = (0,0,0)
GRAY       = (200,200,200)
DARK_GRAY  = (60,60,60)
BLUE       = (0,120,255)
CYAN       = (0,200,200)
GREEN      = (0,200,0)
YELLOW     = (255,230,0)
ORANGE     = (255,165,0)
RED        = (220,60,60)
PURPLE     = (140,60,200)

# Key bindings (10 actions)
KEY2ACTION = {
    pygame.K_w: "UP",
    pygame.K_s: "DOWN",
    pygame.K_a: "LEFT",
    pygame.K_d: "RIGHT",
    pygame.K_q: "UPLEFT",
    pygame.K_e: "UPRIGHT",
    pygame.K_z: "DOWNLEFT",
    pygame.K_c: "DOWNRIGHT",
    pygame.K_t: "ENTER_TUNNEL",
    pygame.K_y: "EXIT_TUNNEL",
}

HELP_LINES = [
    "Controls:",
    "  Move: W/A/S/D, Diagonals: Q E / Z C",
    "  Tunnel: T enter, Y exit",
    "  R = reset, G = toggle grid, F = toggle fog",
    "  +/- = cell size, [ ] = FPS",
]

# ------------------- GUI -------------------
class CampusGUI:
    def __init__(self, env: CampusRouteEnv, cell_size=None):
        pygame.init()
        pygame.display.set_caption("Campus Route – GUI")
        self.env = env

        # dynamic cell size so big maps still fit; default ~32 px
        self.cell = cell_size or max(12, min(48, int(720 / max(12, env.grid_size))))
        self.grid_w = self.cell * env.grid_size
        self.grid_h = self.cell * env.grid_size

        self.width = self.grid_w + CONSOLE_WIDTH
        self.height = self.grid_h
        self.screen = pygame.display.set_mode((self.width, self.height))

        self.clock = pygame.time.Clock()
        self.fps = 30
        self.show_grid = True
        self.show_fog = False

        self.font  = pygame.font.Font(None, 22)
        self.font2 = pygame.font.Font(None, 28)
        self.fontH = pygame.font.Font(None, 36)

        self.recent = []
        self.max_recent = 10

        # state
        self.obs, _info = self._safe_reset()

    # gym/gymnasium compatibility
    def _safe_step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, done, truncated, info = out
            done = bool(done) or bool(truncated)
        else:
            obs, reward, done, info = out
        return obs, reward, bool(done), info

    def _safe_reset(self):
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            return out  # (obs, info) - gymnasium style
        return out, {}  # (obs, reward, done, info) from legacy

    def run(self):
        running = True
        end_text = None

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in KEY2ACTION and not end_text:
                        act = KEY2ACTION[event.key]
                        self.obs, r, done, info = self._safe_step(act)
                        self._push_recent(act, r, info)
                        if done:
                            end_text = "Finished (goal or timeout). Press R to reset."
                    elif event.key == pygame.K_r:
                        self.obs, _info = self._safe_reset()
                        self.recent.clear()
                        end_text = None
                    elif event.key == pygame.K_f:
                        self.show_fog = not self.show_fog
                    elif event.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):  # +
                        self.cell = min(64, self.cell + 2); self._resize()
                    elif event.key == pygame.K_MINUS:
                        self.cell = max(8, self.cell - 2); self._resize()
                    elif event.key == pygame.K_LEFTBRACKET:
                        self.fps = max(5, self.fps - 5)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        self.fps = min(120, self.fps + 5)

            # draw
            self.screen.fill(WHITE)
            self._draw_map()
            self._draw_console(end_text)
            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()
        sys.exit()

    # -------- draw helpers --------
    def _resize(self):
        self.grid_w = self.cell * self.env.grid_size
        self.grid_h = self.cell * self.env.grid_size
        self.width  = self.grid_w + CONSOLE_WIDTH
        self.height = self.grid_h
        self.screen = pygame.display.set_mode((self.width, self.height))

    def _cell_rect(self, r, c):
        return pygame.Rect(c * self.cell, r * self.cell, self.cell, self.cell)

    def _draw_grid(self):
        for r in range(self.env.grid_size):
            for c in range(self.env.grid_size):
                pygame.draw.rect(self.screen, BLACK, self._cell_rect(r, c), 1)

    def _draw_walls(self):
        # walls are map==1
        for r in range(self.env.grid_size):
            for c in range(self.env.grid_size):
                if self.env.map[r, c] == 1:
                    pygame.draw.rect(self.screen, DARK_GRAY, self._cell_rect(r, c))

    def _draw_crowds(self):
        # crowd cells visually orange with hatch
        surf = pygame.Surface((self.cell, self.cell), pygame.SRCALPHA)
        surf.fill((255,165,0,110))
        for (r, c) in getattr(self.env, "crowd_cells", set()):
            self.screen.blit(surf, (c * self.cell, r * self.cell))
            # draw a few diagonal hatch lines
            for k in range(-self.cell, self.cell, 6):
                pygame.draw.line(self.screen, (180,90,0),
                                 (c * self.cell + k, r * self.cell),
                                 (c * self.cell + k + self.cell, r * self.cell + self.cell), 1)

    def _draw_goal(self):
        gx, gy = self.env.goal
        rect = self._cell_rect(gx, gy)
        inner = rect.inflate(-4, -4)
        pygame.draw.rect(self.screen, YELLOW, inner, border_radius=6)
        label = self.font2.render("GOAL", True, BLACK)
        self.screen.blit(label, (inner.x + 6, inner.y + 4))

    def _draw_agent(self):
        (r, c) = self.env.player_position
        rect = self._cell_rect(r, c)
        cx, cy = rect.center
        pygame.draw.circle(self.screen, BLUE, (cx, cy), int(self.cell*0.32))
        pygame.draw.circle(self.screen, WHITE, (cx, cy), int(self.cell*0.32), 2)

    def _draw_tunnels(self):
        # entrances are keys of self.env.tunnels; draw link hint to exit
        for (r, c), (er, ec) in getattr(self.env, "tunnels", {}).items():
            rect = self._cell_rect(r, c)
            cx, cy = rect.center
            pygame.draw.circle(self.screen, PURPLE, (cx, cy), int(self.cell*0.20))
            pygame.draw.circle(self.screen, WHITE,  (cx, cy), int(self.cell*0.20), 2)
            # arrow-ish line toward exit if nearby (for UX hint)
            if abs(er-r) + abs(ec-c) <= 6:
                ex = ec * self.cell + self.cell//2
                ey = er * self.cell + self.cell//2
                pygame.draw.line(self.screen, PURPLE, (cx, cy), (ex, ey), 2)

    def _draw_fog(self):
        # simple 3x3 reveal around agent (like your prior GUI)
        r, c = self.env.player_position
        fog = pygame.Surface((self.grid_w, self.grid_h), pygame.SRCALPHA)
        fog.fill((0,0,0,150))
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                rr, cc = r+dr, c+dc
                if 0 <= rr < self.env.grid_size and 0 <= cc < self.env.grid_size:
                    cell_rect = self._cell_rect(rr, cc)
                    pygame.draw.rect(fog, (0,0,0,0), cell_rect)
        self.screen.blit(fog, (0,0))

    def _draw_map(self):
        # base tiles
        self._draw_walls()
        self._draw_crowds()
        self._draw_goal()
        self._draw_tunnels()
        self._draw_agent()
        if self.show_grid:
            self._draw_grid()
        if self.show_fog:
            self._draw_fog()

    def _draw_console(self, end_text):
        x0 = self.grid_w + PADDING
        y = 10
        # Header
        hdr = self.fontH.render("Campus Route", True, BLUE)
        self.screen.blit(hdr, (x0, y)); y += 42

        # Info
        te = float(getattr(self.env, "time_elapsed", 0.0))
        steps = int(getattr(self.env, "steps", 0))
        max_steps = int(getattr(self.env, "max_steps", 0))
        pos = tuple(getattr(self.env, "player_position", (0,0)))
        goal = tuple(getattr(self.env, "goal", (0,0)))

        lines = [
            f"Grid: {self.env.grid_size}x{self.env.grid_size} | FPS: {self.fps}",
            f"Pos: {pos}  Goal: {goal}",
            f"Steps: {steps}/{max_steps}",
            f"Time elapsed: {te:.2f}",
            "",
        ] + HELP_LINES + ["", "Recent:"]

        for s in lines:
            txt = self.font.render(s, True, BLACK)
            self.screen.blit(txt, (x0, y)); y += 22

        # Recent actions/results
        for r in self.recent[-10:]:
            for chunk in self._wrap_text(r, width=CONSOLE_WIDTH-20):
                txt = self.font.render(chunk, True, BLACK)
                self.screen.blit(txt, (x0, y)); y += 18

        # End text
        if end_text:
            y += 10
            txt = self.font2.render(end_text, True, RED)
            self.screen.blit(txt, (x0, y))

    def _wrap_text(self, s, width):
        words, line = s.split(), ""
        surface = self.font
        out = []
        for w in words:
            trial = f"{line} {w}".strip()
            if surface.size(trial)[0] <= width:
                line = trial
            else:
                out.append(line)
                line = w
        if line:
            out.append(line)
        return out

    def _push_recent(self, action, reward, info):
        msg = f"{action}  R:{reward:.3f}"
        if "cost" in info:
            msg += f"  cost:{info['cost']:.2f}"
        self.recent.append(msg)
        if len(self.recent) > self.max_recent:
            self.recent.pop(0)

# ------------------- main -------------------
def main():
    # you can adjust grid_size/max_steps here while prototyping
    env = CampusRouteEnv(grid_size=20, max_steps=2000)

    # quick demo decorations (optional): a few walls, crowds, and a tunnel pair
    # walls line
    for r in range(3, 12):
        env.map[r, 8] = 1
    # crowd cells
    for r in range(10, 15):
        for c in range(11, 17):
            env.crowd_cells.add((r, c))
    # tunnel: entrance at (2,2) -> exit at (16, 5)
    env.tunnels[(2,2)] = (16,5)

    gui = CampusGUI(env)
    gui.run()

if __name__ == "__main__":
    main()
