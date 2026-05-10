"""
visualization/pygame_vis.py
AIDRA — 12x12 Grid Disaster Response visualizer.

Renders a clean grid-style map with:
  - Light background with alternating cell colours
  - Colour-coded roads by type (primary / secondary / local)
  - Building footprint hints within cells
  - Row / column labels on the grid
  - Realistic ambulance icons with smooth movement
  - Clean info panel (dark sidebar)
  - All AI state (fuzzy, ML, replanning) shown live
"""

import math
import random
from typing import List, Tuple, Optional, Dict

try:
    import pygame
    import pygame.gfxdraw
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False

from datasets.road_network import CoordConverter

# ── Google Maps colour palette ────────────────────────────────────────────────
MAP_BG          = (242, 239, 233)   # warm off-white land
PARK_GREEN      = (200, 227, 186)   # park / green space
WATER_BLUE      = (170, 211, 223)   # water body
RESIDENTIAL     = (237, 234, 226)   # residential block fill
COMMERCIAL      = (255, 250, 235)   # commercial / markaz
BUILDING_FILL   = (218, 212, 205)   # building footprint
BUILDING_SHADOW = (200, 193, 186)   # building shadow

# Road colours (Google Maps style)
HIGHWAY_COL     = (242, 197,  66)   # motorway/highway — yellow
HIGHWAY_CASING  = (210, 162,  30)   # highway border
ARTERIAL_COL    = (255, 255, 255)   # main road — white
ARTERIAL_CASING = (190, 188, 185)   # main road border
LOCAL_COL       = (255, 255, 255)   # local street
LOCAL_CASING    = (210, 207, 203)   # local street border
BLOCKED_COL     = (200,  50,  50)   # blocked road
BLOCKED_CASING  = (140,  20,  20)

# Hazard overlays (semi-transparent)
FIRE_FILL       = (255,  80,  20)   # fire zone
FIRE_BORDER     = (220,  40,   0)
HIGHRISK_FILL   = (255, 150,   0)   # high risk zone
HIGHRISK_BORDER = (200, 110,   0)

# Entity colours
BASE_COL        = ( 34, 139,  34)   # rescue base — forest green
BASE_ICON       = (255, 255, 255)
MC_COL          = ( 30, 100, 210)   # medical centre — blue
MC_ICON         = (255, 255, 255)
VICTIM_COLS     = [(  0, 160,   0),  # minor — green
                   (255, 140,   0),  # moderate — orange
                   (210,  30,  30)]  # critical — red
VICTIM_RING     = (255, 255, 255)

AMB0_BODY       = (  0, 180, 200)   # cyan ambulance
AMB1_BODY       = (220,  80, 160)   # pink ambulance
AMB_CROSS       = (255, 255, 255)
AMB_OUTLINE     = ( 20,  20,  20)

PATH0_COL       = (  0, 120, 255)   # route line — blue (like Google nav)
PATH1_COL       = (  0, 180,  80)   # route line — green
PATH_OUTLINE    = (255, 255, 255)

# Panel
PANEL_BG        = ( 32,  33,  36)   # dark sidebar
PANEL_BORDER    = ( 60,  62,  68)
HEADER_BG       = ( 26,  27,  30)
TEXT_MAIN       = (232, 232, 232)
TEXT_DIM        = (140, 140, 148)
TEXT_GREEN      = ( 52, 199, 100)
TEXT_RED        = (255,  80,  80)
TEXT_ORANGE     = (255, 160,  50)
TEXT_BLUE       = ( 80, 160, 255)
TEXT_YELLOW     = (255, 210,  60)
TEXT_PURPLE     = (180, 120, 255)
SECTION_HEAD    = (100, 160, 255)

MAP_W   = 740
MAP_H   = 640
PANEL_W = 330
HEADER_H = 52
BTN_H    = 28


class AIDRAVisualizer:
    """
    Grid-style AIDRA visualizer for the 12x12 disaster map.
    Draws a clean procedural grid background with road overlays,
    hazard glows, ambulance animations, and a dark sidebar panel.
    """

    def __init__(self, env, algorithm: str = "A*"):
        self.env           = env
        self.algorithm     = algorithm
        self.selected_algo = algorithm
        self.paused        = False
        self.info_lines:   List[str] = []
        self.events_log:   List[str] = []
        self.fuzzy_output: Optional[Dict] = None
        self.ml_output:    Optional[Dict] = None
        self._map_surface: Optional["pygame.Surface"] = None
        self._zone_surface: Optional["pygame.Surface"] = None
        self.conv: Optional[CoordConverter] = None

        self.WIDTH  = MAP_W + PANEL_W
        self.HEIGHT = MAP_H + HEADER_H + BTN_H + 2

    # ------------------------------------------------------------------ #
    #  INIT                                                                #
    # ------------------------------------------------------------------ #
    def init(self):
        if not PYGAME_OK:
            raise ImportError("pygame not installed — run: pip install pygame")
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("AIDRA — 12x12 Grid Disaster Response")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont("Segoe UI",    17, bold=True)
        self.font_head  = pygame.font.SysFont("Segoe UI",    13, bold=True)
        self.font_body  = pygame.font.SysFont("Segoe UI",    12)
        self.font_small = pygame.font.SysFont("Segoe UI",    10)
        self.font_road  = pygame.font.SysFont("Arial",       10)
        self.font_badge = pygame.font.SysFont("Arial",        9, bold=True)

        self.conv = CoordConverter(self.env.G, MAP_W, MAP_H, margin=45)

        # Pre-render static map layers
        self._build_static_map()

    # ------------------------------------------------------------------ #
    #  COORDINATE HELPER                                                   #
    # ------------------------------------------------------------------ #
    def _nxy(self, node_id) -> Tuple[int, int]:
        """Screen pixel for a road node (offset by header)."""
        x, y = self.conv.node_screen(self.env.G, node_id)
        return x, y + HEADER_H

    # ------------------------------------------------------------------ #
    #  STATIC MAP BUILDER (pre-rendered once)                              #
    # ------------------------------------------------------------------ #
    def _build_static_map(self):
        """
        Render a clean grid background for the 12x12 disaster map.
        Each cell is filled with a subtle alternating colour to help
        distinguish blocks at a glance.  Major grid lines (every 3 cells)
        are drawn slightly bolder to reflect arterial roads.
        """
        s = pygame.Surface((MAP_W, MAP_H))
        s.fill(MAP_BG)

        G  = self.env.G
        cv = self.conv

        GRID_ROWS = 12
        GRID_COLS = 12

        def nxy(r, c):
            # normalised (y, x) to screen pixel — no header offset on static surface
            y_n = r / (GRID_ROWS - 1)
            x_n = c / (GRID_COLS - 1)
            return cv.to_screen(y_n, x_n)

        # ── CELL FILLS ───────────────────────────────────────────────────
        # Draw each grid cell as a filled rectangle with alternating tones.
        rng = random.Random(42)
        for r in range(GRID_ROWS - 1):
            for c in range(GRID_COLS - 1):
                x0, y0 = nxy(r,     c)
                x1, y1 = nxy(r + 1, c + 1)
                is_major = (r % 3 == 0 or c % 3 == 0)
                if is_major:
                    cell_col = (230, 228, 220)   # slightly darker — arterial block
                else:
                    v = rng.random()
                    if v < 0.08:
                        cell_col = COMMERCIAL      # occasional commercial cell
                    elif v < 0.15:
                        cell_col = PARK_GREEN      # occasional park cell
                    else:
                        cell_col = RESIDENTIAL
                rect = (x0, y0, x1 - x0, y1 - y0)
                pygame.draw.rect(s, cell_col, rect)

        # ── SUBTLE BUILDING FOOTPRINTS ───────────────────────────────────
        # Draw small building shadows inside non-arterial cells.
        rng2 = random.Random(7)
        for r in range(GRID_ROWS - 1):
            for c in range(GRID_COLS - 1):
                if r % 3 == 0 or c % 3 == 0:
                    continue  # skip major-road corridors
                x0, y0 = nxy(r,     c)
                x1, y1 = nxy(r + 1, c + 1)
                cw, ch  = x1 - x0, y1 - y0
                if cw < 8 or ch < 8:
                    continue
                # 1-3 small block footprints per cell
                count = rng2.randint(1, 2)
                for _ in range(count):
                    bw = rng2.randint(int(cw * 0.15), int(cw * 0.35))
                    bh = rng2.randint(int(ch * 0.15), int(ch * 0.35))
                    bx = x0 + rng2.randint(int(cw * 0.1), int(cw * 0.55))
                    by = y0 + rng2.randint(int(ch * 0.1), int(ch * 0.55))
                    pygame.draw.rect(s, BUILDING_SHADOW, (bx + 1, by + 1, bw, bh))
                    pygame.draw.rect(s, BUILDING_FILL,   (bx,     by,     bw, bh))

        # ── GRID LINES (faint, behind roads) ────────────────────────────
        for r in range(GRID_ROWS):
            x0, y0 = nxy(r, 0)
            x1, y1 = nxy(r, GRID_COLS - 1)
            col = (200, 196, 190) if r % 3 == 0 else (218, 215, 210)
            pygame.draw.line(s, col, (x0, y0), (x1, y1), 1)
        for c in range(GRID_COLS):
            x0, y0 = nxy(0,              c)
            x1, y1 = nxy(GRID_ROWS - 1, c)
            col = (200, 196, 190) if c % 3 == 0 else (218, 215, 210)
            pygame.draw.line(s, col, (x0, y0), (x1, y1), 1)

        # ── ROW / COLUMN LABELS ──────────────────────────────────────────
        font = pygame.font.SysFont("Arial", 9)
        for r in range(GRID_ROWS):
            x, y = nxy(r, 0)
            lbl = font.render(str(r), True, (160, 155, 148))
            s.blit(lbl, (x - lbl.get_width() - 3, y - lbl.get_height() // 2))
        for c in range(GRID_COLS):
            x, y = nxy(0, c)
            lbl = font.render(str(c), True, (160, 155, 148))
            s.blit(lbl, (x - lbl.get_width() // 2, y - lbl.get_height() - 3))

        self._map_surface = s

    # ------------------------------------------------------------------ #
    #  DRAW — STATIC BACKGROUND                                            #
    # ------------------------------------------------------------------ #
    def _draw_map_bg(self):
        if self._map_surface:
            self.screen.blit(self._map_surface, (0, HEADER_H))
        else:
            pygame.draw.rect(self.screen, MAP_BG, (0, HEADER_H, MAP_W, MAP_H))

        # Map attribution (bottom left, like real maps)
        attr = self.font_small.render(
            "12x12 Grid Map  —  AIDRA Disaster Response Simulation", True, (160,155,148))
        self.screen.blit(attr, (6, HEADER_H + MAP_H - 14))

    # ------------------------------------------------------------------ #
    #  DRAW — ROADS                                                        #
    # ------------------------------------------------------------------ #
    def _road_style(self, name: str, length: float, risk: float, blocked: bool,
                    highway: str = "residential"):
        """Return (casing_color, fill_color, casing_width, fill_width)."""
        if blocked:
            return BLOCKED_CASING, BLOCKED_COL, 7, 4
        if risk >= 8.0:
            return FIRE_BORDER, FIRE_FILL, 8, 5
        if risk >= 4.0:
            return HIGHRISK_BORDER, HIGHRISK_FILL, 7, 4
        hw = str(highway).lower()
        if 'primary' in hw or 'trunk' in hw or 'motorway' in hw:
            return HIGHWAY_CASING, HIGHWAY_COL, 10, 6
        if 'secondary' in hw or 'tertiary' in hw:
            return ARTERIAL_CASING, ARTERIAL_COL, 7, 4
        return LOCAL_CASING, LOCAL_COL, 5, 3

    def draw_roads(self):
        G = self.env.G
        # Two-pass: casings first (so fills sit on top cleanly)
        drawn = set()
        road_list = []
        for u, v, k, data in G.edges(data=True, keys=True):
            key = (min(u,v), max(u,v))
            if key in drawn:
                continue
            drawn.add(key)
            x1, y1 = self._nxy(u)
            x2, y2 = self._nxy(v)
            blocked = data.get('blocked', False)
            risk    = float(data.get('risk', 0))
            name    = data.get('name', '')
            length  = float(data.get('length', 50))
            highway = data.get('highway', 'residential')
            casing_col, fill_col, cw, fw = self._road_style(
                name, length, risk, blocked, highway)
            road_list.append((x1, y1, x2, y2, casing_col, fill_col, cw, fw,
                               blocked, name, u, v, highway))

        # Pass 1 — casings
        for x1, y1, x2, y2, cc, fc, cw, fw, blocked, name, u, v, hw in road_list:
            pygame.draw.line(self.screen, cc, (x1,y1), (x2,y2), cw)

        # Pass 2 — fills
        for x1, y1, x2, y2, cc, fc, cw, fw, blocked, name, u, v, hw in road_list:
            pygame.draw.line(self.screen, fc, (x1,y1), (x2,y2), fw)

            # Blocked: draw red hatch marks
            if blocked:
                mx, my = (x1+x2)//2, (y1+y2)//2
                for dx in [-6, 0, 6]:
                    pygame.draw.line(self.screen, (220,30,30),
                                     (mx+dx-5, my-5), (mx+dx+5, my+5), 2)

        # Pass 3 — road name labels on longer segments
        drawn_labels = set()
        for x1, y1, x2, y2, cc, fc, cw, fw, blocked, name, u, v, hw in road_list:
            length = math.hypot(x2-x1, y2-y1)
            if length < 60 or not name or name in drawn_labels or blocked:
                continue
            drawn_labels.add(name)
            mx, my = (x1+x2)//2, (y1+y2)//2
            # Only label if not overlapping map edge
            if mx < 20 or mx > MAP_W-20 or my < 20 or my > MAP_H-20:
                continue
            angle = math.degrees(math.atan2(y2-y1, x2-x1))
            if angle > 90:  angle -= 180
            if angle < -90: angle += 180
            surf = self.font_road.render(str(name), True, (100, 96, 90))
            rotated = pygame.transform.rotate(surf, -angle)
            self.screen.blit(rotated,
                             (mx - rotated.get_width()//2,
                              my - rotated.get_height()//2))

    # ------------------------------------------------------------------ #
    #  DRAW — HAZARD OVERLAYS                                              #
    # ------------------------------------------------------------------ #
    def draw_hazards(self):
        """Draw semi-transparent fire/risk zone overlays on nodes."""
        t = (self.env.step_count % 30) / 30.0
        pulse = abs(math.sin(t * math.pi))

        for node, data in self.env.G.nodes(data=True):
            risk = float(data.get('risk', 0))
            if risk < 4.0:
                continue
            x, y = self._nxy(node)
            radius = int(18 + 4 * pulse) if risk >= 8.0 else 14

            # Glow surface
            glow = pygame.Surface((radius*2+4, radius*2+4), pygame.SRCALPHA)
            if risk >= 8.0:
                alpha = int(100 + 60 * pulse)
                pygame.draw.circle(glow, (*FIRE_FILL, alpha),
                                   (radius+2, radius+2), radius)
                pygame.draw.circle(glow, (*FIRE_FILL, 200),
                                   (radius+2, radius+2), radius//2)
            else:
                pygame.draw.circle(glow, (*HIGHRISK_FILL, 80),
                                   (radius+2, radius+2), radius)
            self.screen.blit(glow, (x-radius-2, y-radius-2))

    # ------------------------------------------------------------------ #
    #  DRAW — ROUTE LINES                                                  #
    # ------------------------------------------------------------------ #
    def draw_paths(self):
        """Draw navigation-style route lines for each ambulance."""
        path_colors = [PATH0_COL, PATH1_COL]
        for amb in self.env.ambulances:
            remaining = amb.route[amb.route_step:] if amb.route else []
            if len(remaining) < 2:
                continue
            pts = [self._nxy(n) for n in remaining]
            color = path_colors[amb.aid % 2]

            # White outline (like Google Maps nav line)
            pygame.draw.lines(self.screen, PATH_OUTLINE, False, pts, 7)
            # Colored fill
            pygame.draw.lines(self.screen, color, False, pts, 4)

            # Animated direction arrow at midpoint
            if len(pts) >= 2:
                mid_i = len(pts) // 2
                ax, ay = pts[mid_i]
                bx, by = pts[min(mid_i+1, len(pts)-1)]
                angle  = math.atan2(by-ay, bx-ax)
                aw = 8
                tip    = (int(ax + math.cos(angle)*aw),
                          int(ay + math.sin(angle)*aw))
                left   = (int(ax + math.cos(angle+2.4)*aw*0.6),
                          int(ay + math.sin(angle+2.4)*aw*0.6))
                right  = (int(ax + math.cos(angle-2.4)*aw*0.6),
                          int(ay + math.sin(angle-2.4)*aw*0.6))
                pygame.draw.polygon(self.screen, color, [tip, left, right])

            # Destination pin
            pygame.draw.circle(self.screen, color, pts[-1], 7)
            pygame.draw.circle(self.screen, WHITE := (255,255,255), pts[-1], 7, 2)

    # ------------------------------------------------------------------ #
    #  DRAW — ENTITIES                                                     #
    # ------------------------------------------------------------------ #
    def draw_base(self):
        x, y = self._nxy(self.env.base_pos)
        # Drop shadow
        pygame.draw.circle(self.screen, (180,175,168), (x+2, y+2), 14)
        # Base circle
        pygame.draw.circle(self.screen, BASE_COL, (x, y), 14)
        pygame.draw.circle(self.screen, (255,255,255), (x, y), 14, 2)
        # House icon
        pts = [(x,y-8),(x+7,y-2),(x+7,y+7),(x-7,y+7),(x-7,y-2)]
        pygame.draw.polygon(self.screen, (255,255,255), pts, 2)
        surf = self.font_badge.render("BASE", True, (255,255,255))
        self.screen.blit(surf, (x-surf.get_width()//2, y+10))

    def draw_medical_centers(self):
        for i, mc in enumerate(self.env.medical_centers):
            x, y = self._nxy(mc)
            # Drop shadow
            pygame.draw.circle(self.screen, (170,165,160), (x+2, y+2), 14)
            pygame.draw.circle(self.screen, MC_COL, (x, y), 14)
            pygame.draw.circle(self.screen, (255,255,255), (x, y), 14, 2)
            # Red cross icon
            pygame.draw.rect(self.screen, (220,40,40), (x-2, y-8, 4, 16))
            pygame.draw.rect(self.screen, (220,40,40), (x-8, y-2, 16, 4))
            surf = self.font_badge.render(f"H{i+1}", True, (255,255,255))
            self.screen.blit(surf, (x-surf.get_width()//2, y+10))

    def draw_victims(self):
        for v in self.env.victims:
            if v.rescued:
                continue
            x, y = self._nxy(v.pos)
            color = VICTIM_COLS[v.severity]

            # Map pin shape (teardrop)
            pygame.draw.circle(self.screen, (80,80,80), (x+2, y-8+2), 10)  # shadow
            pygame.draw.circle(self.screen, color, (x, y-8), 10)
            pygame.draw.circle(self.screen, (255,255,255), (x, y-8), 10, 2)
            # Pin point
            pts = [(x-5,y-2),(x+5,y-2),(x,y+6)]
            pygame.draw.polygon(self.screen, (80,80,80), [(p[0]+2,p[1]+2) for p in pts])
            pygame.draw.polygon(self.screen, color, pts)
            # Victim ID
            surf = self.font_badge.render(str(v.vid), True, (255,255,255))
            self.screen.blit(surf, (x-surf.get_width()//2, y-13))

            # Severity ring pulse
            sev_labels = ["", "!", "!!"]
            if v.severity > 0:
                t = (self.env.step_count % 20)/20.0
                r = int(13 + 3*abs(math.sin(t*math.pi)))
                ring = pygame.Surface((r*2+4,r*2+4), pygame.SRCALPHA)
                pygame.draw.circle(ring, (*color, 100), (r+2,r+2), r)
                self.screen.blit(ring, (x-r-2, y-8-r-2))

    def _amb_screen_pos(self, amb) -> Tuple[int, int]:
        """Smooth interpolated ambulance position."""
        amb.interp_t = min(1.0, amb.interp_t + 0.3)
        if (amb.interp_t < 1.0
                and amb.interp_from >= 0
                and amb.interp_to >= 0
                and amb.interp_from in self.env.G.nodes
                and amb.interp_to   in self.env.G.nodes):
            x1, y1 = self._nxy(amb.interp_from)
            x2, y2 = self._nxy(amb.interp_to)
            t = amb.interp_t
            return int(x1 + (x2-x1)*t), int(y1 + (y2-y1)*t)
        return self._nxy(amb.pos)

    def draw_ambulances(self):
        amb_colors = [AMB0_BODY, AMB1_BODY]
        for amb in self.env.ambulances:
            x, y = self._amb_screen_pos(amb)
            color = amb_colors[amb.aid]

            # Direction of travel
            angle = 0.0
            if amb.route and amb.route_step < len(amb.route):
                nx_, ny_ = self._nxy(amb.route[min(amb.route_step, len(amb.route)-1)])
                angle = math.atan2(ny_-y, nx_-x)

            # Ambulance body (rounded rectangle rotated)
            W, H = 22, 14
            pts_local = [(-W//2,-H//2),(W//2,-H//2),(W//2,H//2),(-W//2,H//2)]
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            pts = [(int(x + px*cos_a - py*sin_a),
                    int(y + px*sin_a + py*cos_a))
                   for px, py in pts_local]

            # Shadow
            shadow_pts = [(px+2, py+2) for px, py in pts]
            pygame.draw.polygon(self.screen, (100,100,100), shadow_pts)
            # Body
            pygame.draw.polygon(self.screen, color, pts)
            pygame.draw.polygon(self.screen, AMB_OUTLINE, pts, 2)

            # Red cross on body
            pygame.draw.rect(self.screen, (220,30,30), (x-2, y-4, 4, 8))
            pygame.draw.rect(self.screen, (220,30,30), (x-5, y-1, 10, 3))

            # Label above
            label_col = (0,0,0)
            surf = self.font_badge.render(f"A{amb.aid}", True, label_col)
            bg = pygame.Surface((surf.get_width()+4, surf.get_height()+2), pygame.SRCALPHA)
            bg.fill((*color, 220))
            self.screen.blit(bg, (x-bg.get_width()//2, y-H-14))
            self.screen.blit(surf, (x-surf.get_width()//2, y-H-13))

            # Passenger badge
            if amb.passengers:
                badge_surf = self.font_badge.render(str(len(amb.passengers)), True, (255,255,255))
                bx, by = x+10, y-10
                pygame.draw.circle(self.screen, (200,30,30), (bx,by), 8)
                self.screen.blit(badge_surf, (bx-badge_surf.get_width()//2,
                                               by-badge_surf.get_height()//2))

    # ------------------------------------------------------------------ #
    #  DRAW — HEADER                                                       #
    # ------------------------------------------------------------------ #
    def draw_header(self):
        pygame.draw.rect(self.screen, HEADER_BG, (0, 0, self.WIDTH, HEADER_H))
        # Left: title
        title = self.font_title.render(
            "AIDRA  ·  12x12 Grid Disaster Response", True, (220, 225, 235))
        self.screen.blit(title, (10, 8))
        # Subtitle
        sub_parts = [
            f"Step {self.env.step_count}",
            f"Saved {self.env.victims_saved}/5",
            f"Replannings {self.env.replanning_triggers}",
            f"Seed {self.env.seed}",
            "[SPACE] Pause   [R] Reset   [Q] Quit",
        ]
        sub = self.font_body.render("  ·  ".join(sub_parts), True, (140,145,160))
        self.screen.blit(sub, (10, HEADER_H-18))
        # Status indicator
        status_col = (80,200,80) if not self.paused else (255,160,40)
        status_txt = "● LIVE" if not self.paused else "⏸ PAUSED"
        stat_surf = self.font_body.render(status_txt, True, status_col)
        self.screen.blit(stat_surf, (MAP_W - stat_surf.get_width() - 10, 16))

    # ------------------------------------------------------------------ #
    #  DRAW — SIDEBAR PANEL                                                #
    # ------------------------------------------------------------------ #
    def draw_panel(self):
        px = MAP_W + 1
        pw = PANEL_W - 8

        # Background
        pygame.draw.rect(self.screen, PANEL_BG, (MAP_W, 0, PANEL_W, self.HEIGHT))
        pygame.draw.line(self.screen, PANEL_BORDER, (MAP_W, 0), (MAP_W, self.HEIGHT), 1)

        y = HEADER_H + 8

        def section(title, color=SECTION_HEAD):
            nonlocal y
            surf = self.font_head.render(title, True, color)
            self.screen.blit(surf, (px+4, y))
            y += 16
            pygame.draw.line(self.screen, (55,58,65), (px+4, y-2), (px+pw, y-2), 1)

        def row(text, color=TEXT_MAIN, indent=0, bold=False):
            nonlocal y
            f = self.font_head if bold else self.font_body
            surf = f.render(text, True, color)
            self.screen.blit(surf, (px+6+indent, y))
            y += 15

        def spacer(h=4):
            nonlocal y; y += h

        # ── VICTIMS ──────────────────────────────────────────────────────
        section("VICTIMS")
        sev_lbl = ["Minor", "Moderate", "Critical"]
        sev_col = [TEXT_GREEN, TEXT_ORANGE, TEXT_RED]
        for v in self.env.victims:
            if v.rescued:
                row(f"V{v.vid} [{sev_lbl[v.severity]}]  ✓ RESCUED",
                    color=(80,180,80))
            else:
                row(f"V{v.vid} [{sev_lbl[v.severity]}]  cond={v.condition:.2f}",
                    color=sev_col[v.severity])
        spacer()

        # ── AMBULANCES ───────────────────────────────────────────────────
        section("AMBULANCES")
        amb_cols = [(0,210,220), (230,100,180)]
        for amb in self.env.ambulances:
            col = amb_cols[amb.aid]
            row(f"Amb {amb.aid}  trips={amb.trips}  risk={amb.risk_exposure:.1f}",
                color=col, bold=True)
            row(f"passengers={amb.passengers}  {amb.last_algo}",
                color=TEXT_DIM, indent=8)
        spacer()

        # ── RESOURCES ────────────────────────────────────────────────────
        section("RESOURCES")
        row(f"Medical Kits:    {self.env.medical_kits}/10")
        row(f"Rescue Team:     {'BUSY' if self.env.rescue_team_busy else 'Available'}")
        row(f"Blocked Edges:   {len(self.env._blocked_edges)}")
        row(f"Replannings:     {self.env.replanning_triggers}")
        spacer()

        # ── SEARCH INFO ──────────────────────────────────────────────────
        if self.info_lines:
            section("SEARCH / MOVEMENT")
            for ln in self.info_lines[-4:]:
                row(ln, color=(160,210,160))
            spacer()

        # ── FUZZY ASSESSMENT ─────────────────────────────────────────────
        if self.fuzzy_output:
            fo = self.fuzzy_output
            section("FUZZY ASSESSMENT", color=(255,200,80))
            rd = fo.get('route_danger', 0)
            ru = fo.get('rescue_urgency', 0)
            or_ = fo.get('overall_risk', 0)
            row(f"Route Danger:    {rd:.1f}/10",
                color=TEXT_RED if fo.get('avoid_route') else TEXT_GREEN)
            row(f"Rescue Urgency:  {ru:.1f}/10",
                color=TEXT_ORANGE if fo.get('prioritize_victim') else TEXT_DIM)
            row(f"Overall Risk:    {or_:.1f}/10")
            if fo.get('replan_recommended'):
                row("⚡ REPLANNING RECOMMENDED", color=TEXT_YELLOW, bold=True)
            spacer()

        # ── ML ASSESSMENT ────────────────────────────────────────────────
        if self.ml_output:
            mo = self.ml_output
            section("ML ASSESSMENT", color=(180,130,255))
            lbl = mo.get('risk_label','?')
            col = (TEXT_RED if lbl=='HIGH' else
                   TEXT_ORANGE if lbl=='MEDIUM' else TEXT_GREEN)
            row(f"Risk Class:      {lbl}", color=col)
            row(f"Survival Prob:   {mo.get('survival_probability',0):.2f}")
            row(mo.get('priority','?')[:38], color=TEXT_ORANGE)
            spacer()

        # ── RECENT EVENTS ────────────────────────────────────────────────
        section("RECENT EVENTS", color=(255,130,80))
        for ev in self.events_log[-5:]:
            col = (TEXT_RED if 'FIRE' in ev
                   else TEXT_ORANGE if 'AFTERSHOCK' in ev
                   else TEXT_BLUE if 'BLOCKED' in ev
                   else TEXT_DIM)
            row(ev[:40], color=col)
        spacer()

        # ── MAP LEGEND ───────────────────────────────────────────────────
        ly = self.HEIGHT - 175
        pygame.draw.line(self.screen, (55,58,65), (px+4, ly-4), (px+pw, ly-4), 1)
        legend_surf = self.font_head.render("LEGEND", True, TEXT_DIM)
        self.screen.blit(legend_surf, (px+4, ly)); ly += 17

        legend_items = [
            (HIGHWAY_COL,      "Highway / Major Road"),
            (ARTERIAL_COL,     "Arterial Road"),
            (FIRE_FILL,        "Fire Zone (impassable)"),
            (HIGHRISK_FILL,    "High Risk Zone"),
            (BLOCKED_COL,      "Blocked Road"),
            (BASE_COL,         "Rescue Base"),
            (MC_COL,           "Medical Centre"),
            (PATH0_COL,        "Amb 0 Route"),
            (PATH1_COL,        "Amb 1 Route"),
        ]
        for color, label in legend_items:
            pygame.draw.rect(self.screen, color, (px+6, ly+1, 14, 10), border_radius=2)
            pygame.draw.rect(self.screen, (90,92,98), (px+6, ly+1, 14, 10), 1, border_radius=2)
            surf = self.font_small.render(label, True, TEXT_DIM)
            self.screen.blit(surf, (px+24, ly)); ly += 14

    # ------------------------------------------------------------------ #
    #  DRAW — BUTTONS                                                      #
    # ------------------------------------------------------------------ #
    def draw_buttons(self):
        by = self.HEIGHT - BTN_H - 2
        buttons = [
            ("▶  START",  (10,  by, 95, BTN_H-2), ( 34,150, 80), (200,255,200)),
            ("⏸  PAUSE",  (112, by, 95, BTN_H-2), (180,120,  0), (255,230,150)),
            ("↺  RESET",  (214, by, 95, BTN_H-2), (160, 30, 30), (255,180,180)),
        ]
        for label, rect, bg, fg in buttons:
            pygame.draw.rect(self.screen, bg, rect, border_radius=5)
            surf = self.font_body.render(label, True, fg)
            self.screen.blit(surf,
                (rect[0]+rect[2]//2-surf.get_width()//2,
                 rect[1]+rect[3]//2-surf.get_height()//2))

    # ------------------------------------------------------------------ #
    #  MAIN RENDER                                                         #
    # ------------------------------------------------------------------ #
    def render(self):
        # 1. Map background (land, parks, blocks)
        self._draw_map_bg()
        # 2. Hazard overlays (glow effects)
        self.draw_hazards()
        # 3. Roads (casings + fills + labels)
        self.draw_roads()
        # 4. Route lines
        self.draw_paths()
        # 5. Base, hospitals, victims
        self.draw_base()
        self.draw_medical_centers()
        self.draw_victims()
        # 6. Ambulances (on top of everything)
        self.draw_ambulances()
        # 7. UI chrome
        self.draw_header()
        self.draw_panel()
        self.draw_buttons()
        pygame.display.flip()

    # ------------------------------------------------------------------ #
    #  EVENT HANDLING                                                      #
    # ------------------------------------------------------------------ #
    def handle_events(self) -> str:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:     return 'quit'
                if event.key == pygame.K_r:     return 'reset'
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    return 'pause'
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, my = event.pos
                by = self.HEIGHT - BTN_H - 2
                if 10  <= x <= 105 and by <= my <= by+BTN_H:
                    self.paused = False;            return 'start'
                if 112 <= x <= 207 and by <= my <= by+BTN_H:
                    self.paused = not self.paused;  return 'pause'
                if 214 <= x <= 309 and by <= my <= by+BTN_H:
                    return 'reset'
        return ''

    # ------------------------------------------------------------------ #
    #  SETTERS                                                             #
    # ------------------------------------------------------------------ #
    def set_search_info(self, info: List[str]):   self.info_lines  = info
    def set_path(self, path, explored=None):       pass   # handled via env.ambulances
    def add_event(self, ev: str):
        self.events_log.append(ev)
        if len(self.events_log) > 20:
            self.events_log.pop(0)
    def set_fuzzy(self, output: Dict):  self.fuzzy_output = output
    def set_ml(self,   output: Dict):  self.ml_output    = output
    def tick(self, fps: int = 30):     self.clock.tick(fps)
    def quit(self):                    pygame.quit()
