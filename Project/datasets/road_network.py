"""
datasets/road_network.py
12x12 grid road network — drop-in replacement for the F-8 Islamabad map.

Grid layout:
  - 144 nodes arranged in a 12x12 grid (row 0 = north, row 11 = south)
  - Node ID = row * 12 + col  (e.g. node 0 = top-left, node 143 = bottom-right)
  - Bidirectional edges: horizontal (E-W) and vertical (N-S)
  - Rows/cols divisible by 3 are "primary" (arterials), even = "secondary", else "residential"
  - Four intersections pre-assigned fire/high-risk hazards for reproducibility

Coordinate system:
  x = col / 11   (0.0 .. 1.0, west -> east)
  y = row / 11   (0.0 .. 1.0, north -> south)

CoordConverter maps these normalised coords directly to screen pixels
with no lat/lon rotation — the grid renders axis-aligned.

All AI logic (A*, BFS, DFS, greedy, local search, CSP, fuzzy, ML, replanner)
is fully preserved — they interact only with node IDs, edge attributes, and
the public methods of DisasterEnvironment.
"""

import os
import math
import random
import networkx as nx
from typing import Tuple, Dict, List, Optional

GRAPHML_PATH = os.path.join(os.path.dirname(__file__), "grid12_map.graphml")

GRID_ROWS = 12
GRID_COLS = 12


# --------------------------------------------------------------------------- #
#  PUBLIC ENTRY POINT                                                           #
# --------------------------------------------------------------------------- #

def load_or_build_graph(seed: int = 42) -> nx.MultiDiGraph:
    if os.path.exists(GRAPHML_PATH):
        try:
            G = _load_graphml(GRAPHML_PATH)
            print(f"[RoadNet] Loaded cached 12x12 grid: {len(G.nodes)} nodes, {len(G.edges)} edges")
            return G
        except Exception as e:
            print(f"[RoadNet] Cache load failed ({e}), rebuilding...")

    G = _build_grid_graph(seed=seed)
    _save_graphml(G, GRAPHML_PATH)
    print(f"[RoadNet] 12x12 grid graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G


# --------------------------------------------------------------------------- #
#  GRID BUILDER                                                                 #
# --------------------------------------------------------------------------- #

def _node_id(row: int, col: int) -> int:
    return row * GRID_COLS + col


def _edge_name(r1, c1, r2, c2) -> str:
    if r1 == r2:
        if r1 % 3 == 0:
            return f"Ave {r1}"
        return f"Street H{r1}"
    else:
        if c1 % 3 == 0:
            return f"Blvd {c1}"
        return f"Street V{c1}"


def _build_grid_graph(seed: int = 42) -> nx.MultiDiGraph:
    random.seed(seed)
    G = nx.MultiDiGraph()

    # Add nodes
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            nid = _node_id(r, c)
            G.add_node(nid,
                       x=c / (GRID_COLS - 1),
                       y=r / (GRID_ROWS - 1),
                       risk=0.0,
                       name=f"({r},{c})")

    def add_edge(r1, c1, r2, c2):
        u = _node_id(r1, c1)
        v = _node_id(r2, c2)
        # Classify by the axis of travel
        if r1 == r2:
            hw = "primary" if r1 % 3 == 0 else ("secondary" if r1 % 2 == 0 else "residential")
        else:
            hw = "primary" if c1 % 3 == 0 else ("secondary" if c1 % 2 == 0 else "residential")
        name   = _edge_name(r1, c1, r2, c2)
        length = 50.0
        dw     = 6 if hw == "primary" else 4 if hw == "secondary" else 2
        for a, b in [(u, v), (v, u)]:
            G.add_edge(a, b, key=0,
                       length=length, blocked=False, risk=0.0,
                       name=name, highway=hw, draw_width=dw)

    # Horizontal edges
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS - 1):
            add_edge(r, c, r, c + 1)

    # Vertical edges
    for r in range(GRID_ROWS - 1):
        for c in range(GRID_COLS):
            add_edge(r, c, r + 1, c)

    # Fire zones (risk=10)
    for r, c in [(2, 2), (5, 8), (9, 4)]:
        nid = _node_id(r, c)
        G.nodes[nid]['risk'] = 10.0
        _propagate_edge_risk(G, nid, 10.0)

    # High-risk zones (risk=5)
    for r, c in [(1, 7), (4, 3), (7, 9), (10, 1)]:
        nid = _node_id(r, c)
        G.nodes[nid]['risk'] = 5.0
        _propagate_edge_risk(G, nid, 5.0)

    return G


def _propagate_edge_risk(G, node, risk):
    for u, v, k in list(G.out_edges(node, keys=True)):
        G[u][v][k]['risk'] = max(G[u][v][k].get('risk', 0), risk)
    for u, v, k in list(G.in_edges(node, keys=True)):
        G[u][v][k]['risk'] = max(G[u][v][k].get('risk', 0), risk)


# --------------------------------------------------------------------------- #
#  GRAPHML SAVE / LOAD                                                          #
# --------------------------------------------------------------------------- #

def _save_graphml(G, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        H = G.copy()
        for n, d in H.nodes(data=True):
            for k in list(d):
                d[k] = str(d[k])
        for u, v, k, d in H.edges(data=True, keys=True):
            for ek in list(d):
                d[ek] = str(d[ek])
        nx.write_graphml(H, path)
    except Exception as e:
        print(f"[RoadNet] Could not save: {e}")


def _load_graphml(path):
    H = nx.read_graphml(path, node_type=int)
    G = nx.MultiDiGraph()
    for n, d in H.nodes(data=True):
        G.add_node(n,
                   x=float(d.get('x', 0)),
                   y=float(d.get('y', 0)),
                   risk=float(d.get('risk', 0)),
                   name=d.get('name', ''))
    for u, v, d in H.edges(data=True):
        G.add_edge(u, v, key=0,
                   length=float(d.get('length', 50)),
                   blocked=d.get('blocked', 'False') == 'True',
                   risk=float(d.get('risk', 0)),
                   name=d.get('name', 'Road'),
                   highway=d.get('highway', 'residential'),
                   draw_width=int(float(d.get('draw_width', 2))))
    return G


# --------------------------------------------------------------------------- #
#  COORDINATE CONVERTER                                                         #
# --------------------------------------------------------------------------- #

class CoordConverter:
    """
    Maps grid node coordinates (x in [0,1], y in [0,1]) to pygame screen pixels.

    No rotation — the 12x12 grid renders axis-aligned.
    The original signature used (lat, lon) == (y, x); kept for compatibility.
    """

    MAP_ROTATION_DEG = 0.0   # kept for API compatibility; unused

    def __init__(self, G: nx.MultiDiGraph,
                 screen_w: int, screen_h: int, margin: int = 45):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.margin   = margin
        self.sw       = screen_w - 2 * margin
        self.sh       = screen_h - 2 * margin

    def to_screen(self, y_norm: float, x_norm: float) -> Tuple[int, int]:
        """
        Convert normalised (y, x) -> screen pixel.
        Signature mirrors original to_screen(lat, lon) for drop-in compatibility.
        """
        sx = int(self.margin + x_norm * self.sw)
        sy = int(self.margin + y_norm * self.sh)
        return sx, sy

    def node_screen(self, G: nx.MultiDiGraph, node_id) -> Tuple[int, int]:
        d = G.nodes[node_id]
        return self.to_screen(d['y'], d['x'])

    def nearest_node(self, G: nx.MultiDiGraph, sx: int, sy: int) -> int:
        best, best_d = None, float('inf')
        for n, d in G.nodes(data=True):
            px, py = self.to_screen(d['y'], d['x'])
            dist = (px - sx) ** 2 + (py - sy) ** 2
            if dist < best_d:
                best_d = dist
                best = n
        return best
