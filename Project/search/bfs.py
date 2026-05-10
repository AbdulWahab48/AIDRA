"""
search/bfs.py
Breadth-First Search for AIDRA route planning.
"""

from collections import deque
from typing import List, Tuple, Optional, Dict
import time


def bfs(env, start: Tuple[int,int], goal: Tuple[int,int]) -> Dict:
    """
    BFS: guarantees shortest path (fewest hops).
    Returns dict with path, metrics.
    """
    t0 = time.perf_counter()
    frontier = deque()
    frontier.append((start, [start]))
    visited = {start}
    nodes_expanded = 0
    max_frontier = 1
    explored_nodes = []

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        pos, path = frontier.popleft()
        nodes_expanded += 1
        explored_nodes.append(pos)

        if pos == goal:
            elapsed = time.perf_counter() - t0
            cost = env.path_cost(path)
            return {
                "algorithm": "BFS",
                "path": path,
                "path_cost": cost,
                "nodes_expanded": nodes_expanded,
                "frontier_size": max_frontier,
                "runtime": elapsed,
                "explored": explored_nodes,
                "found": True,
                "explanation": (
                    f"BFS guarantees fewest hops ({len(path)-1} steps). "
                    f"Expanded {nodes_expanded} nodes. "
                    f"Does NOT minimise risk — only path length."
                )
            }

        for nb in env.neighbors(pos):
            if nb not in visited:
                visited.add(nb)
                frontier.append((nb, path + [nb]))

    elapsed = time.perf_counter() - t0
    return {
        "algorithm": "BFS",
        "path": [],
        "path_cost": float('inf'),
        "nodes_expanded": nodes_expanded,
        "frontier_size": max_frontier,
        "runtime": elapsed,
        "explored": explored_nodes,
        "found": False,
        "explanation": "BFS found no path to goal."
    }
