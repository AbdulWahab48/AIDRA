"""
search/dfs.py
Depth-First Search for AIDRA route planning.
"""

from typing import List, Tuple, Optional, Dict
import time


def dfs(env, start: Tuple[int,int], goal: Tuple[int,int]) -> Dict:
    """
    DFS: not guaranteed optimal; may find long paths quickly.
    Uses iterative approach with explicit stack.
    """
    t0 = time.perf_counter()
    stack = [(start, [start])]
    visited = {start}
    nodes_expanded = 0
    max_frontier = 1
    explored_nodes = []

    while stack:
        max_frontier = max(max_frontier, len(stack))
        pos, path = stack.pop()
        nodes_expanded += 1
        explored_nodes.append(pos)

        if pos == goal:
            elapsed = time.perf_counter() - t0
            cost = env.path_cost(path)
            return {
                "algorithm": "DFS",
                "path": path,
                "path_cost": cost,
                "nodes_expanded": nodes_expanded,
                "frontier_size": max_frontier,
                "runtime": elapsed,
                "explored": explored_nodes,
                "found": True,
                "explanation": (
                    f"DFS explored depth-first; path has {len(path)-1} steps. "
                    f"Expanded {nodes_expanded} nodes. "
                    f"Path may be sub-optimal — DFS does not guarantee shortest route."
                )
            }

        for nb in env.neighbors(pos):
            if nb not in visited:
                visited.add(nb)
                stack.append((nb, path + [nb]))

    elapsed = time.perf_counter() - t0
    return {
        "algorithm": "DFS",
        "path": [],
        "path_cost": float('inf'),
        "nodes_expanded": nodes_expanded,
        "frontier_size": max_frontier,
        "runtime": elapsed,
        "explored": explored_nodes,
        "found": False,
        "explanation": "DFS found no path to goal."
    }
