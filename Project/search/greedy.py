"""
search/greedy.py
Greedy Best-First Search for AIDRA route planning.
"""

import heapq
from typing import Tuple, Dict
import time


def greedy(env, start: Tuple[int,int], goal: Tuple[int,int]) -> Dict:
    """
    Greedy Best-First: expands node closest to goal by heuristic.
    Fast but not optimal.
    """
    t0 = time.perf_counter()

    # heap: (h_value, pos, path)
    h_start = env.manhattan(start, goal)
    heap = [(h_start, start, [start])]
    visited = {start}
    nodes_expanded = 0
    max_frontier = 1
    explored_nodes = []

    while heap:
        max_frontier = max(max_frontier, len(heap))
        h, pos, path = heapq.heappop(heap)
        nodes_expanded += 1
        explored_nodes.append(pos)

        if pos == goal:
            elapsed = time.perf_counter() - t0
            cost = env.path_cost(path)
            return {
                "algorithm": "Greedy Best-First",
                "path": path,
                "path_cost": cost,
                "nodes_expanded": nodes_expanded,
                "frontier_size": max_frontier,
                "runtime": elapsed,
                "explored": explored_nodes,
                "found": True,
                "explanation": (
                    f"Greedy BFS used Manhattan distance heuristic; {len(path)-1} steps. "
                    f"Expanded only {nodes_expanded} nodes — very fast. "
                    f"Trade-off: ignores risk and may not give optimal cost."
                )
            }

        for nb in env.neighbors(pos):
            if nb not in visited:
                visited.add(nb)
                h_nb = env.manhattan(nb, goal)
                heapq.heappush(heap, (h_nb, nb, path + [nb]))

    elapsed = time.perf_counter() - t0
    return {
        "algorithm": "Greedy Best-First",
        "path": [],
        "path_cost": float('inf'),
        "nodes_expanded": nodes_expanded,
        "frontier_size": max_frontier,
        "runtime": elapsed,
        "explored": explored_nodes,
        "found": False,
        "explanation": "Greedy BFS found no path to goal."
    }
