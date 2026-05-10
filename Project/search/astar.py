"""
search/astar.py
A* Search with custom weighted cost function for AIDRA.

Works on BOTH:
  - Grid environments:  pos = (row, col)
  - Road environments:  pos = node_id (int)

Cost function:
  f(n) = g(n) + h(n)
  g(n) = env.move_cost(pos, prev_pos) + risk_weight * node_risk + hazard_penalty
  h(n) = env.manhattan(pos, goal)
"""

import heapq
from typing import Tuple, Dict, Union
import time

Pos = Union[Tuple[int,int], int]   # grid tuple OR road node_id


def _node_risk(env, pos: Pos) -> float:
    """Get risk at a node — works for both grid and road environments."""
    try:
        # Road environment: risk_map is a dict keyed by node_id
        rm = env.risk_map
        if isinstance(rm, dict):
            return float(rm.get(pos, 0.0))
        # Grid environment: risk_map is a 2D numpy array indexed by (row, col)
        r, c = pos
        return float(rm[r][c])
    except Exception:
        return 0.0


def astar(env,
          start: Pos,
          goal:  Pos,
          risk_weight:    float = 1.0,
          hazard_penalty: float = 5.0,
          mode: str = "balanced") -> Dict:
    """
    A* with custom cost function.

    mode:
      "speed"    -> risk_weight=0.1, hazard_penalty=1.0
      "safe"     -> risk_weight=3.0, hazard_penalty=15.0
      "balanced" -> risk_weight=1.0, hazard_penalty=5.0
    """
    if mode == "speed":
        risk_weight, hazard_penalty = 0.1, 1.0
        mode_desc = "SPEED mode: minimise rescue time, higher risk accepted"
    elif mode == "safe":
        risk_weight, hazard_penalty = 3.0, 15.0
        mode_desc = "SAFE mode: minimise risk exposure, longer path accepted"
    else:
        mode_desc = "BALANCED mode: trade-off between time and risk"

    t0 = time.perf_counter()

    def g_cost(pos: Pos, prev: Pos) -> float:
        base  = env.move_cost(pos, prev)
        risk  = _node_risk(env, pos)
        hazard = hazard_penalty if risk >= 8.0 else 0.0
        return base + risk * risk_weight + hazard

    # heap: (f, g, pos, path)
    g_start = 0.0
    h_start = env.manhattan(start, goal)
    heap    = [(g_start + h_start, g_start, start, [start])]
    best_g  = {start: g_start}
    nodes_expanded = 0
    max_frontier   = 1
    explored_nodes = []

    while heap:
        max_frontier = max(max_frontier, len(heap))
        f, g, pos, path = heapq.heappop(heap)
        nodes_expanded += 1
        explored_nodes.append(pos)

        if pos == goal:
            elapsed   = time.perf_counter() - t0
            cost      = env.path_cost(path)
            risk_total = env.path_risk(path)
            return {
                "algorithm":      f"A* ({mode})",
                "path":           path,
                "path_cost":      cost,
                "path_risk":      risk_total,
                "nodes_expanded": nodes_expanded,
                "frontier_size":  max_frontier,
                "runtime":        elapsed,
                "explored":       explored_nodes,
                "found":          True,
                "mode":           mode,
                "explanation": (
                    f"{mode_desc}. "
                    f"Path length: {len(path)-1} steps, cost: {cost:.2f}, "
                    f"risk: {risk_total:.2f}. "
                    f"Expanded {nodes_expanded} nodes in {elapsed*1000:.2f}ms. "
                    f"f(n)=move_cost + {risk_weight}×risk + hazard_penalty({hazard_penalty})."
                )
            }

        if g > best_g.get(pos, float('inf')):
            continue

        for nb in env.neighbors(pos):
            g_new = g + g_cost(nb, pos)
            if g_new < best_g.get(nb, float('inf')):
                best_g[nb] = g_new
                h_nb = env.manhattan(nb, goal)
                heapq.heappush(heap, (g_new + h_nb, g_new, nb, path + [nb]))

    elapsed = time.perf_counter() - t0
    return {
        "algorithm":      f"A* ({mode})",
        "path":           [],
        "path_cost":      float('inf'),
        "path_risk":      float('inf'),
        "nodes_expanded": nodes_expanded,
        "frontier_size":  max_frontier,
        "runtime":        elapsed,
        "explored":       explored_nodes,
        "found":          False,
        "mode":           mode,
        "explanation":    f"A* ({mode_desc}) found no path from {start} to {goal}."
    }


def astar_compare(env, start: Pos, goal: Pos) -> Dict:
    """Run A* in speed and safe modes; return comparison."""
    speed_result = astar(env, start, goal, mode="speed")
    safe_result  = astar(env, start, goal, mode="safe")

    if speed_result["found"] and safe_result["found"]:
        sp_len  = len(speed_result["path"])
        sa_len  = len(safe_result["path"])
        sp_risk = speed_result.get("path_risk", 0)
        sa_risk = safe_result.get("path_risk", 0)
        tradeoff = (
            f"TRADE-OFF ANALYSIS: "
            f"Speed path: {sp_len-1} steps, risk={sp_risk:.1f}. "
            f"Safe path: {sa_len-1} steps, risk={sa_risk:.1f}. "
            f"Safe path is {sa_len-sp_len} steps longer "
            f"but reduces risk by {sp_risk-sa_risk:.1f} units."
        )
    else:
        tradeoff = "Could not complete trade-off comparison (no path found)."

    return {
        "speed":             speed_result,
        "safe":              safe_result,
        "tradeoff_analysis": tradeoff
    }
