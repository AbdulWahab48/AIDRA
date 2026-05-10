"""
search/local_search.py
Simulated Annealing for optimizing victim rescue order.
Also includes Hill Climbing for comparison.
"""

import math
import random
import copy
from typing import List, Dict, Tuple
import time


def _rescue_order_cost(order: List[int], victims, env) -> float:
    """
    Objective: minimize total weighted cost.
    Higher severity = higher priority (lower cost to rescue early).
    Factor in distance from base and victim condition.
    """
    base = env.base_pos
    cost = 0.0
    pos = base
    for i, vid in enumerate(order):
        v = victims[vid]
        dist = env.manhattan(pos, v.pos)
        # Penalty: critical victims rescued late = high cost
        urgency = (v.severity + 1) * (1 + v.condition)
        cost += dist + (i * urgency)  # later position = higher penalty for urgent
        pos = v.pos
    return cost


def hill_climbing(victims, env, max_iter: int = 500) -> Dict:
    """
    Hill Climbing: swap-based local search for rescue order.
    """
    t0 = time.perf_counter()
    unrescued = [v.vid for v in victims if not v.rescued]
    if not unrescued:
        return {"order": [], "initial_cost": 0, "final_cost": 0,
                "improvement_pct": 0, "iterations": 0,
                "algorithm": "Hill Climbing", "runtime": 0, "history": []}

    current = copy.copy(unrescued)
    random.shuffle(current)
    current_cost = _rescue_order_cost(current, victims, env)
    initial_cost = current_cost
    iterations = 0
    history = [current_cost]

    for _ in range(max_iter):
        iterations += 1
        if len(current) < 2:
            break
        i, j = random.sample(range(len(current)), 2)
        neighbor = copy.copy(current)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        nb_cost = _rescue_order_cost(neighbor, victims, env)
        if nb_cost < current_cost:
            current = neighbor
            current_cost = nb_cost
            history.append(current_cost)

    elapsed = time.perf_counter() - t0
    improvement = ((initial_cost - current_cost) / initial_cost * 100
                   if initial_cost > 0 else 0)
    return {
        "algorithm": "Hill Climbing",
        "order": current,
        "initial_cost": initial_cost,
        "final_cost": current_cost,
        "improvement_pct": improvement,
        "iterations": iterations,
        "runtime": elapsed,
        "history": history,
        "explanation": (
            f"Hill Climbing ran {iterations} iterations. "
            f"Rescue order optimized from cost {initial_cost:.2f} to {current_cost:.2f} "
            f"({improvement:.1f}% improvement). "
            f"Order prioritises critical victims early to minimize weighted rescue cost."
        )
    }


def simulated_annealing(victims, env,
                        T_start: float = 100.0,
                        T_end: float = 0.1,
                        cooling_rate: float = 0.98,
                        max_iter: int = 2000) -> Dict:
    """
    Simulated Annealing: accepts worse solutions with decreasing probability.
    Avoids local optima better than Hill Climbing.
    """
    t0 = time.perf_counter()
    unrescued = [v.vid for v in victims if not v.rescued]
    if not unrescued:
        return {"order": [], "initial_cost": 0, "final_cost": 0,
                "improvement_pct": 0, "iterations": 0,
                "algorithm": "Simulated Annealing", "runtime": 0,
                "history": [], "temp_history": []}

    current = copy.copy(unrescued)
    random.shuffle(current)
    current_cost = _rescue_order_cost(current, victims, env)
    best = copy.copy(current)
    best_cost = current_cost
    initial_cost = current_cost

    T = T_start
    iterations = 0
    history = [current_cost]
    temp_history = [T]

    while T > T_end and iterations < max_iter:
        iterations += 1
        if len(current) < 2:
            break
        i, j = random.sample(range(len(current)), 2)
        neighbor = copy.copy(current)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        nb_cost = _rescue_order_cost(neighbor, victims, env)

        delta = nb_cost - current_cost
        # Accept if better, or probabilistically if worse
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = neighbor
            current_cost = nb_cost
            if current_cost < best_cost:
                best = copy.copy(current)
                best_cost = current_cost

        T *= cooling_rate
        history.append(current_cost)
        temp_history.append(T)

    elapsed = time.perf_counter() - t0
    improvement = ((initial_cost - best_cost) / initial_cost * 100
                   if initial_cost > 0 else 0)
    return {
        "algorithm": "Simulated Annealing",
        "order": best,
        "initial_cost": initial_cost,
        "final_cost": best_cost,
        "improvement_pct": improvement,
        "iterations": iterations,
        "runtime": elapsed,
        "history": history,
        "temp_history": temp_history,
        "explanation": (
            f"SA ran {iterations} iterations (T: {T_start}→{T:.4f}). "
            f"Rescue cost reduced from {initial_cost:.2f} to {best_cost:.2f} "
            f"({improvement:.1f}% improvement). "
            f"SA escaped local optima by accepting worse solutions early."
        )
    }


def compare_local_search(victims, env) -> Dict:
    """Run both HC and SA; compare results."""
    hc = hill_climbing(victims, env)
    sa = simulated_annealing(victims, env)
    winner = "SA" if sa.get("final_cost", 0) < hc.get("final_cost", 0) else "HC"
    return {
        "hill_climbing": hc,
        "simulated_annealing": sa,
        "winner": winner,
        "comparison": (
            f"HC final cost: {hc['final_cost']:.2f}, "
            f"SA final cost: {sa['final_cost']:.2f}. "
            f"{winner} produced better rescue order."
        )
    }
