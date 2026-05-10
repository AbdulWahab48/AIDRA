"""
environment.py
AIDRA Disaster Environment — F-8 Islamabad Road Network Edition.

REPLACES: 15×15 grid
WITH:      Real F-8 Islamabad road graph (OSMnx or synthetic fallback)

ALL EXISTING AI LOGIC IS PRESERVED:
  - Victim dataclass (same fields)
  - Ambulance dataclass (same fields)
  - Dynamic events (fire spread, road blockage, aftershock)
  - KPIs (victims saved, rescue times, risk exposure, replanning)
  - Interfaces used by search algorithms, CSP, fuzzy, ML, replanner

Grid concepts → Road graph concepts:
  (row, col)     → node_id  (OSMnx integer)
  neighbors()    → adjacent passable road nodes
  move_cost()    → edge length + risk penalty
  manhattan()    → Euclidean distance between node screen coords
  BLOCKED cell   → edge with blocked=True (removed from passable)
  FIRE_ZONE cell → node with risk=10
  HIGH_RISK cell → node with risk=5
"""

import random
import time
import math
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set

from datasets.road_network import load_or_build_graph, CoordConverter


# ── Risk constants (mirror old CELL_RISK) ────────────────────────────────────
RISK_CLEAR    = 0.0
RISK_HIGH     = 5.0
RISK_FIRE     = 10.0

# Severity labels / colours (unchanged — used by visualizer and agent)
SEVERITY_LABELS = {0: "Minor", 1: "Moderate", 2: "Critical"}
SEVERITY_COLORS  = {0: (0,200,0), 1: (255,165,0), 2: (220,20,60)}


@dataclass
class Victim:
    vid:                int
    pos:                int           # road node_id (was grid tuple)
    severity:           int           # 0=minor,1=moderate,2=critical
    condition:          float         # 0..1  (1 = worst)
    rescued:            bool  = False
    rescue_time:        Optional[float] = None
    assigned_ambulance: Optional[int]   = None
    survival_prob:      float = 1.0
    area_risk:          float = 0.0


@dataclass
class Ambulance:
    aid:      int
    pos:      int           # current road node_id
    base_pos: int           # home node_id
    capacity: int   = 2
    passengers:     List[int] = field(default_factory=list)
    route:          List[int] = field(default_factory=list)   # sequence of node_ids
    route_step:     int   = 0
    busy:           bool  = False
    returning:      bool  = False
    trips:          int   = 0
    risk_exposure:  float = 0.0
    last_algo:      str   = "A* (balanced)"
    # Smooth animation interpolation (used by pygame_vis only)
    interp_t:       float = 0.0   # 0.0 = at prev node, 1.0 = at next node
    interp_from:    int   = -1    # previous node_id
    interp_to:      int   = -1    # next node_id


@dataclass
class DecisionLog:
    timestamp:         float
    event:             str
    reason:            str
    previous_decision: str
    new_decision:      str
    impact:            str


class DisasterEnvironment:
    """
    Road-network disaster environment for F-8 Islamabad.
    Public interface is compatible with all existing AI modules.
    """

    # Screen layout (used by pygame_vis)
    GRID_SIZE  = 15        # kept for backward compat (ignored in road mode)

    def __init__(self, seed: int = None):
        if seed is None:
            seed = random.randint(0, 99999)
        self.seed = seed
        random.seed(seed)

        # ── Load road graph ──────────────────────────────────────────────
        self.G: nx.MultiDiGraph = load_or_build_graph(seed=seed)
        self._all_nodes: List[int] = list(self.G.nodes)
        self._blocked_edges: Set[Tuple] = set()   # (u, v) pairs

        # ── State ────────────────────────────────────────────────────────
        self.step_count:        int   = 0
        self.start_time:        float = time.time()
        self.decision_log:      List[DecisionLog] = []
        self.dynamic_events:    List[str] = []
        self.victims:           List[Victim]    = []
        self.ambulances:        List[Ambulance] = []
        self.medical_centers:   List[int]       = []   # node_ids
        self.base_pos:          int             = 0    # node_id
        self.rescue_team_busy:  bool  = False
        self.rescue_team_pos:   Optional[int]   = None
        self.medical_kits:      int   = 10
        self.replanning_triggers: int = 0
        self.victims_saved:     int   = 0
        self.rescue_times:      List[float] = []
        self._fuzzy             = None   # injected by agent

        self._setup_environment()

    # ------------------------------------------------------------------ #
    #  SETUP                                                               #
    # ------------------------------------------------------------------ #
    def _setup_environment(self):
        nodes = self._all_nodes
        n = len(nodes)

        def rand_node(exclude: set = None) -> int:
            exclude = exclude or set()
            candidates = [nd for nd in nodes if nd not in exclude]
            return random.choice(candidates)

        used: Set[int] = set()

        # Base — first node (stable reference point)
        self.base_pos = nodes[0]
        used.add(self.base_pos)

        # 2 Medical centers — spread across graph
        mc1 = nodes[n // 3]
        mc2 = nodes[2 * n // 3]
        self.medical_centers = [mc1, mc2]
        used.update(self.medical_centers)

        # Fire zones — 4 random nodes, high risk
        fire_nodes = random.sample([nd for nd in nodes if nd not in used],
                                   min(4, n // 6))
        for nd in fire_nodes:
            self.G.nodes[nd]['risk'] = RISK_FIRE
            self._propagate_node_risk(nd, RISK_FIRE)
        used.update(fire_nodes)

        # High-risk zones — 5 random nodes
        hr_nodes = random.sample([nd for nd in nodes if nd not in used],
                                 min(5, n // 5))
        for nd in hr_nodes:
            self.G.nodes[nd]['risk'] = RISK_HIGH
            self._propagate_node_risk(nd, RISK_HIGH)
        used.update(hr_nodes)

        # Blocked roads — disable 4-6 random edges
        all_edges = list(self.G.edges(keys=True))
        block_count = min(random.randint(4, 6), len(all_edges) // 4)
        for u, v, k in random.sample(all_edges, block_count):
            self.G[u][v][k]['blocked'] = True
            self._blocked_edges.add((u, v))

        # Victims: 2 critical, 2 moderate, 1 minor
        severities = [2, 2, 1, 1, 0]
        conditions = [
            round(random.uniform(0.75, 0.95), 2),
            round(random.uniform(0.70, 0.90), 2),
            round(random.uniform(0.45, 0.70), 2),
            round(random.uniform(0.40, 0.65), 2),
            round(random.uniform(0.20, 0.45), 2),
        ]
        for i, (sev, cond) in enumerate(zip(severities, conditions)):
            pos = rand_node(used)
            used.add(pos)
            v = Victim(vid=i, pos=pos, severity=sev, condition=cond)
            v.area_risk = float(self.G.nodes[pos].get('risk', 0))
            self.victims.append(v)

        # Ambulances start at base
        for i in range(2):
            self.ambulances.append(
                Ambulance(aid=i, pos=self.base_pos, base_pos=self.base_pos)
            )

    def _propagate_node_risk(self, node: int, risk: float):
        """Set risk on all edges touching this node."""
        for u, v, k in self.G.edges(node, keys=True):
            self.G[u][v][k]['risk'] = max(self.G[u][v][k].get('risk', 0), risk)
        for u, v, k in self.G.in_edges(node, keys=True):
            self.G[u][v][k]['risk'] = max(self.G[u][v][k].get('risk', 0), risk)

    # ------------------------------------------------------------------ #
    #  NAVIGATION  (same interface as grid version)                        #
    # ------------------------------------------------------------------ #
    def is_passable(self, node: int, from_node: int = None) -> bool:
        """Node is passable if it exists and has at least one open incoming edge."""
        if node not in self.G.nodes:
            return False
        # Fire nodes are impassable
        if self.G.nodes[node].get('risk', 0) >= RISK_FIRE:
            return False
        # If coming from a specific node, check that edge isn't blocked
        if from_node is not None:
            edges = self.G.get_edge_data(from_node, node)
            if not edges:
                return False
            for k, data in edges.items():
                if not data.get('blocked', False):
                    return True
            return False
        return True

    def neighbors(self, node: int) -> List[int]:
        """Return passable neighbors of node."""
        result = []
        for _, v, k, data in self.G.out_edges(node, data=True, keys=True):
            if not data.get('blocked', False) and self.is_passable(v, node):
                result.append(v)
        return result

    def move_cost(self, node: int, from_node: int = None) -> float:
        """Cost to enter node (edge length + risk penalty)."""
        risk = float(self.G.nodes[node].get('risk', 0))
        base = 1.0
        if from_node is not None:
            edges = self.G.get_edge_data(from_node, node) or {}
            lengths = [d.get('length', 50) for d in edges.values()
                       if not d.get('blocked', False)]
            if lengths:
                base = min(lengths) / 50.0   # normalise: 50m = cost 1
        return base + risk * 0.5

    def manhattan(self, a: int, b: int) -> float:
        """
        Heuristic distance between two nodes.
        Uses Euclidean distance on screen coords (replaces grid manhattan).
        """
        if a not in self.G.nodes or b not in self.G.nodes:
            return 0.0
        da, db = self.G.nodes[a], self.G.nodes[b]
        # Approx metres using degree differences
        dlat = (da['y'] - db['y']) * 111_000
        dlon = (da['x'] - db['x']) * 111_000 * math.cos(math.radians(da['y']))
        return math.sqrt(dlat**2 + dlon**2) / 50.0   # normalise same as move_cost

    def path_cost(self, path: List[int]) -> float:
        total = 0.0
        for i in range(1, len(path)):
            total += self.move_cost(path[i], path[i-1])
        return total

    def path_risk(self, path: List[int]) -> float:
        return sum(float(self.G.nodes[n].get('risk', 0)) for n in path)

    def nearest_medical_center(self, pos: int) -> int:
        return min(self.medical_centers, key=lambda mc: self.manhattan(pos, mc))

    # ------------------------------------------------------------------ #
    #  DYNAMIC EVENTS                                                      #
    # ------------------------------------------------------------------ #
    def step(self) -> List[str]:
        self.step_count += 1
        events = []

        if self.step_count % 15 == 0:
            ev = self._spread_fire()
            if ev:
                events.append(ev)

        if self.step_count % 20 == 0:
            ev = self._random_blockage()
            if ev:
                events.append(ev)

        if self.step_count % 25 == 0:
            ev = self._aftershock_event()
            if ev:
                events.append(ev)

        self._move_ambulances()

        if self.step_count % 5 == 0:
            self._deteriorate_victims()

        self.dynamic_events.extend(events)
        return events

    def _spread_fire(self) -> Optional[str]:
        fire_nodes = [n for n in self.G.nodes
                      if self.G.nodes[n].get('risk', 0) >= RISK_FIRE]
        if not fire_nodes:
            return None
        src = random.choice(fire_nodes)
        candidates = list(self.G.successors(src))
        if not candidates:
            return None
        target = random.choice(candidates)
        if self.G.nodes[target].get('risk', 0) < RISK_FIRE:
            self.G.nodes[target]['risk'] = RISK_FIRE
            self._propagate_node_risk(target, RISK_FIRE)
            return f"FIRE SPREAD to node {target}"
        return None

    def _random_blockage(self) -> Optional[str]:
        candidates = [
            (u, v, k) for u, v, k, d in self.G.edges(data=True, keys=True)
            if not d.get('blocked', False)
            and u not in self.medical_centers
            and v not in self.medical_centers
            and u != self.base_pos
            and v != self.base_pos
        ]
        if not candidates:
            return None
        u, v, k = random.choice(candidates)
        self.G[u][v][k]['blocked'] = True
        self._blocked_edges.add((u, v))
        name = self.G[u][v][k].get('name', 'Road')
        return f"ROAD BLOCKED: {name} ({u}→{v})"

    def _aftershock_event(self) -> Optional[str]:
        hr_nodes = [n for n in self.G.nodes
                    if 0 < self.G.nodes[n].get('risk', 0) < RISK_FIRE]
        for n in hr_nodes:
            new_risk = min(self.G.nodes[n]['risk'] + 1.0, RISK_FIRE - 0.1)
            self.G.nodes[n]['risk'] = new_risk
            self._propagate_node_risk(n, new_risk)
        if hr_nodes:
            return "AFTERSHOCK: risk levels increased in high-risk zones"
        return None

    def _deteriorate_victims(self):
        for v in self.victims:
            if not v.rescued:
                v.condition = min(1.0, v.condition + 0.02)
                if v.severity == 2:
                    v.condition = min(1.0, v.condition + 0.01)

    # ------------------------------------------------------------------ #
    #  AMBULANCE MOVEMENT                                                  #
    # ------------------------------------------------------------------ #
    def _move_ambulances(self):
        for amb in self.ambulances:
            if not amb.route or amb.route_step >= len(amb.route):
                continue

            next_node = amb.route[amb.route_step]
            prev_node = amb.route[amb.route_step - 1] if amb.route_step > 0 else amb.pos

            # Check if edge/node still passable
            if not self.is_passable(next_node, prev_node):
                amb.route = []
                amb.route_step = 0
                self._log_decision(
                    event=f"PATH INVALIDATED for Ambulance {amb.aid}",
                    reason=f"Node/edge {prev_node}→{next_node} became impassable",
                    prev=f"Following route",
                    new="Route cleared; replanning needed",
                    impact="Rescue delayed; replanning triggered"
                )
                self.replanning_triggers += 1
                continue

            risk = float(self.G.nodes[next_node].get('risk', 0))
            amb.risk_exposure += risk
            # Set interpolation state for smooth animation
            amb.interp_from = amb.pos
            amb.interp_to   = next_node
            amb.interp_t    = 0.0
            amb.pos = next_node
            amb.route_step += 1

            # Mid-route victim pickup
            for v in self.victims:
                if (not v.rescued
                        and v.pos == amb.pos
                        and len(amb.passengers) < amb.capacity
                        and (v.assigned_ambulance is None
                             or v.assigned_ambulance == amb.aid)):
                    amb.passengers.append(v.vid)
                    v.assigned_ambulance = amb.aid

            # If carrying passengers ensure route ends at medical center
            if amb.passengers and amb.route:
                dest = amb.route[-1]
                if dest not in self.medical_centers:
                    mc = self.nearest_medical_center(amb.pos)
                    self._route_ambulance_to(amb, mc)

            # Opportunistic nearby pickup
            if len(amb.passengers) < amb.capacity:
                self._opportunistic_pickup(amb)

            # Reached end of route
            if amb.route_step >= len(amb.route):
                self._on_ambulance_arrived(amb)

    def _on_ambulance_arrived(self, amb: Ambulance):
        pos = amb.pos

        # Final pickup at destination node
        for v in self.victims:
            if (not v.rescued
                    and v.pos == pos
                    and len(amb.passengers) < amb.capacity
                    and (v.assigned_ambulance is None
                         or v.assigned_ambulance == amb.aid)):
                amb.passengers.append(v.vid)
                v.assigned_ambulance = amb.aid

        # At medical center → rescue all passengers
        if pos in self.medical_centers and amb.passengers:
            for vid in amb.passengers:
                v = self.victims[vid]
                v.rescued = True
                v.rescue_time = time.time() - self.start_time
                self.rescue_times.append(v.rescue_time)
                self.victims_saved += 1
                v.assigned_ambulance = None
            amb.passengers = []
            amb.trips += 1
            amb.returning = False
            amb.busy = False
            self._dispatch_ambulance(amb)
            return

        # Has passengers but not at medical center
        if amb.passengers:
            mc = self.nearest_medical_center(pos)
            self._route_ambulance_to(amb, mc)
        else:
            self._dispatch_ambulance(amb)

    def _opportunistic_pickup(self, amb: Ambulance):
        PICKUP_RADIUS = 2
        DETOUR_LIMIT  = 3

        if len(amb.passengers) >= amb.capacity or not amb.route:
            return

        unrescued = [
            v for v in self.victims
            if not v.rescued
            and v.assigned_ambulance is None
            and self.manhattan(amb.pos, v.pos) <= PICKUP_RADIUS
        ]
        if not unrescued:
            return

        remaining = amb.route[amb.route_step:]
        if not remaining:
            return
        current_dest = remaining[-1]
        direct_remaining = len(remaining)

        best, best_extra = None, float('inf')
        for v in unrescued:
            detour = (self.manhattan(amb.pos, v.pos)
                      + self.manhattan(v.pos, current_dest))
            extra  = detour - direct_remaining
            if extra <= DETOUR_LIMIT and extra < best_extra:
                best_extra = extra
                best = v

        if best:
            dest_victims = [
                v for v in self.victims
                if not v.rescued
                and v.assigned_ambulance == amb.aid
                and v.pos == current_dest
                and v.vid not in amb.passengers
            ]
            for dv in dest_victims:
                dv.assigned_ambulance = None

            best.assigned_ambulance = amb.aid
            from search.astar import astar
            to_victim = astar(self, amb.pos, best.pos, mode="balanced")
            if to_victim["found"] and len(to_victim["path"]) > 1:
                to_dest = astar(self, best.pos, current_dest, mode="balanced")
                if to_dest["found"]:
                    combined = to_victim["path"] + to_dest["path"][1:]
                    amb.route = combined
                    amb.route_step = 1
                    self._log_decision(
                        event=f"OPPORTUNISTIC PICKUP: Amb {amb.aid} → V{best.vid}",
                        reason=f"V{best.vid} within {self.manhattan(amb.pos, best.pos):.1f} units",
                        prev=f"Heading to node {current_dest}",
                        new=f"Rerouted via V{best.vid}",
                        impact=f"Capacity {len(amb.passengers)+1}/2"
                    )

    def _dispatch_ambulance(self, amb: Ambulance):
        DETOUR_THRESHOLD = 4

        unrescued = [
            v for v in self.victims
            if not v.rescued and v.assigned_ambulance is None
        ]

        if not unrescued:
            if amb.pos != amb.base_pos:
                self._route_ambulance_to(amb, amb.base_pos)
            else:
                amb.busy = False
            return

        if len(amb.passengers) >= amb.capacity:
            mc = self.nearest_medical_center(amb.pos)
            self._route_ambulance_to(amb, mc)
            return

        if amb.passengers:
            mc = self.nearest_medical_center(amb.pos)
            direct_dist = self.manhattan(amb.pos, mc)
            best_detour, best_cost = None, float('inf')
            for v in unrescued:
                detour = (self.manhattan(amb.pos, v.pos)
                          + self.manhattan(v.pos, mc))
                extra  = detour - direct_dist
                if extra <= DETOUR_THRESHOLD and detour < best_cost:
                    best_cost   = detour
                    best_detour = v
            if best_detour:
                best_detour.assigned_ambulance = amb.aid
                self._route_ambulance_to(amb, best_detour.pos)
                return
            self._route_ambulance_to(amb, mc)
            return

        target = min(unrescued, key=lambda v: self.manhattan(amb.pos, v.pos))
        target.assigned_ambulance = amb.aid
        self._route_ambulance_to(amb, target.pos)

    def _route_ambulance_to(self, amb: Ambulance, goal: int):
        from search.astar import astar

        # Fuzzy-driven mode selection
        mode = "balanced"
        if self._fuzzy:
            risk_at_pos = float(self.G.nodes[amb.pos].get('risk', 0))
            try:
                fo = self._fuzzy.evaluate(
                    fire_intensity=risk_at_pos,
                    road_stability=max(0.01, 1.0 - risk_at_pos / 10),
                    aftershock_prob=0.3,
                    victim_condition=0.5
                )
                if fo.get("avoid_route"):
                    mode = "safe"
                elif fo.get("route_danger", 5) < 3.0:
                    mode = "speed"
            except Exception:
                mode = "balanced"

        result = astar(self, amb.pos, goal, mode=mode)
        if result["found"] and len(result["path"]) > 1:
            amb.route = result["path"]
            amb.route_step = 1
            amb.busy = True
            amb.last_algo = f"A* ({mode})"
        else:
            amb.busy = False

    # ------------------------------------------------------------------ #
    #  LOGGING                                                             #
    # ------------------------------------------------------------------ #
    def _log_decision(self, event, reason, prev, new, impact):
        self.decision_log.append(DecisionLog(
            timestamp=time.time() - self.start_time,
            event=event, reason=reason,
            previous_decision=prev,
            new_decision=new, impact=impact
        ))

    def log_decision(self, event, reason, prev, new, impact):
        self._log_decision(event, reason, prev, new, impact)

    # ------------------------------------------------------------------ #
    #  RISK MAP ACCESSOR (for search algorithms)                           #
    # ------------------------------------------------------------------ #
    @property
    def risk_map(self):
        """Dict-like accessor: risk_map[node] → float. Used by search algos."""
        return {n: float(d.get('risk', 0))
                for n, d in self.G.nodes(data=True)}

    # ------------------------------------------------------------------ #
    #  KPIs                                                                #
    # ------------------------------------------------------------------ #
    def get_kpis(self) -> Dict:
        import numpy as np
        total = len(self.victims)
        saved = self.victims_saved
        avg_rescue = float(np.mean(self.rescue_times)) if self.rescue_times else 0
        kits_used  = 10 - self.medical_kits
        risk_exp   = sum(a.risk_exposure for a in self.ambulances)
        return {
            "victims_total":       total,
            "victims_saved":       saved,
            "victims_pct":         saved / total * 100 if total else 0,
            "avg_rescue_time":     avg_rescue,
            "medical_kits_used":   kits_used,
            "ambulance_utilization": sum(a.trips for a in self.ambulances),
            "total_risk_exposure": risk_exp,
            "replanning_triggers": self.replanning_triggers,
            "simulation_steps":    self.step_count,
        }
