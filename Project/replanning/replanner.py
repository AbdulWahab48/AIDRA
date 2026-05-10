"""
replanning/replanner.py
Dynamic replanning engine for AIDRA — Road Network Edition.

All grid (r, c) references replaced with road node_id (int).
Interface unchanged — called identically by utils/agent.py and main.py.
"""

from typing import List, Dict, Optional
from search.astar import astar
from search.bfs import bfs
import time


class Replanner:
    """
    Monitors events, validates routes, and triggers replanning.
    Maintains a full decision log.
    Routes are now sequences of road node_ids (int) not grid tuples.
    """

    def __init__(self, env, fuzzy_system=None, ml_system=None):
        self.env          = env
        self.fuzzy        = fuzzy_system
        self.ml           = ml_system
        self.replan_count = 0
        self.decision_log: List[Dict] = []
        self.current_routes: Dict[int, List] = {}   # amb_id → list of node_ids

    # ------------------------------------------------------------------ #
    #  ROUTE VALIDATION                                                    #
    # ------------------------------------------------------------------ #
    def validate_route(self, amb_id: int, route: List) -> bool:
        """
        Check every node in route is still passable.
        Works with road node_ids (int) — no (r,c) unpacking.
        """
        for i, node in enumerate(route):
            prev = route[i - 1] if i > 0 else None
            if not self.env.is_passable(node, prev):
                return False
        return True

    def check_all_routes(self) -> List[int]:
        """Return ambulance IDs whose current routes contain blocked/fire nodes."""
        invalid = []
        for amb in self.env.ambulances:
            if amb.route:
                remaining = amb.route[amb.route_step:]
                if remaining and not self.validate_route(amb.aid, remaining):
                    invalid.append(amb.aid)
        return invalid

    # ------------------------------------------------------------------ #
    #  REPLANNING                                                          #
    # ------------------------------------------------------------------ #
    def replan_ambulance(self, amb_id: int, reason: str,
                          algorithm: str = "astar") -> Optional[List]:
        """
        Replan route for ambulance from its current road node.
        Returns new route (list of node_ids) or [] on failure.
        """
        amb   = self.env.ambulances[amb_id]
        start = amb.pos   # road node_id (int)

        # Determine goal node
        if amb.passengers:
            goal      = self.env.nearest_medical_center(start)
            goal_desc = f"medical center (node {goal})"
        else:
            unrescued = [
                v for v in self.env.victims
                if not v.rescued
                and (v.assigned_ambulance is None or v.assigned_ambulance == amb_id)
            ]
            if not unrescued:
                return []
            victim    = min(unrescued, key=lambda v: self.env.manhattan(start, v.pos))
            goal      = victim.pos
            goal_desc = f"victim {victim.vid} (node {goal})"

        prev_route = list(amb.route)

        # Fuzzy-driven mode selection — use node risk, not grid cell
        mode = "balanced"
        if self.fuzzy:
            fire_val = float(self.env.G.nodes[start].get('risk', 0))
            try:
                fuzzy_out = self.fuzzy.evaluate(
                    fire_intensity=fire_val,
                    road_stability=max(0.01, 1.0 - fire_val / 10),
                    aftershock_prob=0.3,
                    victim_condition=0.5
                )
                if fuzzy_out["avoid_route"]:
                    mode = "safe"
                elif fuzzy_out["route_danger"] < 3:
                    mode = "speed"
            except Exception:
                mode = "balanced"

        # Run search
        if algorithm == "astar":
            result = astar(self.env, start, goal, mode=mode)
        else:
            result = bfs(self.env, start, goal)

        new_route = result["path"] if result["found"] else []

        # Build log entry
        log_entry = {
            "timestamp": round(time.time() - self.env.start_time, 2),
            "event": f"REPLANNING — Ambulance {amb_id}",
            "reason": reason,
            "previous_decision": (
                f"Following route of {len(prev_route)} nodes"
                if prev_route else "No active route"
            ),
            "new_decision": (
                f"A* ({mode}) planned {len(new_route)} nodes to {goal_desc}"
                if new_route else "No valid path found — ambulance halted"
            ),
            "algorithm_used": f"{algorithm} ({mode})",
            "impact": (
                f"Route updated: {len(prev_route)}→{len(new_route)} nodes. "
                f"Mode={mode}: "
                + ("Safety over speed" if mode == "safe"
                   else "Speed over safety" if mode == "speed"
                   else "Balanced time/risk")
            ),
            "tradeoff": (
                f"MODE={mode}: "
                + ("Prioritizing SAFETY — longer road path chosen" if mode == "safe"
                   else "Prioritizing SPEED — risk accepted" if mode == "speed"
                   else "Balancing time and risk")
            )
        }
        self.decision_log.append(log_entry)

        self.env.log_decision(
            event=log_entry["event"],
            reason=log_entry["reason"],
            prev=log_entry["previous_decision"],
            new=log_entry["new_decision"],
            impact=log_entry["impact"]
        )

        # Apply new route
        if new_route:
            amb.route      = new_route
            amb.route_step = 1    # index 0 = current pos
            self.current_routes[amb_id] = new_route
            self.replan_count += 1

        return new_route

    # ------------------------------------------------------------------ #
    #  EVENT PROCESSING                                                    #
    # ------------------------------------------------------------------ #
    def process_events(self, events: List[str]) -> List[Dict]:
        """Process dynamic events, validate routes, trigger replanning."""
        responses = []
        if not events:
            return responses

        invalid_ambs = self.check_all_routes()

        for amb_id in invalid_ambs:
            reason    = "; ".join(events)
            new_route = self.replan_ambulance(amb_id, reason=reason)
            responses.append({
                "ambulance":        amb_id,
                "triggered_by":     events,
                "new_route_length": len(new_route) if new_route else 0
            })
            self.env.replanning_triggers += 1

        # Log event types
        for event in events:
            if "FIRE" in event:
                self._handle_fire_event(event)
            elif "BLOCKED" in event:
                self._handle_blockage_event(event)
            elif "AFTERSHOCK" in event:
                self._handle_aftershock_event()

        return responses

    def _handle_fire_event(self, event: str):
        self.decision_log.append({
            "timestamp":         round(time.time() - self.env.start_time, 2),
            "event":             event,
            "reason":            "Fire spread to new road node",
            "previous_decision": "Route through affected area",
            "new_decision":      "A* reroutes around fire node (impassable)",
            "impact":            "Risk map updated; fire nodes marked impassable",
            "tradeoff":          "Safety prioritized — detour around fire zone"
        })

    def _handle_blockage_event(self, event: str):
        self.decision_log.append({
            "timestamp":         round(time.time() - self.env.start_time, 2),
            "event":             event,
            "reason":            "Road edge blocked by aftershock/debris",
            "previous_decision": "Route using now-blocked road",
            "new_decision":      "A* replans avoiding blocked edge",
            "impact":            "Rescue delayed; new road path computed",
            "tradeoff":          "No alternative — blocked edge forces detour"
        })

    def _handle_aftershock_event(self):
        self.decision_log.append({
            "timestamp":         round(time.time() - self.env.start_time, 2),
            "event":             "AFTERSHOCK — risk levels elevated",
            "reason":            "Seismic activity increased node risk scores",
            "previous_decision": "Standard routing",
            "new_decision":      "A* safe mode engaged — avoid elevated risk nodes",
            "impact":            "Longer road paths chosen to minimize exposure",
            "tradeoff":          "TIME sacrificed to minimize RISK EXPOSURE"
        })

    # ------------------------------------------------------------------ #
    #  METRICS                                                             #
    # ------------------------------------------------------------------ #
    def get_summary(self) -> Dict:
        return {
            "total_replannings":    self.replan_count,
            "decision_log_entries": len(self.decision_log),
            "log":                  self.decision_log
        }

    def print_log(self):
        print("\n" + "="*60)
        print("DECISION LOG — Road Network Replanning")
        print("="*60)
        for entry in self.decision_log[-10:]:
            print(f"[{entry['timestamp']:.1f}s] {entry['event']}")
            print(f"  Reason:    {entry['reason']}")
            print(f"  New plan:  {entry['new_decision']}")
            print(f"  Trade-off: {entry['tradeoff']}")
            print()
