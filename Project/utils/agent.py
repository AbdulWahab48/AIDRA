"""
utils/agent.py
Core AIDRA agent: integrates all modules and coordinates decisions.
"""

import time
import random
from typing import List, Dict, Optional, Tuple


class AIDRAAgent:
    """
    Master agent that integrates:
    - Search algorithms (BFS, DFS, Greedy, A*)
    - CSP resource allocator
    - ML risk assessor
    - Fuzzy uncertainty handler
    - Dynamic replanner
    """

    def __init__(self, env, fuzzy_system, ml_system, replanner, csp_solver_class):
        self.env = env
        self.fuzzy = fuzzy_system
        self.ml = ml_system
        self.replanner = replanner
        self.CSPSolverClass = csp_solver_class
        self.rescue_order: List[int] = []
        self.assessments: Dict[int, Dict] = {}
        self.csp_result: Optional[Dict] = None
        self.initialized = False

    def initialize(self):
        """Run initial planning: prioritize victims, allocate resources."""
        print("\n[Agent] Initializing AIDRA...")

        # 1. Assess all victims via ML + Fuzzy
        print("[Agent] Assessing victims...")
        for v in self.env.victims:
            node = v.pos
            fire_val = float(self.env.G.nodes[node].get('risk', 0))
            assessment = self.ml.assess_victim(
                v,
                env_risk=fire_val,
                fire_intensity=fire_val,
                road_stability=max(0.1, 1.0 - fire_val/10),
                aftershock_prob=0.3,
                rescue_delay=0.0
            )
            # Fuzzy evaluation
            fuzzy_out = self.fuzzy.evaluate(
                fire_intensity=fire_val,
                road_stability=max(0.01, 1.0 - fire_val/10),
                aftershock_prob=0.3,
                victim_condition=v.condition
            )
            assessment["fuzzy"] = fuzzy_out
            assessment["urgency_score"] = (
                v.severity * 3
                + v.condition * 2
                + fuzzy_out["rescue_urgency"] / 10 * 2
                + (1 - assessment["survival_probability"])
            )
            self.assessments[v.vid] = assessment
            print(f"  V{v.vid}: {assessment['risk_label']}, "
                  f"survival={assessment['survival_probability']:.2f}, "
                  f"urgency={assessment['urgency_score']:.2f}")

        # 2. Sort victims by urgency (highest first)
        self.rescue_order = sorted(
            [v.vid for v in self.env.victims],
            key=lambda vid: self.assessments[vid]["urgency_score"],
            reverse=True
        )
        print(f"[Agent] Rescue order: {self.rescue_order}")
        order_explanation = " → ".join([
            f"V{vid}({self.assessments[vid]['risk_label']})"
            for vid in self.rescue_order
        ])
        print(f"[Agent] Order explanation: {order_explanation}")

        # 3. CSP resource allocation
        print("[Agent] Running CSP resource allocation...")
        solver = self.CSPSolverClass(self.env.victims, self.env,
                                     use_mrv=True, use_forward_checking=True)
        self.csp_result = solver.solve()
        if self.csp_result["success"]:
            print(f"[Agent] CSP solved: {self.csp_result['explanation']}")
        else:
            print("[Agent] CSP failed — using default allocation")

        # 4. Plan initial routes for ambulances
        print("[Agent] Planning initial routes...")
        self._assign_initial_routes()

        self.initialized = True
        print("[Agent] Initialization complete.\n")

    def _assign_initial_routes(self):
        """Assign victims to ambulances based on CSP and rescue order."""
        from search.astar import astar

        assignment = self.csp_result.get("assignment", {}) if self.csp_result else {}

        # Group victims by ambulance
        amb_victims: Dict[int, List[int]] = {0: [], 1: []}
        unrescued_vids = {v.vid for v in self.env.victims if not v.rescued}
        for vid, slot in assignment.items():
            if vid in unrescued_vids:
                # slot is (amb_id, trip_slot) from new CSP
                amb_id = slot[0] if isinstance(slot, tuple) else int(slot)
                if amb_id in amb_victims:
                    amb_victims[amb_id].append(vid)

        # Fallback: split rescue order evenly if CSP didn't spread across both
        if not any(amb_victims.values()):
            for i, vid in enumerate(self.rescue_order):
                amb_id = i % 2
                amb_victims[amb_id].append(vid)

        # Secondary fallback: if any ambulance has NO victims, give it the
        # next unassigned victim from rescue_order so it always moves
        all_assigned = {vid for vlist in amb_victims.values() for vid in vlist}
        for amb_id in [0, 1]:
            if not amb_victims[amb_id]:
                for vid in self.rescue_order:
                    if vid not in all_assigned:
                        amb_victims[amb_id].append(vid)
                        all_assigned.add(vid)
                        break

        for amb in self.env.ambulances:
            victims_for_amb = amb_victims.get(amb.aid, [])
            if not victims_for_amb:
                continue

            # Sort victims for this ambulance by urgency
            victims_for_amb.sort(
                key=lambda vid: self.assessments.get(vid, {}).get("urgency_score", 0),
                reverse=True
            )
            # Get fuzzy mode for route
            first_victim = self.env.victims[victims_for_amb[0]]
            node = first_victim.pos
            fire_val = float(self.env.G.nodes[node].get('risk', 0))
            fuzzy_out = self.fuzzy.evaluate(
                fire_intensity=fire_val,
                road_stability=max(0.01, 1.0 - fire_val/10),
                aftershock_prob=0.3,
                victim_condition=first_victim.condition
            )
            mode = "safe" if fuzzy_out["avoid_route"] else \
                   "speed" if fuzzy_out["route_danger"] < 3 else "balanced"

            # Plan route: base → first victim
            result = astar(self.env, amb.pos, first_victim.pos, mode=mode)
            if result["found"]:
                amb.route = result["path"]
                amb.route_step = 0
                amb.busy = True
                print(f"  Amb {amb.aid}: route to V{victims_for_amb[0]} "
                      f"({mode} mode, {len(result['path'])} steps)")
                self.env.log_decision(
                    event=f"Initial route planned for Ambulance {amb.aid}",
                    reason=f"CSP assigned V{victims_for_amb[0]} to Amb {amb.aid}",
                    prev="No route",
                    new=f"A* {mode}: {len(result['path'])} steps to V{victims_for_amb[0]}",
                    impact=f"Trade-off: {mode} mode — "
                           + ("safety over speed" if mode=="safe" else
                              "speed over safety" if mode=="speed" else
                              "balanced time/risk")
                )

    def step(self) -> List[str]:
        """Run one simulation step; return list of events."""
        events = self.env.step()
        if events:
            self.replanner.process_events(events)

        # Keep idle ambulances moving — dispatch to next victim if they have no route
        for amb in self.env.ambulances:
            if not amb.route or amb.route_step >= len(amb.route):
                if not amb.passengers:
                    # Find next unrescued victim not already being fetched
                    fetched = {
                        v.assigned_ambulance: v.vid
                        for v in self.env.victims
                        if not v.rescued and v.assigned_ambulance is not None
                    }
                    target = next(
                        (v for v in self.env.victims
                         if not v.rescued and v.assigned_ambulance is None),
                        None
                    )
                    if target:
                        self.env._dispatch_ambulance(amb)
                else:
                    # Has passengers but no route → go to medical center
                    mc = self.env.nearest_medical_center(amb.pos)
                    self.env._route_ambulance_to(amb, mc)

        return events

    def get_current_assessment(self, victim_id: int) -> Dict:
        return self.assessments.get(victim_id, {})

    def explain_decision(self, victim_id: int) -> str:
        a = self.assessments.get(victim_id, {})
        return (
            f"Victim {victim_id}: {a.get('risk_label','?')} risk, "
            f"survival={a.get('survival_probability',0):.2f}, "
            f"{a.get('priority','?')}"
        )
