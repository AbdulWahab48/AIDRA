"""
fuzzy/fuzzy_system.py
Fuzzy Logic uncertainty handler for AIDRA.

Inputs:  fire_intensity, road_stability, aftershock_probability, victim_condition
Outputs: route_danger, rescue_urgency, overall_risk_score

Uses scikit-fuzzy (skfuzzy) for membership functions and rules.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False
    print("[Fuzzy] scikit-fuzzy not found — using fallback crisp approximation")


class FuzzyDisasterSystem:
    """
    Fuzzy inference system for disaster risk assessment.
    Falls back to crisp computation if skfuzzy unavailable.
    """

    def __init__(self):
        self.rules_text = self._define_rules_text()
        self.sim = None
        if SKFUZZY_AVAILABLE:
            self._build_fuzzy_system()

    def _build_fuzzy_system(self):
        # ── Antecedents ─────────────────────────────────────────────
        fire_intensity    = ctrl.Antecedent(np.linspace(0, 10, 100), 'fire_intensity')
        road_stability    = ctrl.Antecedent(np.linspace(0, 1, 100),  'road_stability')
        aftershock_prob   = ctrl.Antecedent(np.linspace(0, 1, 100),  'aftershock_prob')
        victim_condition  = ctrl.Antecedent(np.linspace(0, 1, 100),  'victim_condition')

        # ── Consequents ─────────────────────────────────────────────
        route_danger    = ctrl.Consequent(np.linspace(0, 10, 100), 'route_danger')
        rescue_urgency  = ctrl.Consequent(np.linspace(0, 10, 100), 'rescue_urgency')
        overall_risk    = ctrl.Consequent(np.linspace(0, 10, 100), 'overall_risk')

        # ── Membership functions ─────────────────────────────────────
        fire_intensity['low']    = fuzz.trimf(fire_intensity.universe, [0, 0, 4])
        fire_intensity['medium'] = fuzz.trimf(fire_intensity.universe, [2, 5, 8])
        fire_intensity['high']   = fuzz.trimf(fire_intensity.universe, [6, 10, 10])

        road_stability['low']    = fuzz.trimf(road_stability.universe, [0, 0, 0.4])
        road_stability['medium'] = fuzz.trimf(road_stability.universe, [0.2, 0.5, 0.8])
        road_stability['high']   = fuzz.trimf(road_stability.universe, [0.6, 1, 1])

        aftershock_prob['low']    = fuzz.trimf(aftershock_prob.universe, [0, 0, 0.35])
        aftershock_prob['medium'] = fuzz.trimf(aftershock_prob.universe, [0.2, 0.5, 0.8])
        aftershock_prob['high']   = fuzz.trimf(aftershock_prob.universe, [0.65, 1, 1])

        victim_condition['stable']   = fuzz.trimf(victim_condition.universe, [0, 0, 0.4])
        victim_condition['moderate'] = fuzz.trimf(victim_condition.universe, [0.2, 0.5, 0.8])
        victim_condition['critical'] = fuzz.trimf(victim_condition.universe, [0.6, 1, 1])

        for var in [route_danger, rescue_urgency, overall_risk]:
            var['low']    = fuzz.trimf(var.universe, [0, 0, 4])
            var['medium'] = fuzz.trimf(var.universe, [2, 5, 8])
            var['high']   = fuzz.trimf(var.universe, [6, 10, 10])
            var['very_high'] = fuzz.trimf(var.universe, [8, 10, 10])

        # ── Fuzzy Rules ─────────────────────────────────────────────
        rules = [
            # Route danger rules
            ctrl.Rule(fire_intensity['high'] & road_stability['low'],
                      route_danger['very_high']),
            ctrl.Rule(fire_intensity['high'] & road_stability['medium'],
                      route_danger['high']),
            ctrl.Rule(fire_intensity['medium'] & road_stability['low'],
                      route_danger['high']),
            ctrl.Rule(fire_intensity['low'] & road_stability['high'],
                      route_danger['low']),
            ctrl.Rule(aftershock_prob['high'] & road_stability['low'],
                      route_danger['very_high']),
            ctrl.Rule(fire_intensity['medium'] & aftershock_prob['medium'],
                      route_danger['medium']),

            # Rescue urgency rules
            ctrl.Rule(victim_condition['critical'] & fire_intensity['high'],
                      rescue_urgency['very_high']),
            ctrl.Rule(victim_condition['critical'],
                      rescue_urgency['high']),
            ctrl.Rule(victim_condition['moderate'] & aftershock_prob['high'],
                      rescue_urgency['high']),
            ctrl.Rule(victim_condition['stable'] & fire_intensity['low'],
                      rescue_urgency['low']),
            ctrl.Rule(victim_condition['moderate'],
                      rescue_urgency['medium']),

            # Overall risk rules
            ctrl.Rule(route_danger['very_high'] & rescue_urgency['very_high'],
                      overall_risk['very_high']),
            ctrl.Rule(route_danger['high'] & rescue_urgency['high'],
                      overall_risk['high']),
            ctrl.Rule(route_danger['medium'] & rescue_urgency['medium'],
                      overall_risk['medium']),
            ctrl.Rule(route_danger['low'] & rescue_urgency['low'],
                      overall_risk['low']),
            ctrl.Rule(route_danger['high'] & rescue_urgency['low'],
                      overall_risk['medium']),
        ]

        system = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(system)
        self._fire_intensity = fire_intensity
        self._road_stability = road_stability
        self._aftershock_prob = aftershock_prob
        self._victim_condition = victim_condition

    def _define_rules_text(self):
        return [
            "IF fire IS HIGH AND road_stability IS LOW → route_danger IS VERY_HIGH",
            "IF fire IS HIGH AND road_stability IS MEDIUM → route_danger IS HIGH",
            "IF fire IS MEDIUM AND road_stability IS LOW → route_danger IS HIGH",
            "IF fire IS LOW AND road_stability IS HIGH → route_danger IS LOW",
            "IF aftershock IS HIGH AND road_stability IS LOW → route_danger IS VERY_HIGH",
            "IF victim IS CRITICAL AND fire IS HIGH → rescue_urgency IS VERY_HIGH",
            "IF victim IS CRITICAL → rescue_urgency IS HIGH",
            "IF victim IS MODERATE AND aftershock IS HIGH → rescue_urgency IS HIGH",
            "IF victim IS STABLE → rescue_urgency IS LOW",
            "IF route_danger IS VERY_HIGH AND rescue_urgency IS VERY_HIGH → overall_risk IS VERY_HIGH",
        ]

    def evaluate(self,
                 fire_intensity: float,
                 road_stability: float,
                 aftershock_prob: float,
                 victim_condition: float) -> dict:
        """
        Evaluate fuzzy system. Returns:
          route_danger, rescue_urgency, overall_risk (0-10 scale)
        """
        fire_intensity   = float(np.clip(fire_intensity,   0.01, 9.99))
        road_stability   = float(np.clip(road_stability,   0.01, 0.99))
        aftershock_prob  = float(np.clip(aftershock_prob,  0.01, 0.99))
        victim_condition = float(np.clip(victim_condition, 0.01, 0.99))

        if SKFUZZY_AVAILABLE and self.sim:
            try:
                self.sim.input['fire_intensity']   = fire_intensity
                self.sim.input['road_stability']   = road_stability
                self.sim.input['aftershock_prob']  = aftershock_prob
                self.sim.input['victim_condition'] = victim_condition
                self.sim.compute()
                route_danger   = float(self.sim.output['route_danger'])
                rescue_urgency = float(self.sim.output['rescue_urgency'])
                overall_risk   = float(self.sim.output['overall_risk'])
            except Exception:
                route_danger, rescue_urgency, overall_risk = self._fallback(
                    fire_intensity, road_stability, aftershock_prob, victim_condition)
        else:
            route_danger, rescue_urgency, overall_risk = self._fallback(
                fire_intensity, road_stability, aftershock_prob, victim_condition)

        # Agent decision based on fuzzy output
        replan = overall_risk > 7.0
        prioritize = rescue_urgency > 6.5
        avoid_route = route_danger > 7.0

        return {
            "route_danger": round(route_danger, 3),
            "rescue_urgency": round(rescue_urgency, 3),
            "overall_risk": round(overall_risk, 3),
            "replan_recommended": replan,
            "prioritize_victim": prioritize,
            "avoid_route": avoid_route,
            "explanation": self._explain(route_danger, rescue_urgency, overall_risk,
                                          replan, prioritize, avoid_route),
            "rules_applied": self.rules_text
        }

    def _fallback(self, fire, road, aftershock, condition):
        """Crisp fallback when skfuzzy unavailable."""
        route_danger   = (fire/10)*4 + (1-road)*3 + aftershock*3
        rescue_urgency = condition*5 + (fire/10)*3 + aftershock*2
        overall_risk   = (route_danger + rescue_urgency) / 2
        return (min(route_danger,10), min(rescue_urgency,10), min(overall_risk,10))

    def _explain(self, rd, ru, or_, replan, prioritize, avoid):
        lines = [
            f"Route Danger: {rd:.1f}/10 — {'⚠ AVOID' if avoid else 'OK'}",
            f"Rescue Urgency: {ru:.1f}/10 — {'🚨 PRIORITIZE' if prioritize else 'Normal'}",
            f"Overall Risk: {or_:.1f}/10 — {'🔄 REPLAN' if replan else 'Proceed'}",
        ]
        return " | ".join(lines)

    def batch_evaluate(self, scenarios: list) -> list:
        """Evaluate multiple scenarios."""
        return [self.evaluate(**s) for s in scenarios]
