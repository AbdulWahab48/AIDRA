"""
run_demo.py
AIDRA — Quick demonstration runner.
Runs the full system, prints a formatted report, and opens charts.

Usage:
    python run_demo.py            # console demo + charts
    python run_demo.py --full     # also launches Pygame window
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── colours for console output ────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    GRAY   = "\033[90m"

def h1(text):
    print(f"\n{C.BOLD}{C.CYAN}{'='*65}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {text}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'='*65}{C.RESET}")

def h2(text):  print(f"\n{C.BOLD}{C.YELLOW}  ── {text} ──{C.RESET}")

def ok(text):  print(f"  {C.GREEN}✔  {text}{C.RESET}")
def info(text):print(f"  {C.BLUE}ℹ  {text}{C.RESET}")
def warn(text):print(f"  {C.YELLOW}⚠  {text}{C.RESET}")
def err(text): print(f"  {C.RED}✘  {text}{C.RESET}")

# Redefine h1 properly (can't span lines in lambda)
def h1(text):
    print(f"\n{C.BOLD}{C.CYAN}{'='*65}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {text}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'='*65}{C.RESET}")


def demo():
    args = argparse.ArgumentParser()
    args.add_argument("--full", action="store_true", help="Launch Pygame after console demo")
    args = args.parse_args()

    h1("AIDRA — Adaptive Intelligent Disaster Response Agent")
    print(f"  {C.GRAY}AIC-201 Artificial Intelligence | CCP Demonstration{C.RESET}")

    # ── ENVIRONMENT ────────────────────────────────────────────────────────────
    h2("1. Environment Setup")
    from environment import DisasterEnvironment
    env = DisasterEnvironment(seed=42)
    ok(f"Road network loaded: {len(env.G.nodes)} nodes, {len(env.G.edges)} edges (F-8 Islamabad)")
    ok(f"Base node: {env.base_pos} | Medical Centers: {env.medical_centers}")
    ok(f"5 victims placed (2 critical, 2 moderate, 1 minor)")
    ok(f"2 ambulances, 1 rescue team, 10 medical kits")
    fire_nodes  = [n for n in env.G.nodes if env.G.nodes[n].get('risk', 0) >= 8.0]
    hr_nodes    = [n for n in env.G.nodes if 0 < env.G.nodes[n].get('risk', 0) < 8.0]
    blocked_edges = [(u,v) for u,v,k,d in env.G.edges(data=True,keys=True) if d.get('blocked')]
    info(f"Fire zone nodes: {len(fire_nodes)} | High-risk nodes: {len(hr_nodes)} | Blocked edges: {len(blocked_edges)}")

    # ── DATASETS ──────────────────────────────────────────────────────────────
    h2("2. Dataset Generation")
    from datasets.generate_dataset import save_datasets
    victim_df, area_df = save_datasets("datasets")
    ok(f"victim_data.csv: {len(victim_df)} samples, {victim_df.shape[1]} features")
    ok(f"area_risk_data.csv: {len(area_df)} samples, {area_df.shape[1]} features")
    for cls in [0,1,2]:
        n = (victim_df['risk_class']==cls).sum()
        info(f"  Victim risk class {cls}: {n} samples ({n/len(victim_df)*100:.0f}%)")

    # ── MACHINE LEARNING ──────────────────────────────────────────────────────
    h2("3. Machine Learning Training")
    from ml.ml_models import MLSystem
    ml = MLSystem()
    ml_results = ml.train_all(victim_df, area_df)

    knn_m = ml_results["knn"]
    mlp_m = ml_results["mlp"]
    print(f"\n  {'Metric':<12} {'kNN':>8} {'MLP':>8}")
    print(f"  {'-'*30}")
    for metric in ["accuracy","precision","recall","f1"]:
        k = knn_m[metric]; m = mlp_m[metric]
        winner = C.GREEN if k>m else C.RESET
        loser  = C.GREEN if m>k else C.RESET
        print(f"  {metric:<12} {winner}{k:>8.3f}{C.RESET} {loser}{m:>8.3f}{C.RESET}")

    better = "MLP" if mlp_m["f1"] > knn_m["f1"] else "kNN"
    ok(f"{better} achieves higher F1-score → used for agent prioritization")

    # ── FUZZY LOGIC ───────────────────────────────────────────────────────────
    h2("4. Fuzzy Logic Assessment")
    from fuzzy.fuzzy_system import FuzzyDisasterSystem
    fuzzy = FuzzyDisasterSystem()

    scenarios = [
        ("Low danger", dict(fire_intensity=1.0, road_stability=0.9, aftershock_prob=0.1, victim_condition=0.2)),
        ("High danger", dict(fire_intensity=9.0, road_stability=0.1, aftershock_prob=0.9, victim_condition=0.9)),
        ("Mixed",       dict(fire_intensity=5.0, road_stability=0.5, aftershock_prob=0.5, victim_condition=0.6)),
    ]
    print(f"\n  {'Scenario':<14} {'RouteDanger':>12} {'Urgency':>10} {'OverallRisk':>12} {'Decision':>20}")
    print(f"  {'-'*70}")
    for name, s in scenarios:
        r = fuzzy.evaluate(**s)
        decision = ("AVOID+PRIORITIZE" if r["avoid_route"] and r["prioritize_victim"]
                    else "AVOID ROUTE" if r["avoid_route"]
                    else "PRIORITIZE" if r["prioritize_victim"]
                    else "NORMAL")
        col = C.RED if "AVOID" in decision else (C.YELLOW if "PRIOR" in decision else C.GREEN)
        print(f"  {name:<14} {r['route_danger']:>12.2f} {r['rescue_urgency']:>10.2f} "
              f"{r['overall_risk']:>12.2f} {col}{decision:>20}{C.RESET}")

    info("Rules applied:")
    for rule in fuzzy.rules_text[:4]:
        print(f"    {C.GRAY}{rule}{C.RESET}")
    print(f"    {C.GRAY}... (+{len(fuzzy.rules_text)-4} more rules){C.RESET}")

    # ── SEARCH ALGORITHMS ─────────────────────────────────────────────────────
    h2("5. Search Algorithm Comparison")
    from search.bfs import bfs
    from search.dfs import dfs
    from search.greedy import greedy
    from search.astar import astar, astar_compare

    start = env.base_pos
    goal  = env.victims[0].pos
    print(f"\n  Route: {start} → {goal}  (Victim 0 — CRITICAL)")
    print(f"\n  {'Algorithm':<20} {'Steps':>6} {'Cost':>8} {'Risk':>8} {'Nodes':>8} {'Time(ms)':>10}")
    print(f"  {'-'*62}")

    search_results = {}
    for name, fn, kwargs in [
        ("BFS",          bfs,    {}),
        ("DFS",          dfs,    {}),
        ("Greedy BFS",   greedy, {}),
        ("A* (speed)",   astar,  {"mode":"speed"}),
        ("A* (safe)",    astar,  {"mode":"safe"}),
        ("A* (balanced)",astar,  {"mode":"balanced"}),
    ]:
        r = fn(env, start, goal, **kwargs)
        search_results[name] = r
        steps = len(r["path"])-1 if r["path"] else 0
        risk  = r.get("path_risk", 0)
        col   = C.GREEN if name.startswith("A*") else C.RESET
        print(f"  {col}{name:<20}{C.RESET} {steps:>6} {r['path_cost']:>8.2f} "
              f"{risk:>8.2f} {r['nodes_expanded']:>8} {r['runtime']*1000:>10.3f}")

    cmp = astar_compare(env, start, goal)
    info(f"Trade-off: {cmp['tradeoff_analysis']}")

    # ── LOCAL SEARCH ──────────────────────────────────────────────────────────
    h2("6. Local Search Optimization (Rescue Order)")
    from search.local_search import compare_local_search
    ls = compare_local_search(env.victims, env)
    sa = ls["simulated_annealing"]
    hc = ls["hill_climbing"]
    print(f"\n  {'Method':<22} {'Initial Cost':>14} {'Final Cost':>12} {'Improvement':>12} {'Iterations':>12}")
    print(f"  {'-'*74}")
    for method, r in [("Simulated Annealing", sa), ("Hill Climbing", hc)]:
        pct = r.get("improvement_pct", 0)
        col = C.GREEN if pct > 20 else C.YELLOW
        print(f"  {method:<22} {r.get('initial_cost',0):>14.2f} {r.get('final_cost',0):>12.2f} "
              f"  {col}{pct:>9.1f}%{C.RESET} {r.get('iterations',0):>12}")
    winner_name = "Simulated Annealing" if sa["final_cost"] < hc["final_cost"] else "Hill Climbing"
    ok(f"Winner: {winner_name}")
    winner_key = 'simulated_annealing' if ls['winner']=='SA' else 'hill_climbing'
    info(f"Optimal rescue order: {ls[winner_key]['order']}")

    # ── CSP ────────────────────────────────────────────────────────────────────
    h2("7. CSP Resource Allocation")
    from csp.csp_solver import CSPSolver, compare_csp
    csp_results = compare_csp(env.victims, env)
    r_with = csp_results["with_heuristics"]
    r_wo   = csp_results["without_heuristics"]

    if r_with["success"]:
        ok("CSP found valid allocation for all 5 victims")
        assignment = r_with["assignment"]
        kits = r_with["kit_allocation"]
        sev_lbl = {0:"minor",1:"moderate",2:"critical"}
        print(f"\n  {'Victim':<10} {'Severity':<12} {'Ambulance':>10} {'Trip':>6} {'Kits':>6}")
        print(f"  {'-'*46}")
        for vid in sorted(assignment.keys()):
            v = env.victims[vid]
            amb, trip = assignment[vid]
            k = kits.get(vid, 0)
            col = C.RED if v.severity==2 else (C.YELLOW if v.severity==1 else C.GREEN)
            print(f"  {col}V{vid:<9}{C.RESET} {sev_lbl[v.severity]:<12} {amb:>10} {trip:>6} {k:>6}")
        info(f"Rescue team deployed to: {r_with['team_location']}")
    else:
        err("CSP failed")

    print(f"\n  {'Solver':<25} {'Backtracks':>12} {'Nodes':>8} {'Time(ms)':>10}")
    print(f"  {'-'*57}")
    for label, r in [("With MRV+FC", csp_results["constrained_with"]),
                      ("Without heuristics", csp_results["constrained_without"])]:
        print(f"  {label:<25} {r['backtrack_count']:>12} {r['nodes_visited']:>8} "
              f"{r['runtime']*1000:>10.3f}")

    # ── AGENT INIT ─────────────────────────────────────────────────────────────
    h2("8. Agent Initialization & Route Planning")
    from replanning.replanner import Replanner
    from utils.agent import AIDRAAgent

    replanner = Replanner(env, fuzzy_system=fuzzy, ml_system=ml)
    agent = AIDRAAgent(env, fuzzy, ml, replanner, CSPSolver)
    agent.initialize()

    ok("Victim assessments complete")
    info(f"Rescue order (by urgency): {agent.rescue_order}")
    for vid in agent.rescue_order:
        a = agent.assessments[vid]
        col = C.RED if a["risk_label"]=="HIGH" else (C.YELLOW if a["risk_label"]=="MEDIUM" else C.GREEN)
        print(f"    V{vid}: {col}{a['risk_label']:<7}{C.RESET} | "
              f"survival={a['survival_probability']:.2f} | "
              f"urgency={a['urgency_score']:.2f} | "
              f"{a['priority']}")

    # ── DYNAMIC SIMULATION ────────────────────────────────────────────────────
    h2("9. Dynamic Simulation (100 steps)")
    event_count = 0
    for step in range(100):
        events = agent.step()
        if events:
            for ev in events:
                event_count += 1
                col = C.RED if "FIRE" in ev else (C.YELLOW if "AFTERSHOCK" in ev else C.CYAN)
                print(f"  {C.GRAY}Step {env.step_count:>3}{C.RESET}  {col}{ev}{C.RESET}")

    ok(f"Simulation complete: {event_count} dynamic events processed")
    ok(f"Replanning triggered: {env.replanning_triggers} times")
    ok(f"Victims saved: {env.victims_saved}/5")

    # ── DECISION LOG ──────────────────────────────────────────────────────────
    h2("10. Decision Log (last 5 entries)")
    log = replanner.decision_log[-5:]
    for entry in log:
        print(f"\n  {C.BOLD}[{entry['timestamp']:.1f}s]{C.RESET} {C.CYAN}{entry['event']}{C.RESET}")
        print(f"    Reason:    {entry['reason']}")
        print(f"    New plan:  {entry['new_decision']}")
        print(f"    Trade-off: {C.YELLOW}{entry['tradeoff']}{C.RESET}")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    h2("11. Key Performance Indicators")
    kpis = env.get_kpis()
    metrics = [
        ("Victims Saved",         f"{kpis['victims_saved']}/{kpis['victims_total']} ({kpis['victims_pct']:.0f}%)"),
        ("Avg Rescue Time",       f"{kpis['avg_rescue_time']:.3f}s"),
        ("Total Risk Exposure",   f"{kpis['total_risk_exposure']:.2f}"),
        ("Replanning Events",     f"{kpis['replanning_triggers']}"),
        ("Medical Kits Used",     f"{kpis['medical_kits_used']}/10"),
        ("Simulation Steps",      f"{kpis['simulation_steps']}"),
        ("kNN F1-score",          f"{knn_m['f1']:.3f}"),
        ("MLP F1-score",          f"{mlp_m['f1']:.3f}"),
        ("CSP Backtracks (MRV+FC)", f"{csp_results['with_heuristics']['backtrack_count']}"),
        ("SA Improvement",        f"{sa['improvement_pct']:.1f}%"),
    ]
    print()
    for k, v in metrics:
        print(f"  {C.BOLD}{k:<30}{C.RESET} {C.GREEN}{v}{C.RESET}")

    # ── CHARTS ────────────────────────────────────────────────────────────────
    h2("12. Generating KPI Charts")
    from visualization.kpi_charts import generate_all_charts
    chart_paths = generate_all_charts(
        env, search_results, ml_results, csp_results, sa, hc
    )
    for p in chart_paths:
        ok(f"Saved: {p}")

    h1("AIDRA DEMO COMPLETE")
    print(f"  {C.BOLD}All modules operational. Charts saved to results/{C.RESET}")
    print(f"  {C.GRAY}Run  python main.py --headless   for full simulation{C.RESET}")
    print(f"  {C.GRAY}Run  python main.py              to launch Pygame window{C.RESET}\n")

    # ── OPTIONALLY LAUNCH PYGAME ──────────────────────────────────────────────
    if args.full:
        h2("13. Launching Pygame Simulation")
        try:
            import pygame
            from visualization.pygame_vis import AIDRAVisualizer
            env2 = DisasterEnvironment(seed=42)
            replanner2 = Replanner(env2, fuzzy_system=fuzzy, ml_system=ml)
            agent2 = AIDRAAgent(env2, fuzzy, ml, replanner2, CSPSolver)
            agent2.initialize()
            vis = AIDRAVisualizer(env2)
            vis.init()
            while True:
                action = vis.handle_events()
                if action == "quit":
                    break
                if not vis.paused:
                    events = agent2.step()
                    for ev in events:
                        vis.add_event(ev)
                    for amb in env2.ambulances:
                        if amb.route:
                            vis.set_path(amb.route[amb.route_step:])
                            break
                    if env2.victims:
                        v0 = env2.victims[0]
                        fv = float(env2.G.nodes[v0.pos].get("risk", 0))
                        vis.set_fuzzy(fuzzy.evaluate(fv, max(0.01,1-fv/10), 0.3, v0.condition))
                        vis.set_ml(agent2.get_current_assessment(v0.vid))
                vis.render()
                vis.tick(10)
            vis.quit()
        except ImportError:
            warn("Pygame not installed. Install with: pip install pygame")


if __name__ == "__main__":
    demo()
