"""
visualization/kpi_charts.py
Generates all KPI charts and performance metrics for AIDRA.
Uses matplotlib for professional graphs.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from typing import Dict, List

RESULTS_DIR = "results"


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_search_comparison(search_results: Dict):
    """Bar chart comparing BFS, DFS, Greedy, A* on multiple metrics."""
    ensure_results_dir()
    algos = []
    costs, nodes, times, frontier = [], [], [], []

    for name, res in search_results.items():
        if res.get("found"):
            algos.append(name)
            costs.append(res.get("path_cost", 0))
            nodes.append(res.get("nodes_expanded", 0))
            times.append(res.get("runtime", 0) * 1000)
            frontier.append(res.get("frontier_size", 0))

    if not algos:
        return

    x = np.arange(len(algos))
    width = 0.2

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Search Algorithm Comparison", fontsize=14, fontweight='bold')

    data_sets = [
        (costs, "Path Cost", "steelblue"),
        (nodes, "Nodes Expanded", "coral"),
        (times, "Runtime (ms)", "mediumseagreen"),
        (frontier, "Max Frontier Size", "mediumpurple"),
    ]

    for ax, (data, label, color) in zip(axes, data_sets):
        bars = ax.bar(x, data, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_title(label, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=15, ha='right', fontsize=8)
        ax.set_ylabel(label)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "search_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[KPI] Saved: {path}")
    return path


def plot_ml_metrics(ml_results: Dict):
    """Compare kNN vs MLP across accuracy, precision, recall, F1."""
    ensure_results_dir()
    models = []
    metrics_data = {"Accuracy": [], "Precision": [], "Recall": [], "F1": []}

    for model_name, res in ml_results.items():
        models.append(res.get("model", model_name))
        metrics_data["Accuracy"].append(res.get("accuracy", 0))
        metrics_data["Precision"].append(res.get("precision", 0))
        metrics_data["Recall"].append(res.get("recall", 0))
        metrics_data["F1"].append(res.get("f1", 0))

    x = np.arange(len(models))
    width = 0.2
    colors = ['steelblue', 'coral', 'mediumseagreen', 'gold']

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric, values) in enumerate(metrics_data.items()):
        bars = ax.bar(x + i * width, values, width, label=metric,
                      color=colors[i], alpha=0.85, edgecolor='black', linewidth=0.5)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("ML Model Comparison: kNN vs MLP", fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "ml_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[KPI] Saved: {path}")
    return path


def plot_confusion_matrices(ml_results: Dict):
    """Plot confusion matrices for both models."""
    ensure_results_dir()
    labels = ["LOW", "MED", "HIGH"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrices", fontsize=13, fontweight='bold')

    for ax, (model_name, res) in zip(axes, ml_results.items()):
        cm = np.array(res.get("confusion_matrix", [[0]]))
        if cm.size == 0 or cm.ndim < 2:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            continue
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(res.get("model", model_name), fontweight='bold')
        tick_marks = np.arange(len(labels[:cm.shape[0]]))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels[:cm.shape[1]])
        ax.set_yticklabels(labels[:cm.shape[0]])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[KPI] Saved: {path}")
    return path


def plot_local_search_convergence(sa_result: Dict, hc_result: Dict):
    """Plot convergence curves for SA and Hill Climbing."""
    ensure_results_dir()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Local Search Optimization", fontsize=13, fontweight='bold')

    # SA convergence
    if sa_result.get("history"):
        ax1.plot(sa_result["history"], color='coral', linewidth=1.5, alpha=0.8)
        ax1.set_title(f"Simulated Annealing\n"
                      f"Improvement: {sa_result.get('improvement_pct', 0):.1f}%",
                      fontweight='bold')
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Rescue Order Cost")
        ax1.grid(alpha=0.3)
        ax1.axhline(y=sa_result.get("final_cost", 0), color='red',
                    linestyle='--', alpha=0.6, label="Final cost")
        ax1.legend()

    # HC convergence
    if hc_result.get("history"):
        ax2.plot(hc_result["history"], color='steelblue', linewidth=1.5, alpha=0.8,
                 marker='o', markersize=3)
        ax2.set_title(f"Hill Climbing\n"
                      f"Improvement: {hc_result.get('improvement_pct', 0):.1f}%",
                      fontweight='bold')
        ax2.set_xlabel("Improvement Step")
        ax2.set_ylabel("Rescue Order Cost")
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "local_search_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[KPI] Saved: {path}")
    return path


def plot_csp_comparison(csp_results: Dict):
    """Compare CSP with/without heuristics using constrained instance for clear difference."""
    ensure_results_dir()
    categories = ["Backtrack Count", "Runtime (ms)", "Allocation Success"]

    # Use constrained instances for a visible comparison; fall back to standard
    with_h    = csp_results.get("constrained_with", csp_results.get("with_heuristics", {}))
    without_h = csp_results.get("constrained_without", csp_results.get("without_heuristics", {}))

    data_with    = [with_h.get("backtrack_count", 0),
                    with_h.get("runtime", 0) * 1000,
                    1 if with_h.get("success") else 0]
    data_without = [without_h.get("backtrack_count", 0),
                    without_h.get("runtime", 0) * 1000,
                    1 if without_h.get("success") else 0]

    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, data_with,    width, label='With MRV+FC',
                   color='mediumseagreen', alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x + width/2, data_without, width, label='Without Heuristics',
                   color='salmon', alpha=0.85, edgecolor='black')

    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Metric")
    ax.set_title("CSP Solver: With vs Without Heuristics", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "csp_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[KPI] Saved: {path}")
    return path


def plot_kpi_summary(kpis: Dict, search_results: Dict):
    """Final KPI summary dashboard."""
    ensure_results_dir()
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("AIDRA — KPI Performance Dashboard", fontsize=15, fontweight='bold',
                 color='darkred')

    # 1. Victims saved pie
    ax1 = fig.add_subplot(3, 3, 1)
    saved = kpis.get("victims_saved", 0)
    total = kpis.get("victims_total", 5)
    not_saved = total - saved
    ax1.pie([saved, not_saved], labels=["Saved", "Not Saved"],
            colors=["#2ecc71", "#e74c3c"], autopct='%1.0f%%',
            startangle=90, textprops={'fontsize': 10})
    ax1.set_title(f"Victims Saved\n({saved}/{total})", fontweight='bold')

    # 2. Path optimality
    ax2 = fig.add_subplot(3, 3, 2)
    if search_results:
        algo_names = [r for r in search_results if search_results[r].get("found")]
        if algo_names:
            best_cost = min(search_results[r]["path_cost"]
                           for r in algo_names)
            ratios = [search_results[r]["path_cost"] / best_cost
                     for r in algo_names]
            bars = ax2.bar(algo_names, ratios,
                           color=['#3498db','#e74c3c','#2ecc71','#f39c12'][:len(algo_names)],
                           alpha=0.85, edgecolor='black')
            ax2.set_title("Path Optimality Ratio\n(1.0 = optimal)", fontweight='bold')
            ax2.set_ylabel("Cost / Best Cost")
            ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.7)
            for bar in bars:
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.02,
                         f'{bar.get_height():.2f}', ha='center', fontsize=8)

    # 3. Risk exposure bar
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.bar(["Total Risk\nExposure"], [kpis.get("total_risk_exposure", 0)],
            color='tomato', alpha=0.8)
    ax3.bar(["Replanning\nTriggers"], [kpis.get("replanning_triggers", 0)],
            color='steelblue', alpha=0.8)
    ax3.set_title("Risk & Replanning", fontweight='bold')

    # 4. KPI text summary
    ax4 = fig.add_subplot(3, 1, 3)
    ax4.axis('off')
    summary_lines = [
        f"Victims Saved: {kpis.get('victims_saved',0)}/{kpis.get('victims_total',5)} "
        f"({kpis.get('victims_pct',0):.0f}%)",
        f"Average Rescue Time: {kpis.get('avg_rescue_time',0):.2f}s",
        f"Total Risk Exposure: {kpis.get('total_risk_exposure',0):.2f}",
        f"Replanning Events: {kpis.get('replanning_triggers',0)}",
        f"Medical Kits Used: {kpis.get('medical_kits_used',0)}/10",
        f"Simulation Steps: {kpis.get('simulation_steps',0)}",
    ]
    ax4.text(0.05, 0.9, "\n".join(summary_lines), transform=ax4.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             fontfamily='monospace')
    ax4.set_title("Performance Summary", fontweight='bold')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "kpi_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[KPI] Saved: {path}")
    return path


def generate_all_charts(env, search_results, ml_results, csp_results,
                         sa_result, hc_result):
    """Generate all charts and return list of paths."""
    paths = []
    paths.append(plot_search_comparison(search_results))
    if ml_results:
        paths.append(plot_ml_metrics(ml_results))
        paths.append(plot_confusion_matrices(ml_results))
    if sa_result and hc_result:
        paths.append(plot_local_search_convergence(sa_result, hc_result))
    if csp_results:
        paths.append(plot_csp_comparison(csp_results))
    paths.append(plot_kpi_summary(env.get_kpis(), search_results))
    return [p for p in paths if p]
