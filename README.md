# AIDRA — Adaptive Intelligent Disaster Response Agent
**AIC-201 Artificial Intelligence | CCP Project**

---

## Overview

AIDRA is a hybrid AI disaster response simulator integrating:
- **Search Algorithms** (BFS, DFS, Greedy Best-First, A*)
- **Local Search** (Simulated Annealing + Hill Climbing)
- **Constraint Satisfaction** (Backtracking CSP with MRV + Forward Checking)
- **Machine Learning** (kNN + MLP risk classifiers)
- **Fuzzy Logic** (scikit-fuzzy uncertainty handling)
- **Dynamic Replanning** (event-driven route invalidation)
- **Real-time Pygame Visualization**

---

## Project Structure

```
AIDRA/
├── main.py                    # Entry point
├── environment.py             # Disaster grid simulation
├── search/
│   ├── bfs.py                 # Breadth-First Search
│   ├── dfs.py                 # Depth-First Search
│   ├── greedy.py              # Greedy Best-First Search
│   ├── astar.py               # A* (speed/safe/balanced modes)
│   └── local_search.py        # Hill Climbing + Simulated Annealing
├── csp/
│   └── csp_solver.py          # Backtracking CSP with MRV + FC
├── ml/
│   └── ml_models.py           # kNN + MLP classifiers
├── fuzzy/
│   └── fuzzy_system.py        # scikit-fuzzy inference system
├── replanning/
│   └── replanner.py           # Dynamic route replanning engine
├── visualization/
│   ├── pygame_vis.py          # Real-time Pygame display
│   └── kpi_charts.py          # Matplotlib KPI charts
├── datasets/
│   └── generate_dataset.py    # Synthetic dataset generator
├── utils/
│   └── agent.py               # Master agent coordinator
├── results/                   # Auto-generated charts
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the simulation
```bash
# Full Pygame interactive simulation
python main.py

# Headless mode (no window, generates charts only)
python main.py --headless

# Algorithm benchmark only
python main.py --benchmark
```

### 3. Controls (Pygame)
| Key | Action |
|-----|--------|
| SPACE | Pause/Resume |
| R | Reset simulation |
| Q | Quit |

---

## Modules

### Environment (`environment.py`)
- 15×15 grid with fire zones, blocked roads, high-risk areas
- 5 victims (2 critical, 2 moderate, 1 minor)
- 2 ambulances, 1 rescue team, 10 medical kits
- Dynamic events: fire spread, road blockages, aftershocks

### Search & Planning (`search/`)
| Algorithm | Optimality | Speed | Risk-Aware |
|-----------|-----------|-------|-----------|
| BFS | ✅ Shortest hops | ✗ Slow | ✗ |
| DFS | ✗ | ✅ Fast | ✗ |
| Greedy | ✗ | ✅✅ Very fast | ✗ |
| A* | ✅ Optimal | ✅ Fast | ✅ |

**A* cost function:** `f(n) = distance + risk_weight × risk + hazard_penalty`

### CSP (`csp/`)
- Variables: victims; Domains: ambulances
- Hard constraints: max 2 victims/ambulance
- Heuristics: MRV (Minimum Remaining Values) + Forward Checking
- Comparison: with vs. without heuristics

### ML (`ml/`)
- **kNN (k=5):** Predicts victim risk class (Low/Medium/High)
- **MLP (64,32):** Predicts area risk severity
- Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
- **Agent impact:** High risk → prioritize; High area risk → avoid route

### Fuzzy Logic (`fuzzy/`)
Inputs: `fire_intensity`, `road_stability`, `aftershock_prob`, `victim_condition`
Outputs: `route_danger`, `rescue_urgency`, `overall_risk`

Example rule:
```
IF fire IS HIGH AND road_stability IS LOW → route_danger IS VERY_HIGH
```

### Dynamic Replanning (`replanning/`)
Triggered by: road blockages, fire spread, aftershocks
Logs: event, reason, previous decision, new decision, trade-off

---

## KPI Metrics (Auto-generated in `results/`)
- `search_comparison.png` — BFS/DFS/Greedy/A* metrics
- `ml_comparison.png` — kNN vs MLP performance
- `confusion_matrices.png` — ML confusion matrices
- `local_search_convergence.png` — SA vs HC optimization
- `csp_comparison.png` — CSP with/without heuristics
- `kpi_dashboard.png` — Overall performance dashboard

---

## CCP Requirements Coverage

| Requirement | Status | Module |
|-------------|--------|--------|
| BFS/DFS/Greedy/A* | ✅ | search/ |
| Local Search (SA + HC) | ✅ | search/local_search.py |
| CSP + MRV + FC | ✅ | csp/ |
| 2 ML Models + Metrics | ✅ | ml/ |
| Fuzzy Logic | ✅ | fuzzy/ |
| Dynamic Replanning | ✅ | replanning/ |
| Decision Log | ✅ | replanning/ + environment.py |
| KPI Charts | ✅ | visualization/kpi_charts.py |
| Pygame Visualization | ✅ | visualization/pygame_vis.py |
| Modular Structure | ✅ | All modules |

---

## Architecture

```
DisasterEnvironment
       │
       ├── AIDRAAgent (coordinator)
       │       ├── MLSystem (kNN + MLP)
       │       ├── FuzzyDisasterSystem
       │       ├── CSPSolver (MRV + FC)
       │       └── Replanner
       │               └── A* / BFS (on replan)
       │
       └── AIDRAVisualizer (Pygame + matplotlib)
```

---

## Notes
- Synthetic dataset is auto-generated on first run
- ML models are retrained each run (saved to `ml/saved_models/`)
- All decision logs are printed to console and stored in environment
- Charts are saved to `results/` directory
