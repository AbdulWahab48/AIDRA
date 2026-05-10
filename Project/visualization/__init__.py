try:
    from .pygame_vis import AIDRAVisualizer
except ImportError:
    AIDRAVisualizer = None
from .kpi_charts import generate_all_charts
