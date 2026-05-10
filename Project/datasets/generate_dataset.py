"""
datasets/generate_dataset.py
Generates synthetic training data for ML models.
Features: severity, condition, fire_intensity, road_stability, aftershock_prob, area_risk
Targets: survival_probability (regression), risk_class (classification)
"""

import numpy as np
import pandas as pd
import os

SEED = 42


def generate_victim_dataset(n_samples: int = 500) -> pd.DataFrame:
    """
    Synthetic dataset: victim features → survival probability + risk class.
    """
    np.random.seed(SEED)
    severity = np.random.randint(0, 3, n_samples)            # 0=minor,1=mod,2=critical
    condition = np.random.uniform(0.1, 1.0, n_samples)       # 0=fine,1=critical
    fire_intensity = np.random.uniform(0.0, 10.0, n_samples) # 0-10
    road_stability = np.random.uniform(0.0, 1.0, n_samples)  # 0=broken,1=stable
    aftershock_prob = np.random.uniform(0.0, 1.0, n_samples)
    area_risk = np.random.uniform(0.0, 15.0, n_samples)
    rescue_delay = np.random.uniform(0, 30, n_samples)       # minutes

    # Survival probability formula
    base_survival = 1.0 - (severity / 3.0) * 0.4
    survival_prob = (
        base_survival
        - condition * 0.3
        - (fire_intensity / 10.0) * 0.15
        - (1 - road_stability) * 0.1
        - aftershock_prob * 0.1
        - (area_risk / 15.0) * 0.1
        - (rescue_delay / 30.0) * 0.15
        + np.random.normal(0, 0.05, n_samples)
    )
    survival_prob = np.clip(survival_prob, 0.05, 0.99)

    # Risk class: 0=low, 1=medium, 2=high
    risk_score = (
        (severity / 2.0) * 3
        + condition * 2
        + (fire_intensity / 10.0) * 3
        + (1 - road_stability) * 2
        + aftershock_prob * 2
        + (area_risk / 15.0) * 3
    ) / 15.0
    risk_class = np.digitize(risk_score, bins=[0.33, 0.66])  # 0,1,2

    df = pd.DataFrame({
        "severity": severity,
        "condition": condition.round(4),
        "fire_intensity": fire_intensity.round(4),
        "road_stability": road_stability.round(4),
        "aftershock_prob": aftershock_prob.round(4),
        "area_risk": area_risk.round(4),
        "rescue_delay": rescue_delay.round(2),
        "survival_prob": survival_prob.round(4),
        "risk_class": risk_class,
    })
    return df


def generate_area_risk_dataset(n_samples: int = 500) -> pd.DataFrame:
    """Synthetic area risk classification dataset."""
    np.random.seed(SEED + 1)
    fire = np.random.uniform(0, 10, n_samples)
    stability = np.random.uniform(0, 1, n_samples)
    aftershock = np.random.uniform(0, 1, n_samples)
    structure_damage = np.random.uniform(0, 1, n_samples)
    time_elapsed = np.random.uniform(0, 120, n_samples)

    score = (
        fire * 0.3
        + (1 - stability) * 3
        + aftershock * 2
        + structure_damage * 2
        + (time_elapsed / 120) * 1
    ) / 9.0
    risk_class = np.digitize(score, bins=[0.33, 0.66])

    df = pd.DataFrame({
        "fire_intensity": fire.round(4),
        "road_stability": stability.round(4),
        "aftershock_prob": aftershock.round(4),
        "structure_damage": structure_damage.round(4),
        "time_elapsed": time_elapsed.round(2),
        "risk_class": risk_class,
    })
    return df


def save_datasets(base_dir: str = "datasets"):
    os.makedirs(base_dir, exist_ok=True)
    victim_df = generate_victim_dataset()
    area_df = generate_area_risk_dataset()
    victim_df.to_csv(os.path.join(base_dir, "victim_data.csv"), index=False)
    area_df.to_csv(os.path.join(base_dir, "area_risk_data.csv"), index=False)
    print(f"[Dataset] Saved victim_data.csv ({len(victim_df)} rows)")
    print(f"[Dataset] Saved area_risk_data.csv ({len(area_df)} rows)")
    return victim_df, area_df


if __name__ == "__main__":
    save_datasets()
