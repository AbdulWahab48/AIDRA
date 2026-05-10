"""
ml/ml_models.py
Machine Learning models for AIDRA.

Model 1: kNN — victim risk classification
Model 2: MLPClassifier — area risk severity prediction

Both models directly influence agent decisions.
"""

import numpy as np
import pandas as pd
import os
import joblib
import time
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = "ml/saved_models"
RISK_LABELS = ["LOW", "MEDIUM", "HIGH"]


class VictimRiskKNN:
    """kNN model predicting victim risk class (0=low, 1=medium, 2=high)."""

    FEATURES = ["severity", "condition", "fire_intensity",
                 "road_stability", "aftershock_prob", "area_risk", "rescue_delay"]
    TARGET = "risk_class"

    def __init__(self, k: int = 5):
        self.k = k
        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(n_neighbors=k, weights='distance',
                                           metric='euclidean')
        self.trained = False
        self.metrics = {}

    def train(self, df: pd.DataFrame) -> Dict:
        X = df[self.FEATURES].values
        y = df[self.TARGET].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
        X_tr = self.scaler.fit_transform(X_tr)
        X_te = self.scaler.transform(X_te)

        t0 = time.perf_counter()
        self.model.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0

        y_pred = self.model.predict(X_te)
        self.metrics = self._compute_metrics(y_te, y_pred, train_time)
        self.trained = True
        return self.metrics

    def predict(self, features: Dict) -> Tuple[int, float]:
        """Predict risk class for a victim. Returns (class, probability)."""
        if not self.trained:
            return 1, 0.5
        x = np.array([[features.get(f, 0) for f in self.FEATURES]])
        x = self.scaler.transform(x)
        pred_class = int(self.model.predict(x)[0])
        proba = self.model.predict_proba(x)[0]
        return pred_class, float(proba[pred_class])

    def _compute_metrics(self, y_true, y_pred, train_time) -> Dict:
        return {
            "model": "kNN (k=5)",
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "train_time": train_time,
            "report": classification_report(y_true, y_pred,
                        target_names=RISK_LABELS, zero_division=0)
        }

    def save(self, path: str = MODEL_DIR):
        os.makedirs(path, exist_ok=True)
        joblib.dump((self.model, self.scaler), os.path.join(path, "knn_model.pkl"))

    def load(self, path: str = MODEL_DIR):
        fp = os.path.join(path, "knn_model.pkl")
        if os.path.exists(fp):
            self.model, self.scaler = joblib.load(fp)
            self.trained = True


class AreaRiskMLP:
    """MLP classifier predicting area risk severity (0=low,1=medium,2=high)."""

    FEATURES = ["fire_intensity", "road_stability", "aftershock_prob",
                 "structure_damage", "time_elapsed"]
    TARGET = "risk_class"

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.trained = False
        self.metrics = {}

    def train(self, df: pd.DataFrame) -> Dict:
        X = df[self.FEATURES].values
        y = df[self.TARGET].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
        X_tr = self.scaler.fit_transform(X_tr)
        X_te = self.scaler.transform(X_te)

        t0 = time.perf_counter()
        self.model.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0

        y_pred = self.model.predict(X_te)
        self.metrics = self._compute_metrics(y_te, y_pred, train_time)
        self.trained = True
        return self.metrics

    def predict(self, features: Dict) -> Tuple[int, float]:
        if not self.trained:
            return 1, 0.5
        x = np.array([[features.get(f, 0) for f in self.FEATURES]])
        x = self.scaler.transform(x)
        pred_class = int(self.model.predict(x)[0])
        proba = self.model.predict_proba(x)[0]
        return pred_class, float(proba[pred_class])

    def _compute_metrics(self, y_true, y_pred, train_time) -> Dict:
        return {
            "model": "MLPClassifier (64,32)",
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "train_time": train_time,
            "report": classification_report(y_true, y_pred,
                        target_names=RISK_LABELS, zero_division=0)
        }

    def save(self, path: str = MODEL_DIR):
        os.makedirs(path, exist_ok=True)
        joblib.dump((self.model, self.scaler), os.path.join(path, "mlp_model.pkl"))

    def load(self, path: str = MODEL_DIR):
        fp = os.path.join(path, "mlp_model.pkl")
        if os.path.exists(fp):
            self.model, self.scaler = joblib.load(fp)
            self.trained = True


class MLSystem:
    """Unified ML system integrating both models with the agent."""

    def __init__(self):
        self.knn = VictimRiskKNN()
        self.mlp = AreaRiskMLP()
        self.trained = False

    def train_all(self, victim_df: pd.DataFrame, area_df: pd.DataFrame) -> Dict:
        print("[ML] Training kNN model...")
        knn_metrics = self.knn.train(victim_df)
        print(f"    kNN Accuracy: {knn_metrics['accuracy']:.3f}")

        print("[ML] Training MLP model...")
        mlp_metrics = self.mlp.train(area_df)
        print(f"    MLP Accuracy: {mlp_metrics['accuracy']:.3f}")

        self.trained = True
        return {"knn": knn_metrics, "mlp": mlp_metrics}

    def assess_victim(self, victim, env_risk: float, fire_intensity: float,
                       road_stability: float, aftershock_prob: float,
                       rescue_delay: float = 0.0) -> Dict:
        """
        Assess a victim and return risk class + survival probability.
        This directly affects agent decisions:
         - HIGH risk → prioritize victim
         - LOW survival prob → urgent rescue
        """
        knn_features = {
            "severity": victim.severity,
            "condition": victim.condition,
            "fire_intensity": fire_intensity,
            "road_stability": road_stability,
            "aftershock_prob": aftershock_prob,
            "area_risk": env_risk,
            "rescue_delay": rescue_delay
        }
        risk_class, risk_conf = self.knn.predict(knn_features)

        mlp_features = {
            "fire_intensity": fire_intensity,
            "road_stability": road_stability,
            "aftershock_prob": aftershock_prob,
            "structure_damage": victim.condition,
            "time_elapsed": rescue_delay
        }
        area_class, area_conf = self.mlp.predict(mlp_features)

        # Survival probability: inverse of risk
        survival_prob = max(0.05, 1.0 - (risk_class / 3.0) - (victim.condition * 0.3))

        # Agent decision influence
        if risk_class == 2:
            priority = "HIGH PRIORITY — Rescue immediately"
            route_advice = "Avoid high-risk zones near victim"
        elif risk_class == 1:
            priority = "MEDIUM PRIORITY — Rescue soon"
            route_advice = "Balance speed and safety"
        else:
            priority = "LOW PRIORITY — Rescue after critical cases"
            route_advice = "Standard route acceptable"

        return {
            "victim_id": victim.vid,
            "risk_class": risk_class,
            "risk_label": RISK_LABELS[risk_class],
            "area_risk_class": area_class,
            "area_risk_label": RISK_LABELS[area_class],
            "survival_probability": round(survival_prob, 3),
            "priority": priority,
            "route_advice": route_advice,
        }

    def compare_models(self) -> str:
        if not self.trained:
            return "Models not yet trained."
        k = self.knn.metrics
        m = self.mlp.metrics
        return (
            f"MODEL COMPARISON:\n"
            f"kNN:  acc={k['accuracy']:.3f}, P={k['precision']:.3f}, "
            f"R={k['recall']:.3f}, F1={k['f1']:.3f}\n"
            f"MLP:  acc={m['accuracy']:.3f}, P={m['precision']:.3f}, "
            f"R={m['recall']:.3f}, F1={m['f1']:.3f}\n"
            f"{'MLP' if m['f1'] > k['f1'] else 'kNN'} achieves higher F1-score."
        )
