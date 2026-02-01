#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ABSOLUTE-ZERO FULL RESEARCH PIPELINE (CPU-FRIENDLY, PAPER-READY) - FIXED VERSION (v2)
====================================================================================

This version fixes the error:

  ValueError: 'inverse_transform' works only when 'SimpleImputer' is instantiated
  with 'add_indicator=True'. Got 'add_indicator=False' instead.

Root cause:
- The previous script attempted to call `Pipeline.inverse_transform(...)` on a pipeline
  that includes `SimpleImputer(add_indicator=False)`. In scikit-learn, `SimpleImputer`
  only supports `inverse_transform` when `add_indicator=True`.

Correct fix (principled and CPU-safe):
- Remove ALL calls to `pre_num.inverse_transform(...)` (pipeline-level inverse).
- Use ONLY the scaler's inverse transform:
    scaler.inverse_transform(X_scaled)
  This returns values back to the imputed numeric space (original units after scaling),
  which is the correct and standard approach because imputation is not invertible in
  general (missing values cannot be recovered exactly).

Additionally, as requested, the dataset path and type are enforced:

  DATASET PATH (fixed):
    C:/Users/Asus.S510UNR/Desktop/TODO-INVESTIGACION/INVESTIGACION/ARTICULOS-AVANCES-2025/LIBRO-07-ABSOLUTE-ZERO/water_potability.csv

  DATASET TYPE (fixed):
    CSV

Usage:
  python code-absolute-zero-03.py --mode paper --stage all --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, wasserstein_distance, norm

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt

# Optional imbalanced-learn baselines
IMBLEARN_AVAILABLE = True
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
except Exception:
    IMBLEARN_AVAILABLE = False

# =============================================================================
# FIXED DATASET SETTINGS (as requested)
# =============================================================================
FIXED_DATASET_PATH = r"C:/Users/Asus.S510UNR/Desktop/TODO-INVESTIGACION/INVESTIGACION/ARTICULOS-AVANCES-2025/LIBRO-07-ABSOLUTE-ZERO/water_potability.csv"
FIXED_DATASET_TYPE = "csv"  # enforce CSV for the fixed dataset path


# =============================================================================
# 1) Reproducibility utilities
# =============================================================================

def set_global_seed(seed: int) -> None:
    """Set global seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")


def file_mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except Exception:
        return None


# =============================================================================
# 2) Logging
# =============================================================================

def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("AZR_FULL_PIPELINE")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# =============================================================================
# 3) Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    # Data
    dataset_path: str
    dataset_type: str  # "csv" or "excel" (enforced)
    target_col: str

    # Splits
    test_size: float = 0.25
    val_size: float = 0.20
    random_seed: int = 42

    # Column handling
    treat_low_card_int_as_categorical: bool = True
    low_cardinality_threshold: int = 12

    # Oversampling baselines
    run_smote: bool = True
    run_borderline_smote: bool = True
    run_adasyn: bool = True
    smote_k_neighbors: int = 5
    borderline_smote_k_neighbors: int = 5
    adasyn_n_neighbors: int = 5

    # AZR generator settings
    azr_rounds: int = 8
    azr_candidates_per_round: int = 6
    azr_kmeans_clusters: int = 3
    azr_quantile_low: float = 0.01
    azr_quantile_high: float = 0.99
    azr_curriculum_tol_start: float = 0.02
    azr_curriculum_tol_end: float = 0.10
    azr_tail_jitter_strength: float = 0.010
    azr_reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "fidelity": 0.35,
        "structure": 0.25,
        "utility": 0.30,
        "verification": 0.10
    })

    # Predictive models
    models_to_run: List[str] = field(default_factory=lambda: ["LR", "RF", "GBC", "HGBT"])

    # Repeated runs for robustness
    n_runs: int = 5
    seeds_offset: int = 1000

    # Near-duplicate / duplicate checks
    duplicate_check: bool = True
    near_duplicate_check: bool = True
    near_duplicate_sample_size: int = 600
    near_duplicate_threshold: float = 1e-3

    # Plots
    figure_dpi: int = 200
    max_features_for_distribution_plots: int = 8
    max_features_for_qq_plots: int = 6

    # Output root
    outdir: str = "./outputs_run"

    # Orchestration
    mode: str = "paper"  # quick|paper
    stage: str = "all"   # preprocess|generate|evaluate|plot|report|all
    force: bool = False  # ignore cache


def make_config(mode: str,
                dataset_path: str,
                dataset_type: str,
                target_col: str,
                outdir: str,
                seed: int,
                stage: str,
                force: bool) -> PipelineConfig:
    if mode not in ["quick", "paper"]:
        mode = "paper"

    # Enforce fixed dataset settings if user uses the fixed path or leaves it empty
    if (not dataset_path) or (dataset_path.strip() == "") or (dataset_path.strip() == FIXED_DATASET_PATH):
        dataset_path = FIXED_DATASET_PATH
        dataset_type = FIXED_DATASET_TYPE

    dataset_type = (dataset_type or "").strip().lower()
    if dataset_type not in ["csv", "excel"]:
        dataset_type = "csv"

    cfg = PipelineConfig(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        target_col=target_col,
        outdir=outdir,
        random_seed=seed,
        mode=mode,
        stage=stage,
        force=force
    )

    if mode == "quick":
        cfg.n_runs = 2
        cfg.azr_rounds = 4
        cfg.azr_candidates_per_round = 3
        cfg.models_to_run = ["LR", "RF"]
        cfg.figure_dpi = 160
        cfg.max_features_for_distribution_plots = 6
        cfg.max_features_for_qq_plots = 4
        cfg.near_duplicate_sample_size = 350

    return cfg


# =============================================================================
# 4) Project architecture (folders + cache)
# =============================================================================

@dataclass
class ProjectPaths:
    root: Path
    data_raw: Path
    data_processed: Path
    data_synthetic: Path
    outputs: Path
    outputs_tables: Path
    outputs_figures: Path
    outputs_logs: Path
    outputs_reports: Path
    outputs_models: Path
    cache: Path

    @staticmethod
    def from_outdir(outdir: str) -> "ProjectPaths":
        root = Path(outdir).resolve()
        return ProjectPaths(
            root=root,
            data_raw=root / "data" / "raw",
            data_processed=root / "data" / "processed",
            data_synthetic=root / "data" / "synthetic",
            outputs=root / "outputs",
            outputs_tables=root / "outputs" / "tables",
            outputs_figures=root / "outputs" / "figures",
            outputs_logs=root / "outputs" / "logs",
            outputs_reports=root / "outputs" / "reports",
            outputs_models=root / "outputs" / "models",
            cache=root / "cache"
        )

    def ensure_all(self) -> None:
        for p in [
            self.data_raw, self.data_processed, self.data_synthetic,
            self.outputs_tables, self.outputs_figures, self.outputs_logs,
            self.outputs_reports, self.outputs_models, self.cache
        ]:
            ensure_dir(p)


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=None, separators=(",", ":"))


def compute_run_hash(cfg: PipelineConfig, dataset_mtime: Optional[float]) -> str:
    payload = asdict(cfg)
    payload["dataset_mtime"] = dataset_mtime
    raw = stable_json_dumps(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def should_skip_stage(manifest: Dict[str, Any], stage: str, run_hash: str) -> bool:
    s = manifest.get("stages", {}).get(stage, {})
    return (s.get("run_hash") == run_hash) and bool(s.get("completed", False))


def mark_stage_done(manifest: Dict[str, Any], stage: str, run_hash: str, artifacts: Dict[str, str]) -> Dict[str, Any]:
    manifest.setdefault("stages", {})
    manifest["stages"][stage] = {
        "run_hash": run_hash,
        "completed": True,
        "timestamp": utc_timestamp(),
        "artifacts": artifacts
    }
    return manifest


# =============================================================================
# 5) Data ingestion + typing
# =============================================================================

def load_or_make_dataset(dataset_path: str, dataset_type: str, target_col: str, seed: int) -> pd.DataFrame:
    """
    Load the dataset with enforced type.
    For the requested fixed dataset path, dataset_type is enforced as 'csv' and loaded via pd.read_csv.
    If the file does not exist, a toy dataset is created so the pipeline remains runnable.
    """
    set_global_seed(seed)
    p = Path(dataset_path)

    if p.exists():
        if dataset_type == "csv":
            df = pd.read_csv(p)  # <-- ACTUAL LOAD LINE FOR THE WATER DATASET
        elif dataset_type == "excel":
            df = pd.read_excel(p)
        else:
            df = pd.read_csv(p)

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' was not found in the dataset.")
        return df

    # Fallback toy dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=2600,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.65, 0.35],
        class_sep=1.1,
        random_state=seed
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    df[target_col] = y

    miss = 0.06
    mask = np.random.rand(*df.drop(columns=[target_col]).shape) < miss
    df.loc[:, df.columns != target_col] = df.loc[:, df.columns != target_col].mask(mask)
    return df


def infer_column_types(
    df: pd.DataFrame,
    target_col: str,
    treat_low_card_int_as_categorical: bool,
    low_cardinality_threshold: int
) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = []
    categorical_cols = []

    for c in feature_cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            if treat_low_card_int_as_categorical and pd.api.types.is_integer_dtype(s):
                nunq = s.dropna().nunique()
                if nunq <= low_cardinality_threshold:
                    categorical_cols.append(c)
                else:
                    numeric_cols.append(c)
            else:
                numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def stratified_splits(df: pd.DataFrame, target_col: str, seed: int, test_size: float, val_size: float
                      ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[target_col]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    tr_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed + 1,
        stratify=train_df[target_col]
    )
    return tr_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# =============================================================================
# 6) Metrics (fidelity + structure)
# =============================================================================

def ks_w1_per_feature(real: np.ndarray, synth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ks_vals = []
    w1_vals = []
    for j in range(real.shape[1]):
        r = real[:, j]
        s = synth[:, j]
        r = r[np.isfinite(r)]
        s = s[np.isfinite(s)]
        if len(r) < 2 or len(s) < 2:
            ks_vals.append(np.nan)
            w1_vals.append(np.nan)
            continue
        ks_vals.append(ks_2samp(r, s).statistic)
        w1_vals.append(wasserstein_distance(r, s))
    return np.array(ks_vals, dtype=float), np.array(w1_vals, dtype=float)


def safe_corr(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[0] < 3:
        return np.zeros((X.shape[1], X.shape[1]), dtype=float)
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    return C


def frobenius_distance(C_real: np.ndarray, C_synth: np.ndarray) -> float:
    return float(np.linalg.norm(C_real - C_synth, ord="fro"))


def sign_matching(C_real: np.ndarray, C_synth: np.ndarray) -> float:
    n = C_real.shape[0]
    if n <= 1:
        return 1.0
    mask = ~np.eye(n, dtype=bool)
    sr = np.sign(C_real[mask])
    ss = np.sign(C_synth[mask])
    return float(np.mean(sr == ss))


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x)
    mn, mx = float(np.min(finite)), float(np.max(finite))
    if abs(mx - mn) < 1e-12:
        return np.zeros_like(x)
    y = (x - mn) / (mx - mn)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


# =============================================================================
# 7) Predictive evaluation (TRTR, TSTR, TSTR+)
# =============================================================================

def make_models(model_keys: List[str], seed: int) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for k in model_keys:
        if k == "LR":
            models[k] = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
        elif k == "RF":
            models[k] = RandomForestClassifier(n_estimators=350, random_state=seed, n_jobs=-1)
        elif k == "GBC":
            models[k] = GradientBoostingClassifier(random_state=seed)
        elif k == "HGBT":
            models[k] = HistGradientBoostingClassifier(random_state=seed)
        else:
            raise ValueError(f"Unknown model key '{k}'. Use one of: LR, RF, GBC, HGBT.")
    return models


def predict_proba_1(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return (s - s.min()) / (s.max() - s.min() + 1e-12)
    y = model.predict(X).astype(float)
    return y


def auc_and_roc(model: Any, X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    model.fit(X_train, y_train)
    p = predict_proba_1(model, X_test)
    auc = float(roc_auc_score(y_test, p))
    fpr, tpr, _ = roc_curve(y_test, p)
    return auc, fpr, tpr


# =============================================================================
# 8) Validations + consistency checks
# =============================================================================

def validate_no_nan_inf(X: np.ndarray) -> Dict[str, Any]:
    return {
        "has_nan": bool(np.isnan(X).any()),
        "has_inf": bool(np.isinf(X).any()),
        "nan_count": int(np.isnan(X).sum()),
        "inf_count": int(np.isinf(X).sum())
    }


def duplicate_rate(X: np.ndarray) -> float:
    Xr = np.round(X, 12)
    hashes = pd.util.hash_pandas_object(pd.DataFrame(Xr), index=False).to_numpy()
    return float(1.0 - (np.unique(hashes).size / hashes.size))


def near_duplicate_rate(X: np.ndarray, sample_size: int, threshold: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < 5:
        return 0.0
    m = min(sample_size, n)
    idx = rng.choice(n, size=m, replace=False)
    S = X[idx]
    G = S @ S.T
    norms = np.sum(S * S, axis=1, keepdims=True)
    D2 = norms + norms.T - 2.0 * G
    D2 = np.maximum(D2, 0.0)
    np.fill_diagonal(D2, np.inf)
    min_d = np.sqrt(np.min(D2, axis=1))
    rate = float(np.mean(min_d < threshold))
    return rate


def quantile_range_compliance(X_synth: np.ndarray, X_ref: np.ndarray, q_low: float, q_high: float, tol: float) -> float:
    scores = []
    for j in range(X_ref.shape[1]):
        r = X_ref[:, j]
        r = r[np.isfinite(r)]
        if r.size < 20:
            scores.append(1.0)
            continue
        ql = np.quantile(r, q_low)
        qh = np.quantile(r, q_high)
        span = (qh - ql) + 1e-12
        lo = ql - tol * span
        hi = qh + tol * span
        s = X_synth[:, j]
        s = s[np.isfinite(s)]
        if s.size == 0:
            scores.append(0.0)
            continue
        scores.append(float(np.mean((s >= lo) & (s <= hi))))
    return float(np.mean(scores))


# =============================================================================
# 9) Internal CPU oversamplers (fallbacks if imblearn is missing)
# =============================================================================

def _smote_generate(X_min: np.ndarray, n_new: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if X_min.shape[0] < 2:
        base = X_min[0:1] if X_min.shape[0] == 1 else np.zeros((1, X_min.shape[1]))
        noise = rng.normal(0.0, 0.01, size=(n_new, base.shape[1]))
        return base.repeat(n_new, axis=0) + noise

    k_eff = min(k, max(1, X_min.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min)
    neigh = nn.kneighbors(X_min, return_distance=False)[:, 1:]

    idx_base = rng.integers(0, X_min.shape[0], size=n_new)
    idx_nei = np.array([rng.choice(neigh[i]) for i in idx_base], dtype=int)
    lam = rng.random(size=n_new).reshape(-1, 1)

    X_new = X_min[idx_base] + lam * (X_min[idx_nei] - X_min[idx_base])
    return X_new


def _borderline_mask(X: np.ndarray, y: np.ndarray, cls: Any, k: int) -> np.ndarray:
    if X.shape[0] < (k + 2):
        return np.ones(int(np.sum(y == cls)), dtype=bool)

    k_eff = min(k, max(1, X.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X)
    X_min = X[y == cls]
    neigh = nn.kneighbors(X_min, return_distance=False)[:, 1:]
    y_nei = y[neigh]
    maj_frac = np.mean(y_nei != cls, axis=1)
    mask = (maj_frac >= 0.5) & (maj_frac < 1.0)
    if not np.any(mask):
        mask[:] = True
    return mask


def _adasyn_allocation(X: np.ndarray, y: np.ndarray, cls: Any, k: int, n_new: int) -> np.ndarray:
    X_min = X[y == cls]
    if X.shape[0] < (k + 2) or X_min.shape[0] < 2:
        base = np.full(X_min.shape[0], 1.0 / max(1, X_min.shape[0]), dtype=float)
    else:
        k_eff = min(k, max(1, X.shape[0] - 1))
        nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X)
        neigh = nn.kneighbors(X_min, return_distance=False)[:, 1:]
        y_nei = y[neigh]
        g = np.mean(y_nei != cls, axis=1)
        if np.sum(g) <= 1e-12:
            base = np.full_like(g, 1.0 / g.size)
        else:
            base = g / np.sum(g)

    alloc = np.floor(base * n_new).astype(int)
    rem = int(n_new - alloc.sum())
    if rem > 0:
        order = np.argsort(-base)
        for i in range(rem):
            alloc[order[i % order.size]] += 1
    return alloc


def oversample_fallback(method: str,
                        X: np.ndarray,
                        y: np.ndarray,
                        seed: int,
                        smote_k: int,
                        bsmote_k: int,
                        adasyn_k: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    maxc = int(np.max(counts))

    X_out = [X]
    y_out = [y]

    for cls, cnt in zip(classes, counts):
        cnt = int(cnt)
        if cnt >= maxc:
            continue
        n_new = maxc - cnt
        X_min = X[y == cls]
        if X_min.shape[0] < 1:
            continue

        if method == "SMOTE":
            X_new = _smote_generate(X_min, n_new=n_new, k=smote_k, seed=seed + int(hash(cls)) % 99991)
        elif method == "BorderlineSMOTE":
            mask = _borderline_mask(X, y, cls=cls, k=bsmote_k)
            X_border = X_min[mask]
            if X_border.shape[0] < 2:
                X_border = X_min
            X_new = _smote_generate(X_border, n_new=n_new, k=bsmote_k, seed=seed + 17 + int(hash(cls)) % 99991)
        elif method == "ADASYN":
            alloc = _adasyn_allocation(X, y, cls=cls, k=adasyn_k, n_new=n_new)
            if X_min.shape[0] < 2:
                X_new = _smote_generate(X_min, n_new=n_new, k=1, seed=seed + 33)
            else:
                k_eff = min(adasyn_k, max(1, X_min.shape[0] - 1))
                nn_min = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min)
                neigh = nn_min.kneighbors(X_min, return_distance=False)[:, 1:]
                synth_parts = []
                for i, ni in enumerate(alloc):
                    if ni <= 0:
                        continue
                    base = np.repeat(X_min[i:i+1], repeats=ni, axis=0)
                    nei_idx = rng.choice(neigh[i], size=ni, replace=True)
                    lam = rng.random(size=ni).reshape(-1, 1)
                    part = base + lam * (X_min[nei_idx] - base)
                    synth_parts.append(part)
                if synth_parts:
                    X_new = np.vstack(synth_parts)
                else:
                    X_new = _smote_generate(X_min, n_new=n_new, k=adasyn_k, seed=seed + 55)

                if X_new.shape[0] > n_new:
                    X_new = X_new[:n_new]
                elif X_new.shape[0] < n_new:
                    extra = _smote_generate(X_min, n_new=n_new - X_new.shape[0], k=adasyn_k, seed=seed + 77)
                    X_new = np.vstack([X_new, extra])
        else:
            raise ValueError("Unknown oversampling method for fallback.")

        X_out.append(X_new)
        y_out.append(np.full(X_new.shape[0], cls, dtype=y.dtype))

    X_res = np.vstack(X_out)
    y_res = np.concatenate(y_out)
    return X_res, y_res


def oversample(method: str,
               X: np.ndarray,
               y: np.ndarray,
               seed: int,
               smote_k: int,
               bsmote_k: int,
               adasyn_k: int) -> Tuple[np.ndarray, np.ndarray]:
    if IMBLEARN_AVAILABLE:
        if method == "SMOTE":
            sampler = SMOTE(random_state=seed, k_neighbors=smote_k)
        elif method == "BorderlineSMOTE":
            sampler = BorderlineSMOTE(random_state=seed, k_neighbors=bsmote_k)
        elif method == "ADASYN":
            sampler = ADASYN(random_state=seed, n_neighbors=adasyn_k)
        else:
            raise ValueError("Unknown oversampling method.")
        X_res, y_res = sampler.fit_resample(X, y)
        return X_res, y_res

    return oversample_fallback(method, X, y, seed, smote_k, bsmote_k, adasyn_k)


# =============================================================================
# 10) FIX: Safe inverse scaling helper (no SimpleImputer inverse_transform)
# =============================================================================

def inverse_scale_only(pre_num: Pipeline, X_scaled: np.ndarray) -> np.ndarray:
    """
    Convert scaled numeric arrays back to original numeric units using ONLY the scaler inverse.
    This avoids calling SimpleImputer.inverse_transform (which is unsupported unless add_indicator=True).
    """
    if "scaler" not in pre_num.named_steps:
        return np.asarray(X_scaled, dtype=float)
    scaler = pre_num.named_steps["scaler"]
    return scaler.inverse_transform(np.asarray(X_scaled, dtype=float))


# =============================================================================
# 11) AZR-style generator (mixture of Gaussian copulas + verifiable curriculum)
# =============================================================================

class EmpiricalQuantileMap:
    def __init__(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 20:
            x = np.linspace(-1, 1, 80)
        self.sorted_x = np.sort(x)
        self.n = self.sorted_x.size

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(self.sorted_x, x, side="right")
        u = idx / (self.n + 1.0)
        return np.clip(u, 1e-6, 1 - 1e-6)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        u = np.clip(u, 1e-6, 1 - 1e-6)
        q = u * (self.n - 1)
        lo = np.floor(q).astype(int)
        hi = np.ceil(q).astype(int)
        frac = q - lo
        x_lo = self.sorted_x[lo]
        x_hi = self.sorted_x[hi]
        return (1 - frac) * x_lo + frac * x_hi


@dataclass
class CopulaModel:
    qmaps: List[EmpiricalQuantileMap]
    corr: np.ndarray

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        d = len(self.qmaps)
        Z = rng.multivariate_normal(np.zeros(d), self.corr, size=n)
        U = norm.cdf(Z)
        X = np.zeros_like(U)
        for j in range(d):
            X[:, j] = self.qmaps[j].ppf(U[:, j])
        return X


@dataclass
class MixtureCopula:
    copulas: List[CopulaModel]
    weights: np.ndarray

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        w = np.asarray(self.weights, dtype=float)
        w = w / (w.sum() + 1e-12)
        counts = rng.multinomial(n, w)
        parts = []
        for k, ck in enumerate(counts):
            if ck <= 0:
                continue
            parts.append(self.copulas[k].sample(ck, rng))
        if not parts:
            return self.copulas[0].sample(n, rng)
        X = np.vstack(parts)
        rng.shuffle(X)
        return X


def fit_gaussian_copula(X: np.ndarray) -> CopulaModel:
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    qmaps = [EmpiricalQuantileMap(X[:, j]) for j in range(d)]
    U = np.column_stack([qmaps[j].cdf(X[:, j]) for j in range(d)])
    Z = norm.ppf(U)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.corrcoef(Z, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = (C + C.T) / 2.0
    C = C + 1e-6 * np.eye(d)
    return CopulaModel(qmaps=qmaps, corr=C)


def fit_mixture_copula(X: np.ndarray, k: int, seed: int) -> MixtureCopula:
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n < max(60, 12 * k):
        cm = fit_gaussian_copula(X)
        return MixtureCopula([cm], np.array([1.0], dtype=float))

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)

    copulas: List[CopulaModel] = []
    weights: List[float] = []
    for cl in range(k):
        Xc = X[labels == cl]
        if Xc.shape[0] < 35:
            continue
        copulas.append(fit_gaussian_copula(Xc))
        weights.append(float(Xc.shape[0]))

    if not copulas:
        cm = fit_gaussian_copula(X)
        return MixtureCopula([cm], np.array([1.0], dtype=float))

    return MixtureCopula(copulas=copulas, weights=np.array(weights, dtype=float))


def quantile_map_to_reference(X_synth: np.ndarray, X_ref: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    X_synth = np.asarray(X_synth, dtype=float)
    X_ref = np.asarray(X_ref, dtype=float)
    out = X_synth.copy()
    qs = np.linspace(q_low, q_high, 121)

    for j in range(X_synth.shape[1]):
        r = X_ref[:, j]
        s = X_synth[:, j]
        r = r[np.isfinite(r)]
        s = s[np.isfinite(s)]
        if r.size < 30 or s.size < 30:
            continue
        rq = np.quantile(r, qs)
        sq = np.quantile(s, qs)
        sq = np.maximum.accumulate(sq)
        out[:, j] = np.interp(X_synth[:, j], sq, rq, left=rq[0], right=rq[-1])

    return out


def tail_jitter(X: np.ndarray, X_ref: np.ndarray, strength: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    out = X.copy()
    for j in range(out.shape[1]):
        r = X_ref[:, j]
        r = r[np.isfinite(r)]
        if r.size < 30:
            continue
        iqr = np.quantile(r, 0.75) - np.quantile(r, 0.25)
        sigma = max(iqr, 1e-6)
        ranks = pd.Series(out[:, j]).rank(pct=True).to_numpy()
        tail_w = np.abs(ranks - 0.5) * 2.0
        noise = rng.normal(0.0, strength * sigma, size=out.shape[0]) * tail_w
        out[:, j] = out[:, j] + noise
    return out


def azr_generate(
    X_train_num: np.ndarray,
    y_train: np.ndarray,
    X_val_num: np.ndarray,
    y_val: np.ndarray,
    minority_class: Any,
    n_needed: int,
    seed: int,
    rounds: int,
    candidates_per_round: int,
    k_clusters: int,
    q_low: float,
    q_high: float,
    tol_start: float,
    tol_end: float,
    tail_strength: float,
    reward_w: Dict[str, float]
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    rng = np.random.default_rng(seed)

    X_min = X_train_num[y_train == minority_class]
    if X_min.shape[0] < 60:
        X_min = X_train_num.copy()

    mix = fit_mixture_copula(X_min, k=k_clusters, seed=seed)
    quick_model = LogisticRegression(max_iter=1500, solver="lbfgs", random_state=seed + 99)

    hist_rows = []
    best_global = None
    best_global_reward = -1e18
    best_global_metrics: Dict[str, Any] = {}

    for r in range(rounds):
        tol = tol_start + (tol_end - tol_start) * (r / max(1, rounds - 1))

        best_round_reward = -1e18
        best_round_batch = None
        best_round_metrics = {}

        for c in range(candidates_per_round):
            Xs = mix.sample(n_needed, rng=rng)
            Xs = quantile_map_to_reference(Xs, X_min, q_low=q_low, q_high=q_high)
            Xs = tail_jitter(Xs, X_min, strength=tail_strength, seed=seed + 1000 * r + 10 * c)

            range_score = quantile_range_compliance(Xs, X_min, q_low=q_low, q_high=q_high, tol=tol)
            C_r = safe_corr(X_min)
            C_s = safe_corr(Xs)
            signm = sign_matching(C_r, C_s)
            verification = 0.60 * range_score + 0.40 * signm

            ks_vals, w1_vals = ks_w1_per_feature(X_min, Xs)
            ks_avg = float(np.nanmean(ks_vals))
            w1_avg = float(np.nanmean(w1_vals))

            frob = float(frobenius_distance(C_r, C_s))

            X_aug = np.vstack([X_train_num, Xs])
            y_aug = np.concatenate([y_train, np.full(Xs.shape[0], minority_class, dtype=y_train.dtype)])
            try:
                auc_val, _, _ = auc_and_roc(quick_model, X_aug, y_aug, X_val_num, y_val)
            except Exception:
                auc_val = 0.0

            fidelity_term = 0.6 * (1.0 - ks_avg) + 0.4 * (1.0 / (1.0 + w1_avg))
            structure_term = 0.6 * signm + 0.4 * (1.0 / (1.0 + frob))
            utility_term = auc_val
            verification_term = verification

            reward = (
                reward_w["fidelity"] * fidelity_term
                + reward_w["structure"] * structure_term
                + reward_w["utility"] * utility_term
                + reward_w["verification"] * verification_term
            )

            if reward > best_round_reward:
                best_round_reward = reward
                best_round_batch = Xs
                best_round_metrics = {
                    "round": r,
                    "candidate": c,
                    "tolerance": float(tol),
                    "reward": float(reward),
                    "verification": float(verification),
                    "range_score": float(range_score),
                    "signmatch": float(signm),
                    "ks_avg": float(ks_avg),
                    "w1_avg": float(w1_avg),
                    "frob": float(frob),
                    "auc_val": float(auc_val)
                }

        hist_rows.append(best_round_metrics)

        if best_round_reward > best_global_reward:
            best_global_reward = best_round_reward
            best_global = best_round_batch
            best_global_metrics = best_round_metrics

    hist_df = pd.DataFrame(hist_rows)

    if best_global is None:
        mu = np.mean(X_min, axis=0)
        sd = np.std(X_min, axis=0) + 1e-6
        best_global = rng.normal(mu, sd, size=(n_needed, X_min.shape[1]))
        best_global_metrics = {"fallback": True, "reason": "No best batch selected; used Gaussian fallback."}

    return best_global, hist_df, best_global_metrics


# =============================================================================
# 12) Categorical synthesis (class-conditional frequency model)
# =============================================================================

def fit_categorical_distributions(df: pd.DataFrame, cat_cols: List[str], target_col: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    dists: Dict[str, Dict[str, Dict[str, float]]] = {}
    classes = df[target_col].dropna().unique().tolist()
    for cl in classes:
        cl_key = str(cl)
        dists[cl_key] = {}
        sub = df[df[target_col] == cl]
        for c in cat_cols:
            vc = sub[c].fillna("<<MISSING>>").value_counts(normalize=True)
            dists[cl_key][c] = {str(k): float(v) for k, v in vc.items()}
    return dists


def sample_categorical(dists: Dict[str, Dict[str, Dict[str, float]]], cl: Any, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cl_key = str(cl)
    cols = list(dists[cl_key].keys())
    out = {}
    for c in cols:
        items = list(dists[cl_key][c].items())
        cats = [k for k, _ in items]
        probs = np.array([v for _, v in items], dtype=float)
        probs = probs / (probs.sum() + 1e-12)
        draws = rng.choice(cats, size=n, replace=True, p=probs)
        draws = pd.Series(draws).replace("<<MISSING>>", np.nan).to_numpy()
        out[c] = draws
    return pd.DataFrame(out)


# =============================================================================
# 13) Visualization utilities
# =============================================================================

def save_roc_plot(curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
                  title: str, outpath: Path, dpi: int) -> None:
    plt.figure()
    for name, (fpr, tpr, auc) in curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()


def plot_corr_heatmap(C: np.ndarray, title: str, outpath: Path, dpi: int) -> None:
    plt.figure()
    plt.imshow(C, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()


def plot_distributions(real: np.ndarray, synth: np.ndarray, feature_names: List[str], outpath: Path, dpi: int) -> None:
    nfeat = real.shape[1]
    ncols = 2
    nrows = int(math.ceil(nfeat / ncols))
    plt.figure(figsize=(10, 4 * nrows))
    for j in range(nfeat):
        plt.subplot(nrows, ncols, j + 1)
        r = real[:, j]
        s = synth[:, j]
        r = r[np.isfinite(r)]
        s = s[np.isfinite(s)]
        plt.hist(r, bins=30, alpha=0.55, density=True, label="Real")
        plt.hist(s, bins=30, alpha=0.55, density=True, label="Synthetic")
        plt.title(feature_names[j])
        plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()


def plot_qq(real: np.ndarray, synth: np.ndarray, feature_names: List[str], outpath: Path, dpi: int) -> None:
    nfeat = real.shape[1]
    ncols = 2
    nrows = int(math.ceil(nfeat / ncols))
    plt.figure(figsize=(10, 4 * nrows))
    qs = np.linspace(0.01, 0.99, 99)
    for j in range(nfeat):
        plt.subplot(nrows, ncols, j + 1)
        r = real[:, j]
        s = synth[:, j]
        r = r[np.isfinite(r)]
        s = s[np.isfinite(s)]
        if r.size < 30 or s.size < 30:
            plt.text(0.1, 0.5, "Insufficient data", fontsize=10)
            plt.title(feature_names[j])
            continue
        rq = np.quantile(r, qs)
        sq = np.quantile(s, qs)
        plt.plot(rq, sq, marker="o", linestyle="", markersize=3)
        lo = min(rq.min(), sq.min())
        hi = max(rq.max(), sq.max())
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        plt.title(f"QQ: {feature_names[j]}")
        plt.xlabel("Real quantiles")
        plt.ylabel("Synthetic quantiles")
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()


# =============================================================================
# 14) Pipeline stages
# =============================================================================

def stage_preprocess(cfg: PipelineConfig, paths: ProjectPaths, logger: logging.Logger) -> Dict[str, Any]:
    dataset_path = Path(cfg.dataset_path)
    df = load_or_make_dataset(cfg.dataset_path, cfg.dataset_type, cfg.target_col, cfg.random_seed)
    logger.info(f"[Preprocess] Loaded dataset shape: {df.shape}. Target: {cfg.target_col}")
    logger.info(f"[Preprocess] Dataset path enforced: {cfg.dataset_path}")
    logger.info(f"[Preprocess] Dataset type enforced: {cfg.dataset_type.upper()}")

    # Save raw snapshot if dataset exists
    if dataset_path.exists():
        raw_copy = paths.data_raw / f"raw_snapshot_{utc_timestamp()}{dataset_path.suffix.lower()}"
        try:
            if cfg.dataset_type == "csv":
                df.to_csv(raw_copy, index=False)
            else:
                df.to_excel(raw_copy, index=False)
            logger.info(f"[Preprocess] Raw snapshot saved: {raw_copy}")
        except Exception as e:
            logger.warning(f"[Preprocess] Could not save raw snapshot: {repr(e)}")

    numeric_cols, categorical_cols = infer_column_types(
        df, cfg.target_col,
        cfg.treat_low_card_int_as_categorical,
        cfg.low_cardinality_threshold
    )

    if len(numeric_cols) < 1 and not categorical_cols:
        raise ValueError("Insufficient features: require at least 1 numeric feature or categorical features.")

    tr_df, val_df, test_df = stratified_splits(
        df, cfg.target_col, cfg.random_seed, cfg.test_size, cfg.val_size
    )

    tr_path = paths.data_processed / "train.csv"
    val_path = paths.data_processed / "val.csv"
    test_path = paths.data_processed / "test.csv"
    tr_df.to_csv(tr_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    cat_dists_path = None
    if categorical_cols:
        cat_dists = fit_categorical_distributions(pd.concat([tr_df, val_df], axis=0), categorical_cols, cfg.target_col)
        cat_dists_path = paths.data_processed / "categorical_distributions.json"
        cat_dists_path.write_text(json.dumps(cat_dists, indent=2), encoding="utf-8")

    meta = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "train_path": str(tr_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
        "categorical_distributions_path": str(cat_dists_path) if cat_dists_path else None
    }
    return meta


def stage_generate(cfg: PipelineConfig, paths: ProjectPaths, logger: logging.Logger, meta: Dict[str, Any]) -> Dict[str, Any]:
    tr_df = pd.read_csv(meta["train_path"])
    val_df = pd.read_csv(meta["val_path"])
    numeric_cols = meta["numeric_cols"]
    cat_cols = meta["categorical_cols"]

    y_tr_raw = tr_df[cfg.target_col].to_numpy()
    classes, counts = np.unique(y_tr_raw, return_counts=True)
    minority = classes[np.argmin(counts)]
    majority = classes[np.argmax(counts)]
    n_needed = int(np.max(counts) - np.min(counts))
    if n_needed <= 0:
        n_needed = max(200, int(0.15 * tr_df.shape[0]))

    logger.info(f"[Generate] Train class distribution: {dict(zip(classes.tolist(), counts.tolist()))}")
    logger.info(f"[Generate] Minority={minority}, Majority={majority}, n_needed_for_balance={n_needed}")
    logger.info(f"[Generate] imblearn available: {IMBLEARN_AVAILABLE} (fallback oversamplers enabled if False)")

    pre_num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    if numeric_cols:
        X_tr_num = pre_num.fit_transform(tr_df[numeric_cols])
        X_val_num = pre_num.transform(val_df[numeric_cols])
    else:
        X_tr_num = np.zeros((tr_df.shape[0], 1))
        X_val_num = np.zeros((val_df.shape[0], 1))

    y_tr = tr_df[cfg.target_col].to_numpy()
    y_val = val_df[cfg.target_col].to_numpy()

    cat_dists = None
    if cat_cols:
        dpath = meta.get("categorical_distributions_path")
        if dpath and Path(dpath).exists():
            cat_dists = json.loads(Path(dpath).read_text(encoding="utf-8"))
        else:
            cat_dists = fit_categorical_distributions(pd.concat([tr_df, val_df], axis=0), cat_cols, cfg.target_col)

    methods = []
    if cfg.run_smote:
        methods.append("SMOTE")
    if cfg.run_borderline_smote:
        methods.append("BorderlineSMOTE")
    if cfg.run_adasyn:
        methods.append("ADASYN")
    methods.append("AZR")

    for m in methods:
        ensure_dir(paths.data_synthetic / m)

    runs_info: List[Dict[str, Any]] = []

    for run_i in range(cfg.n_runs):
        seed_run = cfg.random_seed + cfg.seeds_offset * (run_i + 1)
        set_global_seed(seed_run)

        # -------------------------
        # Baseline oversamplers
        # -------------------------
        for m in ["SMOTE", "BorderlineSMOTE", "ADASYN"]:
            if m not in methods:
                continue
            try:
                X_res, y_res = oversample(
                    m, X_tr_num, y_tr, seed_run,
                    cfg.smote_k_neighbors, cfg.borderline_smote_k_neighbors, cfg.adasyn_n_neighbors
                )

                # FIX: Only inverse scaling (no SimpleImputer inverse_transform)
                X_res_num_orig = inverse_scale_only(pre_num, X_res) if numeric_cols else X_res
                df_num = pd.DataFrame(X_res_num_orig, columns=numeric_cols if numeric_cols else ["dummy_num"])
                df_out = df_num.copy()

                if cat_cols and cat_dists is not None:
                    cat_out_rows = []
                    for cl in y_res:
                        tmp_df = sample_categorical(cat_dists, cl, n=1, seed=seed_run + 13 + int(hash(cl)) % 99991)
                        cat_out_rows.append(tmp_df.iloc[0])
                    cat_out = pd.DataFrame(cat_out_rows).reset_index(drop=True)
                    for cc in cat_cols:
                        df_out[cc] = cat_out[cc].values

                df_out[cfg.target_col] = y_res
                outpath = paths.data_synthetic / m / f"run_{run_i+1:02d}.csv"
                df_out.to_csv(outpath, index=False)
                runs_info.append({"run": run_i + 1, "method": m, "path": str(outpath), "seed": seed_run})

            except Exception as e:
                logger.warning(f"[Generate] {m} failed in run {run_i+1}: {repr(e)}. Creating fallback dataset.")
                rng = np.random.default_rng(seed_run + 999)
                idx = rng.choice(X_tr_num.shape[0], size=X_tr_num.shape[0], replace=True)
                X_fb = X_tr_num[idx] + rng.normal(0.0, 0.02, size=X_tr_num[idx].shape)
                y_fb = y_tr[idx]

                # FIX: Only inverse scaling
                X_fb_orig = inverse_scale_only(pre_num, X_fb) if numeric_cols else X_fb
                df_out = pd.DataFrame(X_fb_orig, columns=numeric_cols if numeric_cols else ["dummy_num"])

                if cat_cols and cat_dists is not None:
                    cat_out_rows = []
                    for cl in y_fb:
                        tmp_df = sample_categorical(cat_dists, cl, n=1, seed=seed_run + 123)
                        cat_out_rows.append(tmp_df.iloc[0])
                    cat_out = pd.DataFrame(cat_out_rows).reset_index(drop=True)
                    for cc in cat_cols:
                        df_out[cc] = cat_out[cc].values

                df_out[cfg.target_col] = y_fb
                outpath = paths.data_synthetic / m / f"run_{run_i+1:02d}.csv"
                df_out.to_csv(outpath, index=False)
                runs_info.append({"run": run_i + 1, "method": m, "path": str(outpath), "seed": seed_run})

        # -------------------------
        # AZR generator
        # -------------------------
        try:
            X_azr_min, hist_df, best_metrics = azr_generate(
                X_train_num=X_tr_num,
                y_train=y_tr,
                X_val_num=X_val_num,
                y_val=y_val,
                minority_class=minority,
                n_needed=n_needed,
                seed=seed_run,
                rounds=cfg.azr_rounds,
                candidates_per_round=cfg.azr_candidates_per_round,
                k_clusters=cfg.azr_kmeans_clusters,
                q_low=cfg.azr_quantile_low,
                q_high=cfg.azr_quantile_high,
                tol_start=cfg.azr_curriculum_tol_start,
                tol_end=cfg.azr_curriculum_tol_end,
                tail_strength=cfg.azr_tail_jitter_strength,
                reward_w=cfg.azr_reward_weights
            )

            n_min_real = int(np.sum(y_tr == minority))
            n_maj_real = int(np.sum(y_tr == majority))

            n_syn_min = max(n_min_real, n_min_real + n_needed)
            n_syn_maj = n_maj_real

            rng = np.random.default_rng(seed_run + 1234)
            X_min_syn = X_azr_min
            if X_min_syn.shape[0] < n_syn_min:
                idx2 = rng.choice(X_min_syn.shape[0], size=n_syn_min, replace=True)
                X_min_syn = X_min_syn[idx2]
            else:
                X_min_syn = X_min_syn[:n_syn_min]

            X_maj_real = X_tr_num[y_tr == majority]
            idxm = rng.choice(X_maj_real.shape[0], size=n_syn_maj, replace=True)
            X_maj_syn = X_maj_real[idxm] + rng.normal(0.0, 0.015, size=(n_syn_maj, X_maj_real.shape[1]))

            X_syn_num = np.vstack([X_maj_syn, X_min_syn])
            y_syn = np.concatenate([
                np.full(X_maj_syn.shape[0], majority, dtype=y_tr.dtype),
                np.full(X_min_syn.shape[0], minority, dtype=y_tr.dtype)
            ])

            # FIX: Only inverse scaling
            X_syn_num_orig = inverse_scale_only(pre_num, X_syn_num) if numeric_cols else X_syn_num
            df_syn = pd.DataFrame(X_syn_num_orig, columns=numeric_cols if numeric_cols else ["dummy_num"])

            if cat_cols and cat_dists is not None:
                cat_out_rows = []
                for cl in y_syn:
                    tmp_df = sample_categorical(cat_dists, cl, n=1, seed=seed_run + 222 + int(hash(cl)) % 99991)
                    cat_out_rows.append(tmp_df.iloc[0])
                cat_out = pd.DataFrame(cat_out_rows).reset_index(drop=True)
                for cc in cat_cols:
                    df_syn[cc] = cat_out[cc].values

            df_syn[cfg.target_col] = y_syn

            outpath = paths.data_synthetic / "AZR" / f"run_{run_i+1:02d}.csv"
            df_syn.to_csv(outpath, index=False)
            runs_info.append({"run": run_i + 1, "method": "AZR", "path": str(outpath), "seed": seed_run})

            hist_path = paths.data_synthetic / "AZR" / f"curriculum_run_{run_i+1:02d}.csv"
            hist_df.to_csv(hist_path, index=False)
            best_path = paths.data_synthetic / "AZR" / f"best_metrics_run_{run_i+1:02d}.json"
            best_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")

        except Exception as e:
            logger.warning(f"[Generate] AZR failed in run {run_i+1}: {repr(e)}. Creating robust fallback AZR dataset.")
            rng = np.random.default_rng(seed_run + 2025)
            idx_all = rng.choice(X_tr_num.shape[0], size=X_tr_num.shape[0], replace=True)
            X_fb = X_tr_num[idx_all] + rng.normal(0.0, 0.02, size=X_tr_num[idx_all].shape)
            y_fb = y_tr[idx_all]

            # FIX: Only inverse scaling
            X_fb_orig = inverse_scale_only(pre_num, X_fb) if numeric_cols else X_fb
            df_syn = pd.DataFrame(X_fb_orig, columns=numeric_cols if numeric_cols else ["dummy_num"])

            if cat_cols and cat_dists is not None:
                cat_out_rows = []
                for cl in y_fb:
                    tmp_df = sample_categorical(cat_dists, cl, n=1, seed=seed_run + 333)
                    cat_out_rows.append(tmp_df.iloc[0])
                cat_out = pd.DataFrame(cat_out_rows).reset_index(drop=True)
                for cc in cat_cols:
                    df_syn[cc] = cat_out[cc].values

            df_syn[cfg.target_col] = y_fb
            outpath = paths.data_synthetic / "AZR" / f"run_{run_i+1:02d}.csv"
            df_syn.to_csv(outpath, index=False)
            runs_info.append({"run": run_i + 1, "method": "AZR", "path": str(outpath), "seed": seed_run})

            (paths.data_synthetic / "AZR" / f"curriculum_run_{run_i+1:02d}.csv").write_text("round,candidate,tolerance,reward\n", encoding="utf-8")
            (paths.data_synthetic / "AZR" / f"best_metrics_run_{run_i+1:02d}.json").write_text(
                json.dumps({"fallback": True, "reason": "AZR failed; used bootstrap+jitter fallback."}, indent=2),
                encoding="utf-8"
            )

    runs_df = pd.DataFrame(runs_info, columns=["run", "method", "path", "seed"])
    runs_index_path = paths.data_synthetic / "synthetic_index.csv"
    runs_df.to_csv(runs_index_path, index=False)

    if runs_df.shape[0] == 0:
        logger.error("[Generate] No synthetic datasets were produced. This indicates an unexpected failure.")
        raise RuntimeError("No synthetic datasets were produced. Please run with --force and check logs.")

    return {"synthetic_index": str(runs_index_path), "minority_class": str(minority), "majority_class": str(majority)}


# =============================================================================
# 15) Remaining stages (evaluate/plot/report) - unchanged from prior logic
#     (They do NOT call inverse_transform on the numeric pipeline, so no further fix needed.)
# =============================================================================

def stage_evaluate(cfg: PipelineConfig, paths: ProjectPaths, logger: logging.Logger, meta: Dict[str, Any], gen_meta: Dict[str, Any]) -> Dict[str, Any]:
    tr_df = pd.read_csv(meta["train_path"])
    test_df = pd.read_csv(meta["test_path"])

    numeric_cols = meta["numeric_cols"]
    cat_cols = meta["categorical_cols"]

    pre_full = build_preprocessor(numeric_cols, cat_cols)
    feat_cols = (numeric_cols + cat_cols) if cat_cols else numeric_cols
    if not feat_cols:
        tr_df["_dummy"] = 0.0
        test_df["_dummy"] = 0.0
        feat_cols = ["_dummy"]

    X_tr_full = pre_full.fit_transform(tr_df[feat_cols])
    y_tr = tr_df[cfg.target_col].to_numpy()

    X_test_full = pre_full.transform(test_df[feat_cols])
    y_test = test_df[cfg.target_col].to_numpy()

    pre_num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    if numeric_cols:
        X_tr_num = pre_num.fit_transform(tr_df[numeric_cols])
    else:
        X_tr_num = np.zeros((tr_df.shape[0], 1))

    classes, counts = np.unique(y_tr, return_counts=True)
    minority = classes[np.argmin(counts)]
    X_ref = X_tr_num[y_tr == minority] if np.sum(y_tr == minority) >= 20 else X_tr_num

    models = make_models(cfg.models_to_run, cfg.random_seed)

    trtr_rows = []
    trtr_curves = {}
    for mname, model in models.items():
        auc, fpr, tpr = auc_and_roc(model, X_tr_full, y_tr, X_test_full, y_test)
        trtr_rows.append({"scheme": "TRTR", "model": mname, "auc": auc})
        trtr_curves[mname] = (fpr, tpr, auc)
    trtr_df = pd.DataFrame(trtr_rows)
    trtr_df.to_csv(paths.outputs_tables / "TRTR_AUC.csv", index=False)

    sidx_path = Path(gen_meta["synthetic_index"])
    if not sidx_path.exists():
        raise RuntimeError(f"Synthetic index not found: {sidx_path}")

    try:
        sidx = pd.read_csv(sidx_path)
    except pd.errors.EmptyDataError:
        raise RuntimeError(
            "Synthetic index file is empty. Run generation with --force. "
            "This version writes headers always, so check whether an older cached file was used."
        )

    if sidx.empty:
        raise RuntimeError("No synthetic datasets found (synthetic_index.csv is empty). Run with --force and check logs.")

    required_cols = {"run", "method", "path", "seed"}
    if not required_cols.issubset(set(sidx.columns)):
        raise RuntimeError(f"Synthetic index missing required columns. Found: {list(sidx.columns)}")

    long_rows = []
    validations_rows = []
    roc_curves: Dict[str, Dict[str, List[Tuple[np.ndarray, np.ndarray, float]]]] = {}

    for _, row in sidx.iterrows():
        run_id = int(row["run"])
        method = str(row["method"])
        spath = Path(str(row["path"]))
        if not spath.exists():
            logger.warning(f"[Evaluate] Missing synthetic file: {spath}")
            continue

        df_syn = pd.read_csv(spath)

        if "_dummy" in feat_cols and "_dummy" not in df_syn.columns:
            df_syn["_dummy"] = 0.0

        X_syn_full = pre_full.transform(df_syn[feat_cols])
        y_syn = df_syn[cfg.target_col].to_numpy()

        if numeric_cols:
            X_syn_num = pre_num.transform(df_syn[numeric_cols])
        else:
            X_syn_num = np.zeros((df_syn.shape[0], 1))

        ks_avg = np.nan
        w1_avg = np.nan
        frob = np.nan
        signm = np.nan
        if X_syn_num.shape[1] > 0 and X_ref.shape[0] > 10:
            ks_vals, w1_vals = ks_w1_per_feature(X_ref, X_syn_num)
            ks_avg = float(np.nanmean(ks_vals))
            w1_avg = float(np.nanmean(w1_vals))
            C_r = safe_corr(X_ref)
            C_s = safe_corr(X_syn_num)
            frob = float(frobenius_distance(C_r, C_s))
            signm = float(sign_matching(C_r, C_s))

        v_nan_inf = validate_no_nan_inf(X_syn_num)
        dup_rate = duplicate_rate(X_syn_num) if (cfg.duplicate_check and X_syn_num.size) else np.nan
        near_dup = near_duplicate_rate(X_syn_num, cfg.near_duplicate_sample_size, cfg.near_duplicate_threshold, cfg.random_seed + run_id) \
            if (cfg.near_duplicate_check and X_syn_num.shape[0] >= 30 and X_syn_num.size) else np.nan

        validations_rows.append({
            "run": run_id,
            "method": method,
            "has_nan": v_nan_inf["has_nan"],
            "has_inf": v_nan_inf["has_inf"],
            "nan_count": v_nan_inf["nan_count"],
            "inf_count": v_nan_inf["inf_count"],
            "duplicate_rate": dup_rate,
            "near_duplicate_rate": near_dup
        })

        X_aug_full = np.vstack([X_tr_full, X_syn_full])
        y_aug = np.concatenate([y_tr, y_syn])

        for mname, model in models.items():
            try:
                auc_tstr, fpr_tstr, tpr_tstr = auc_and_roc(model, X_syn_full, y_syn, X_test_full, y_test)
            except Exception:
                auc_tstr, fpr_tstr, tpr_tstr = np.nan, np.array([0.0, 1.0]), np.array([0.0, 1.0])

            try:
                auc_tstrp, fpr_tstrp, tpr_tstrp = auc_and_roc(model, X_aug_full, y_aug, X_test_full, y_test)
            except Exception:
                auc_tstrp, fpr_tstrp, tpr_tstrp = np.nan, np.array([0.0, 1.0]), np.array([0.0, 1.0])

            roc_curves.setdefault(f"TSTR::{method}", {}).setdefault(mname, []).append((fpr_tstr, tpr_tstr, float(auc_tstr) if np.isfinite(auc_tstr) else np.nan))
            roc_curves.setdefault(f"TSTR+::{method}", {}).setdefault(mname, []).append((fpr_tstrp, tpr_tstrp, float(auc_tstrp) if np.isfinite(auc_tstrp) else np.nan))

            trtr_auc = float(pd.read_csv(paths.outputs_tables / "TRTR_AUC.csv").query("model == @mname")["auc"].iloc[0])

            long_rows.append({
                "run": run_id,
                "method": method,
                "model": mname,
                "KS_avg": ks_avg,
                "W1_avg": w1_avg,
                "FrobCorrDist": frob,
                "SignMatch": signm,
                "AUC_TRTR": trtr_auc,
                "AUC_TSTR": auc_tstr,
                "AUC_TSTR_plus": auc_tstrp
            })

    long_df = pd.DataFrame(long_rows)
    val_df_out = pd.DataFrame(validations_rows)

    if long_df.empty:
        raise RuntimeError("Evaluation produced no rows. Check that synthetic datasets were created and readable.")

    long_df.to_csv(paths.outputs_tables / "metrics_long.csv", index=False)
    val_df_out.to_csv(paths.outputs_tables / "validations.csv", index=False)

    method_summary = (long_df
                      .groupby("method", as_index=False)
                      .agg(
                          AUC_TSTR_mean=("AUC_TSTR", "mean"),
                          AUC_TSTR_std=("AUC_TSTR", "std"),
                          AUC_TSTRp_mean=("AUC_TSTR_plus", "mean"),
                          AUC_TSTRp_std=("AUC_TSTR_plus", "std"),
                          AUC_TRTR_mean=("AUC_TRTR", "mean"),
                          KS_avg=("KS_avg", "mean"),
                          W1_avg=("W1_avg", "mean"),
                          FrobCorrDist=("FrobCorrDist", "mean"),
                          SignMatch=("SignMatch", "mean")
                      ))
    method_summary.to_csv(paths.outputs_tables / "method_summary.csv", index=False)

    ms = method_summary.copy()
    ms["FidelityProxy"] = 0.6 * (1.0 - ms["KS_avg"]) + 0.4 * (1.0 / (1.0 + ms["W1_avg"]))
    ms["StructureProxy"] = 0.6 * ms["SignMatch"] + 0.4 * (1.0 / (1.0 + ms["FrobCorrDist"]))
    ms["UtilityProxy_TSTR"] = ms["AUC_TSTR_mean"]
    ms["UtilityProxy_TSTRp"] = ms["AUC_TSTRp_mean"]

    ms["FidelityNorm"] = minmax_normalize(ms["FidelityProxy"].to_numpy())
    ms["StructureNorm"] = minmax_normalize(ms["StructureProxy"].to_numpy())
    ms["UtilityNorm_TSTR"] = minmax_normalize(ms["UtilityProxy_TSTR"].to_numpy())
    ms["UtilityNorm_TSTRp"] = minmax_normalize(ms["UtilityProxy_TSTRp"].to_numpy())

    w = cfg.azr_reward_weights
    ms["FinalScore_TSTR"] = (w["fidelity"] * ms["FidelityNorm"] +
                             w["structure"] * ms["StructureNorm"] +
                             w["utility"] * ms["UtilityNorm_TSTR"])
    ms["FinalScore_TSTRp"] = (w["fidelity"] * ms["FidelityNorm"] +
                              w["structure"] * ms["StructureNorm"] +
                              w["utility"] * ms["UtilityNorm_TSTRp"])

    ms.to_csv(paths.outputs_tables / "normalized_scores.csv", index=False)
    ms.sort_values("FinalScore_TSTR", ascending=False).to_csv(paths.outputs_tables / "ranking_TSTR.csv", index=False)
    ms.sort_values("FinalScore_TSTRp", ascending=False).to_csv(paths.outputs_tables / "ranking_TSTR_plus.csv", index=False)

    xlsx_path = paths.outputs_tables / "metrics_summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.read_csv(paths.outputs_tables / "TRTR_AUC.csv").to_excel(writer, sheet_name="TRTR_AUC", index=False)
        long_df.to_excel(writer, sheet_name="Metrics_Long", index=False)
        method_summary.to_excel(writer, sheet_name="Method_Summary", index=False)
        ms.to_excel(writer, sheet_name="Normalized_Scores", index=False)
        pd.read_csv(paths.outputs_tables / "ranking_TSTR.csv").to_excel(writer, sheet_name="Ranking_TSTR", index=False)
        pd.read_csv(paths.outputs_tables / "ranking_TSTR_plus.csv").to_excel(writer, sheet_name="Ranking_TSTR_plus", index=False)
        val_df_out.to_excel(writer, sheet_name="Validations", index=False)

    rep_curves = {}
    for key, model_map in roc_curves.items():
        rep_curves[key] = {}
        for mname, curve_list in model_map.items():
            aucs = np.array([c[2] for c in curve_list if np.isfinite(c[2])], dtype=float)
            if aucs.size == 0:
                fpr, tpr, aucv = curve_list[0]
                rep_curves[key][mname] = {"auc": float(aucv) if np.isfinite(aucv) else float("nan"),
                                          "fpr": fpr.tolist(), "tpr": tpr.tolist()}
                continue
            order = np.argsort(aucs)
            med_auc = aucs[order[len(order)//2]]
            best_i = 0
            best_d = 1e18
            for i, (fpr, tpr, aucv) in enumerate(curve_list):
                if not np.isfinite(aucv):
                    continue
                d = abs(aucv - med_auc)
                if d < best_d:
                    best_d = d
                    best_i = i
            fpr, tpr, aucv = curve_list[best_i]
            rep_curves[key][mname] = {"auc": float(aucv), "fpr": fpr.tolist(), "tpr": tpr.tolist()}

    (paths.cache / "roc_curves_representative.json").write_text(json.dumps(rep_curves, indent=2), encoding="utf-8")
    (paths.cache / "roc_curves_trtr.json").write_text(
        json.dumps(
            {"TRTR": {m: {"auc": float(a), "fpr": f.tolist(), "tpr": t.tolist()} for m, (f, t, a) in trtr_curves.items()}},
            indent=2
        ),
        encoding="utf-8"
    )

    return {"excel": str(xlsx_path), "metrics_long": str(paths.outputs_tables / "metrics_long.csv")}


def stage_plot(cfg: PipelineConfig, paths: ProjectPaths, logger: logging.Logger, meta: Dict[str, Any]) -> Dict[str, Any]:
    numeric_cols = meta["numeric_cols"]

    rep_path = paths.cache / "roc_curves_representative.json"
    rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {}

    trtr_path = paths.cache / "roc_curves_trtr.json"
    if trtr_path.exists():
        trtr = json.loads(trtr_path.read_text(encoding="utf-8")).get("TRTR", {})
        curves = {}
        for mname, d in trtr.items():
            curves[mname] = (np.array(d["fpr"]), np.array(d["tpr"]), float(d["auc"]))
        save_roc_plot(curves, "ROC (TRTR) - Train Real / Test Real", paths.outputs_figures / "ROC_TRTR.png", cfg.figure_dpi)

    for key, model_map in rep.items():
        safe_key = key.replace("+", "PLUS").replace(":", "_")
        curves = {}
        for mname, d in model_map.items():
            curves[mname] = (np.array(d["fpr"]), np.array(d["tpr"]), float(d["auc"]) if d["auc"] is not None else float("nan"))
        save_roc_plot(curves, f"ROC ({key})", paths.outputs_figures / f"ROC_{safe_key}.png", cfg.figure_dpi)

    ns_path = paths.outputs_tables / "normalized_scores.csv"
    if ns_path.exists():
        ms = pd.read_csv(ns_path)
        methods = ms["method"].tolist()
        x = np.arange(len(methods))

        plt.figure(figsize=(10, 5))
        plt.bar(x - 0.30, ms["FidelityNorm"], width=0.25, label="Fidelity (0-1)")
        plt.bar(x - 0.05, ms["StructureNorm"], width=0.25, label="Structure (0-1)")
        plt.bar(x + 0.20, ms["UtilityNorm_TSTR"], width=0.25, label="Utility TSTR (0-1)")
        plt.xticks(x, methods, rotation=0)
        plt.ylim(0, 1.05)
        plt.ylabel("Normalized score")
        plt.title("Normalized Comparison: Fidelity vs Structure vs Utility (TSTR)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(paths.outputs_figures / "Normalized_Metrics_TSTR.png", dpi=cfg.figure_dpi)
        plt.close()

    syn_azr = paths.data_synthetic / "AZR" / "run_01.csv"
    tr_path = paths.data_processed / "train.csv"
    if syn_azr.exists() and tr_path.exists() and numeric_cols:
        tr_df = pd.read_csv(tr_path)
        azr_df = pd.read_csv(syn_azr)

        pre_num = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        X_tr_num = pre_num.fit_transform(tr_df[numeric_cols])
        X_azr_num = pre_num.transform(azr_df[numeric_cols])

        C_r = safe_corr(X_tr_num)
        C_s = safe_corr(X_azr_num)
        plot_corr_heatmap(C_r, "Correlation Heatmap (Real Train, numeric)", paths.outputs_figures / "Corr_RealTrain.png", cfg.figure_dpi)
        plot_corr_heatmap(C_s, "Correlation Heatmap (AZR Synthetic, numeric)", paths.outputs_figures / "Corr_AZR.png", cfg.figure_dpi)
        plot_corr_heatmap(C_r - C_s, "Correlation Difference (Real - AZR)", paths.outputs_figures / "Corr_Diff_RealMinusAZR.png", cfg.figure_dpi)

        feat_names = numeric_cols[: cfg.max_features_for_distribution_plots]
        plot_distributions(X_tr_num[:, :len(feat_names)], X_azr_num[:, :len(feat_names)],
                           feat_names, paths.outputs_figures / "Distributions_Real_vs_AZR.png", cfg.figure_dpi)

        feat_qq = numeric_cols[: cfg.max_features_for_qq_plots]
        plot_qq(X_tr_num[:, :len(feat_qq)], X_azr_num[:, :len(feat_qq)],
                feat_qq, paths.outputs_figures / "QQ_Real_vs_AZR.png", cfg.figure_dpi)

    return {"figures_dir": str(paths.outputs_figures)}


def stage_report(cfg: PipelineConfig, paths: ProjectPaths, logger: logging.Logger, meta: Dict[str, Any]) -> Dict[str, Any]:
    env_info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": __import__("sklearn").__version__,
        "scipy_version": __import__("scipy").__version__,
        "imblearn_available": IMBLEARN_AVAILABLE,
        "dataset_path": cfg.dataset_path,
        "dataset_type": cfg.dataset_type,
        "note": "If imblearn is not available, internal CPU oversamplers are used."
    }

    (paths.outputs / "env.json").write_text(json.dumps(env_info, indent=2), encoding="utf-8")
    (paths.outputs / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    lines = []
    lines.append("# Absolute-Zero-Inspired Synthetic Data Pipeline Report\n")
    lines.append("## Dataset (Enforced)\n")
    lines.append(f"- Path: `{cfg.dataset_path}`\n")
    lines.append(f"- Type: `{cfg.dataset_type.upper()}`\n")
    lines.append(f"- Target column: `{cfg.target_col}`\n\n")

    lines.append("## Environment\n")
    lines.append("```json\n" + json.dumps(env_info, indent=2) + "\n```\n")

    report_path = paths.outputs_reports / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"[Report] Written: {report_path}")
    return {"report": str(report_path)}


# =============================================================================
# 16) Orchestrator (stages + caching)
# =============================================================================

def run_pipeline(cfg: PipelineConfig) -> None:
    paths = ProjectPaths.from_outdir(cfg.outdir)
    paths.ensure_all()
    logger = setup_logger(paths.outputs_logs / "run.log")

    dataset_path = Path(cfg.dataset_path)
    dmt = file_mtime(dataset_path) if dataset_path.exists() else None
    run_hash = compute_run_hash(cfg, dmt)

    manifest_path = paths.cache / "manifest.json"
    manifest = load_manifest(manifest_path)

    logger.info("=== ABSOLUTE-ZERO FULL PIPELINE (CPU) - FIXED (v2) ===")
    logger.info(f"Mode: {cfg.mode} | Stage: {cfg.stage} | Force: {cfg.force}")
    logger.info(f"Dataset: {cfg.dataset_path} | Type: {cfg.dataset_type.upper()} | Exists: {dataset_path.exists()} | mtime: {dmt}")
    logger.info(f"Run hash: {run_hash}")
    logger.info(f"imblearn available: {IMBLEARN_AVAILABLE} (fallback oversamplers enabled if False)")

    stage_order = ["preprocess", "generate", "evaluate", "plot", "report"]
    requested = cfg.stage
    if requested == "all":
        stages_to_run = stage_order
    else:
        if requested not in stage_order:
            raise ValueError("Invalid stage. Use preprocess|generate|evaluate|plot|report|all.")
        idx = stage_order.index(requested)
        stages_to_run = stage_order[: idx + 1]

    meta = {}
    gen_meta = {}
    eval_meta = {}
    plot_meta = {}
    rep_meta = {}

    if "preprocess" in stages_to_run:
        if (not cfg.force) and should_skip_stage(manifest, "preprocess", run_hash):
            logger.info("[Cache] Skipping preprocess stage (cached).")
            meta_path = paths.cache / "meta_preprocess.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            meta = stage_preprocess(cfg, paths, logger)
            (paths.cache / "meta_preprocess.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            manifest = mark_stage_done(manifest, "preprocess", run_hash, {"meta": "cache/meta_preprocess.json"})
            save_manifest(manifest_path, manifest)

    if "generate" in stages_to_run:
        if not meta:
            meta_path = paths.cache / "meta_preprocess.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                raise RuntimeError("Missing preprocess metadata; run stage preprocess first.")

        if (not cfg.force) and should_skip_stage(manifest, "generate", run_hash):
            logger.info("[Cache] Skipping generate stage (cached).")
            gm_path = paths.cache / "meta_generate.json"
            if gm_path.exists():
                gen_meta = json.loads(gm_path.read_text(encoding="utf-8"))
        else:
            gen_meta = stage_generate(cfg, paths, logger, meta)
            (paths.cache / "meta_generate.json").write_text(json.dumps(gen_meta, indent=2), encoding="utf-8")
            manifest = mark_stage_done(manifest, "generate", run_hash, {"synthetic_index": gen_meta.get("synthetic_index", "")})
            save_manifest(manifest_path, manifest)

    if "evaluate" in stages_to_run:
        if not meta:
            meta = json.loads((paths.cache / "meta_preprocess.json").read_text(encoding="utf-8"))
        if not gen_meta:
            gen_meta = json.loads((paths.cache / "meta_generate.json").read_text(encoding="utf-8"))

        if (not cfg.force) and should_skip_stage(manifest, "evaluate", run_hash):
            logger.info("[Cache] Skipping evaluate stage (cached).")
            ev_path = paths.cache / "meta_evaluate.json"
            if ev_path.exists():
                eval_meta = json.loads(ev_path.read_text(encoding="utf-8"))
        else:
            eval_meta = stage_evaluate(cfg, paths, logger, meta, gen_meta)
            (paths.cache / "meta_evaluate.json").write_text(json.dumps(eval_meta, indent=2), encoding="utf-8")
            manifest = mark_stage_done(manifest, "evaluate", run_hash, {"excel": eval_meta.get("excel", "")})
            save_manifest(manifest_path, manifest)

    if "plot" in stages_to_run:
        if not meta:
            meta = json.loads((paths.cache / "meta_preprocess.json").read_text(encoding="utf-8"))

        if (not cfg.force) and should_skip_stage(manifest, "plot", run_hash):
            logger.info("[Cache] Skipping plot stage (cached).")
            pl_path = paths.cache / "meta_plot.json"
            if pl_path.exists():
                plot_meta = json.loads(pl_path.read_text(encoding="utf-8"))
        else:
            plot_meta = stage_plot(cfg, paths, logger, meta)
            (paths.cache / "meta_plot.json").write_text(json.dumps(plot_meta, indent=2), encoding="utf-8")
            manifest = mark_stage_done(manifest, "plot", run_hash, {"figures_dir": plot_meta.get("figures_dir", "")})
            save_manifest(manifest_path, manifest)

    if "report" in stages_to_run:
        if not meta:
            meta = json.loads((paths.cache / "meta_preprocess.json").read_text(encoding="utf-8"))

        if (not cfg.force) and should_skip_stage(manifest, "report", run_hash):
            logger.info("[Cache] Skipping report stage (cached).")
            rp_path = paths.cache / "meta_report.json"
            if rp_path.exists():
                rep_meta = json.loads(rp_path.read_text(encoding="utf-8"))
        else:
            rep_meta = stage_report(cfg, paths, logger, meta)
            (paths.cache / "meta_report.json").write_text(json.dumps(rep_meta, indent=2), encoding="utf-8")
            manifest = mark_stage_done(manifest, "report", run_hash, {"report": rep_meta.get("report", "")})
            save_manifest(manifest_path, manifest)

    logger.info("Pipeline completed successfully.")
    logger.info(f"Root directory: {paths.root}")
    logger.info(f"Outputs: {paths.outputs}")


# =============================================================================
# 17) CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Absolute-Zero-inspired full pipeline (CPU): oversampling baselines vs AZR with verifiable curriculum, full evaluation + plots + Excel exports."
    )
    parser.add_argument("--dataset_path", type=str, default=FIXED_DATASET_PATH,
                        help="Dataset path. Default is the fixed water_potability.csv path (as requested).")
    parser.add_argument("--dataset_type", type=str, default=FIXED_DATASET_TYPE, choices=["csv", "excel"],
                        help="Dataset type (enforced). Default is CSV (as requested).")
    parser.add_argument("--target", type=str, default="Potability",
                        help="Target column name.")
    parser.add_argument("--outdir", type=str, default="./ABSOLUTE_ZERO_FULL_OUTPUTS",
                        help="Root output directory (project-style).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--mode", type=str, default="paper", choices=["quick", "paper"],
                        help="Compute mode: quick (faster) or paper (more robust).")
    parser.add_argument("--stage", type=str, default="all",
                        help="Stage: preprocess|generate|evaluate|plot|report|all.")
    parser.add_argument("--force", action="store_true",
                        help="Force rerun stages ignoring cache.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = make_config(
        mode=args.mode,
        dataset_path=args.dataset_path,
        dataset_type=args.dataset_type,
        target_col=args.target,
        outdir=args.outdir,
        seed=args.seed,
        stage=args.stage,
        force=args.force
    )

    set_global_seed(cfg.random_seed)
    t0 = time.time()
    run_pipeline(cfg)
    dt = time.time() - t0
    print(f"\n[Done] Total runtime: {dt:.2f} seconds (CPU).\n")


if __name__ == "__main__":
    main()



