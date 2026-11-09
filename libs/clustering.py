"""
Кластерный анализ множества нечётких величин (п.6 задания).

Три режима:
  1. K-Means по (x₀, σ) — основной, по критерию WCSS.
  2. GMM + BIC — обоснование выбора k.
  3. Отбор подмножества по α-порогу + кластеризация — ключевая демонстрация.

Математический аппарат:
  - Критерий K-Means: J = Σ_c Σ_{i∈C_c} ‖z_i − μ_c‖² → min
  - BIC = -2·ln(L) + p·ln(n) (Schwarz, 1978)
  - α-отбор: величина включается в подмножество, если μ(x₀) ≥ α
"""
from __future__ import annotations
import math
from typing import Dict, List, Sequence, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from .fuzzy_number import FuzzyNumber
from .logger import get_logger

log = get_logger("clustering")
RANDOM_STATE = 42

def feature_matrix(fuzzy_set: Sequence[FuzzyNumber]) -> np.ndarray:
    return np.array([[fn.x0, fn.sigma] for fn in fuzzy_set], dtype=float)

def calculate_distance(a: FuzzyNumber, b: FuzzyNumber) -> float:
    return math.hypot(a.x0 - b.x0, a.sigma - b.sigma)

def _validate_k(fuzzy_set: Sequence[FuzzyNumber], k: int) -> None:
    n = len(fuzzy_set)
    if n < 2:
        raise ValueError("Для кластеризации нужно минимум 2 величины.")
    if not (1 <= k <= n):
        raise ValueError(f"Число кластеров k должно быть в диапазоне 1..{n}.")

# ----------------------------------------------------------------------
# Основной метод: K-Means по критерию WCSS
# ----------------------------------------------------------------------
def perform_clustering(fuzzy_set: Sequence[FuzzyNumber],
                       n_clusters: int, scale: bool = True) -> Dict:
    _validate_k(fuzzy_set, n_clusters)
    X = feature_matrix(fuzzy_set)

    # ВАЖНО: сохраняем scaler в переменную, чтобы потом использовать его же
    if scale:
        scaler = StandardScaler()
        Z = scaler.fit_transform(X)
    else:
        scaler = None
        Z = X

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(Z)

    # Используем ТОТ ЖЕ scaler, на котором делали fit_transform
    if scaler is not None:
        centers = scaler.inverse_transform(model.cluster_centers_)
    else:
        centers = model.cluster_centers_

    summary = []
    for c in sorted(set(labels)):
        members = [fn for fn, lbl in zip(fuzzy_set, labels) if lbl == c]
        summary.append({
            "cluster": c, "size": len(members),
            "mean_x0": float(np.mean([m.x0 for m in members])),
            "mean_sigma": float(np.mean([m.sigma for m in members])),
            "min_x0": float(min(m.x0 for m in members)),
            "max_x0": float(max(m.x0 for m in members)),
        })
    log.info("K-Means: N=%d, k=%d, инерция J=%.4f", len(fuzzy_set), n_clusters, model.inertia_)
    return {"method": "K-Means (критерий: минимум WCSS)", "labels": labels.tolist(),
            "model": model, "inertia": float(model.inertia_),
            "centers": centers.tolist(), "summary": summary, "k": n_clusters}

# ----------------------------------------------------------------------
# Подбор числа кластеров
# ----------------------------------------------------------------------
def elbow_method(fuzzy_set: Sequence[FuzzyNumber], max_k: int = 10) -> Tuple[List[int], List[float]]:
    n = len(fuzzy_set)
    ks = list(range(1, min(max_k, n) + 1))
    inertias = [perform_clustering(fuzzy_set, k)["inertia"] for k in ks]
    return ks, inertias

def gmm_bic(fuzzy_set: Sequence[FuzzyNumber], max_k: int = 10,
            scale: bool = True) -> Tuple[List[int], List[float], int]:
    n = len(fuzzy_set)
    X = feature_matrix(fuzzy_set)
    Z = StandardScaler().fit_transform(X) if scale else X
    ks = list(range(1, min(max_k, n) + 1))
    bics = []
    for k in ks:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=3, max_iter=300, random_state=RANDOM_STATE)
        gmm.fit(Z)
        bics.append(float(gmm.bic(Z)))
    optimal_k = ks[int(np.argmin(bics))]
    log.info("GMM+BIC: оптимальное k=%d", optimal_k)
    return ks, bics, optimal_k

# ----------------------------------------------------------------------
# α-перекрытие (доп. режим)
# ----------------------------------------------------------------------
def cluster_by_alpha_overlap(fuzzy_set: Sequence[FuzzyNumber], alpha: float) -> Dict:
    """Кластеризация по перекрытию α-срезов (single-linkage)."""
    intervals = []
    for fn in fuzzy_set:
        if fn.passes_alpha_threshold(alpha):
            try:
                intervals.append(fn.alpha_cut(alpha))
            except ValueError:
                intervals.append(None)
        else:
            intervals.append(None)

    valid = [(i, iv) for i, iv in enumerate(intervals) if iv is not None]
    if not valid:
        return {"method": f"α-перекрытие (α={alpha})", "alpha": alpha,
                "labels": [-1] * len(fuzzy_set), "k": 0, "summary": []}

    order = sorted(valid, key=lambda x: x[1][0])
    labels = [-1] * len(fuzzy_set)
    cluster_id, current_right = -1, -math.inf
    for idx, (left, right) in order:
        if cluster_id == -1 or left > current_right:
            cluster_id += 1
            current_right = right
        else:
            current_right = max(current_right, right)
        labels[idx] = cluster_id

    summary = []
    for c in range(cluster_id + 1):
        members = [fn for fn, lbl in zip(fuzzy_set, labels) if lbl == c]
        summary.append({"cluster": c, "size": len(members),
                        "mean_x0": float(np.mean([m.x0 for m in members])),
                        "mean_sigma": float(np.mean([m.sigma for m in members]))})
    log.info("α-перекрытие: α=%.3f → %d кластер(ов)", alpha, cluster_id + 1)
    return {"method": f"Перекрытие α-срезов (α={alpha})", "alpha": alpha,
            "labels": labels, "k": cluster_id + 1, "summary": summary}

def alpha_sweep(fuzzy_set: Sequence[FuzzyNumber],
                alphas: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)) -> List[Dict]:
    rows = []
    for alpha in alphas:
        passed = sum(1 for fn in fuzzy_set if fn.passes_alpha_threshold(alpha))
        res = cluster_by_alpha_overlap(fuzzy_set, alpha)
        rows.append({"alpha": alpha, "passed": passed, "k": res["k"]})
    return rows

# ----------------------------------------------------------------------
# КЛЮЧЕВАЯ ФИЧА: α-отбор подмножества + кластеризация
# ----------------------------------------------------------------------
def filter_by_alpha_threshold(fuzzy_set: Sequence[FuzzyNumber],
                              alpha: float) -> List[FuzzyNumber]:
    """
    Отбор нечётких величин по α-порогу.
    Остаются только те, у кого μ(x₀) ≥ α.
    Это — «определённость по Гегелю»: да/нет, любит/не любит.
    """
    filtered = [fn for fn in fuzzy_set if fn.passes_alpha_threshold(alpha)]
    log.info("α-отбор: α=%.3f, было %d, стало %d величин (μ_max ≥ α)",
             alpha, len(fuzzy_set), len(filtered))
    return filtered

def clustering_on_subset(fuzzy_set: Sequence[FuzzyNumber],
                         alpha: float, n_clusters: int) -> Dict:
    """
    Полный сценарий демонстрации:
      1) Отбор подмножества по α-порогу
      2) Кластеризация подмножества
      3) Сравнение с кластеризацией полного множества
    """
    full_result = perform_clustering(fuzzy_set, n_clusters)
    subset = filter_by_alpha_threshold(fuzzy_set, alpha)

    if len(subset) < n_clusters:
        log.warning("В подмножестве меньше величин, чем кластеров (%d < %d)",
                    len(subset), n_clusters)
        return {
            "full": full_result,
            "alpha": alpha,
            "subset_size": len(subset),
            "subset_result": None,
            "message": f"Подмножество слишком мало ({len(subset)} < k={n_clusters})"
        }

    subset_result = perform_clustering(subset, n_clusters)
    return {
        "full": full_result,
        "alpha": alpha,
        "subset_size": len(subset),
        "subset_result": subset_result,
        "message": f"Было {len(fuzzy_set)} → стало {len(subset)} по α={alpha}"
    }