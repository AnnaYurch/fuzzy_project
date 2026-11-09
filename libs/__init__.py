"""
Пакет libs — независимая библиотека для работы с нечёткими величинами
с нормальной (гауссовской) функцией принадлежности.

Источники математического аппарата:
  - Функция принадлежности: нормальное нечёткое множество по Заде (1965)
    с коэффициентом плотности (без нормировки максимума на 1).
  - Аппроксимация нормального распределения: Абчук В.А. и др.,
    «Справочник по исследованию операций», табл. 2.2.5, стр. 64-65.
  - Критерий BIC: Schwarz G., Annals of Statistics, 1978, Vol. 6, No. 2.
"""

from .fuzzy_number import FuzzyNumber
from .fuzzy_ops import (
    add, subtract, multiply, divide, multiply_by_scalar, compare, compare_str
)
from .generator import (
    generate_fuzzy_set, generate_uniform_fuzzy_set,
    standard_normal_approx, check_generator_moments,
    save_to_tsv, load_from_tsv
)
from .clustering import (
    calculate_distance, perform_clustering, elbow_method, gmm_bic,
    cluster_by_alpha_overlap, alpha_sweep,
    filter_by_alpha_threshold, clustering_on_subset
)
from .logger import get_logger

__all__ = [
    "FuzzyNumber",
    "add", "subtract", "multiply", "divide", "multiply_by_scalar", "compare", "compare_str",
    "generate_fuzzy_set", "generate_uniform_fuzzy_set", "standard_normal_approx",
    "check_generator_moments", "save_to_tsv", "load_from_tsv",
    "calculate_distance", "perform_clustering", "elbow_method", "gmm_bic",
    "cluster_by_alpha_overlap", "alpha_sweep",
    "filter_by_alpha_threshold", "clustering_on_subset",
    "get_logger",
]
__version__ = "2.0.0"