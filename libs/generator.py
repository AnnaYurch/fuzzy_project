"""
Генерация множества нечётких величин (п.5 задания, 100-300 штук).

Математический аппарат (источник: Абчук В.А. и др., «Справочник по исследованию
операций», табл. 2.2.5, стр. 64-65):

  Шаг 1. ξ_i ~ U[0,1], i=1..n — равномерные величины.

  Шаг 2. Нормированная центрированная сумма:
         ω_n = (Σξ_i − n/2) / √(n/12)
         Обоснование: E[Σξ] = n/2, Var(Σξ) = n/12
         → E[ω_n] = 0, Var(ω_n) = 1 ТОЧНО.

  Шаг 3. Поправочный член (полиномы Эрмита):
         n ≥ 5: η = ω − (ω³ − 3ω)/(20n)              (H₃)
         n < 5: η = ω − (41/(13440n²))·(ω⁵ − 10ω³ + 15ω)  (H₅)

  Шаг 4. Масштабирование: x' = x₀_target + σ_target · η.

Генерация множества с группировкой (авторская конструкция для создания
содержательной структуры данных, в учебнике не указана):
  x₀ = cx_g + group_spread · η₁
  σ  = cs_g · (1 + 0.15 · η₂)
где cx_g, cs_g — центры «истинных» групп, η₁, η₂ — независимые реализации.

ВАЖНО: диапазон σ подбирается так, чтобы μ_max = 1/(σ·√(2π)) варьировался
от ≈0.1 до ≈0.8. Это позволяет продемонстрировать α-отбор подмножества.
"""
from __future__ import annotations
import math, random
from pathlib import Path
from typing import List, Sequence, Tuple
from .fuzzy_number import FuzzyNumber
from .logger import get_logger

log = get_logger("generator")
MIN_COUNT, MAX_COUNT, DEFAULT_N_UNIFORM = 100, 300, 12

def standard_normal_approx(n: int = DEFAULT_N_UNIFORM,
                           rng: random.Random | None = None) -> float:
    """Одна реализация η ~ N(0,1) по формуле аппроксимации."""
    if n < 2:
        raise ValueError("Число равномерных слагаемых n должно быть >= 2.")
    r = rng if rng is not None else random
    uniforms = [r.random() for _ in range(n)]
    omega = (math.fsum(uniforms) - n / 2.0) / math.sqrt(n / 12.0)

    # ПРАВИЛЬНЫЕ ГРАНИЦЫ ПО УЧЕБНИКУ (Абчук, табл. 2.2.5)
    if n >= 5:
        correction = (omega ** 3 - 3.0 * omega) / (20.0 * n)
    else:
        correction = (41.0 / (13440.0 * n * n)) * (
            omega ** 5 - 10.0 * omega ** 3 + 15.0 * omega
        )
    return omega - correction

def check_generator_moments(sample_size: int = 20000,
                            n: int = DEFAULT_N_UNIFORM,
                            seed: int | None = 42) -> dict:
    """Эмпирическая проверка генератора (ответ на вопрос «почему Var=1»)."""
    rng = random.Random(seed)
    values = [standard_normal_approx(n, rng) for _ in range(sample_size)]
    m = math.fsum(values) / sample_size
    var = math.fsum((v - m) ** 2 for v in values) / (sample_size - 1)
    sd = math.sqrt(var)
    skew = math.fsum(((v - m) / sd) ** 3 for v in values) / sample_size
    kurt = math.fsum(((v - m) / sd) ** 4 for v in values) / sample_size - 3.0
    log.info("Проверка генератора (n=%d, N=%d): mean=%.4f, std=%.4f, skew=%.4f, excess=%.4f",
             n, sample_size, m, sd, skew, kurt)
    return {"sample_size": sample_size, "mean": m, "std": sd, "variance": var,
            "skewness": skew, "excess_kurtosis": kurt}

def generate_fuzzy_set(count: int = 150, groups: int = 3,
                       n_uniform: int = DEFAULT_N_UNIFORM,
                       seed: int | None = 42,
                       x_range: Tuple[float, float] = (10.0, 90.0),
                       sigma_range: Tuple[float, float] = (0.5, 3.5),
                       group_spread: float = 3.0) -> List[FuzzyNumber]:
    """
    Генерирует count нечётких величин, сгруппированных в groups «сгустков».
    Диапазон σ = [0.5, 3.5] подобран так, чтобы μ_max варьировался:
      σ=0.5 → μ_max ≈ 0.798 (пройдёт α=0.43)
      σ=3.5 → μ_max ≈ 0.114 (НЕ пройдёт α=0.43)
    """
    if not (MIN_COUNT <= count <= MAX_COUNT):
        raise ValueError(f"По заданию количество величин должно быть {MIN_COUNT}..{MAX_COUNT}.")
    rng = random.Random(seed)

    lo_x, hi_x = x_range
    lo_s, hi_s = sigma_range
    if groups == 1:
        centers = [((lo_x + hi_x) / 2.0, (lo_s + hi_s) / 2.0)]
    else:
        centers = [
            (lo_x + (hi_x - lo_x) * g / (groups - 1),
             lo_s + (hi_s - lo_s) * g / (groups - 1))
            for g in range(groups)
        ]

    fuzzy_set: List[FuzzyNumber] = []
    per_group, remainder = count // groups, count % groups
    for g, (cx, cs) in enumerate(centers):
        size = per_group + (1 if g < remainder else 0)
        for _ in range(size):
            eta1 = standard_normal_approx(n_uniform, rng)
            eta2 = standard_normal_approx(n_uniform, rng)
            x0 = cx + group_spread * eta1
            sigma = max(0.3, cs * (1.0 + 0.2 * eta2))
            fuzzy_set.append(FuzzyNumber(x0, sigma, name=f"A{len(fuzzy_set) + 1}"))

    rng.shuffle(fuzzy_set)
    for i, fn in enumerate(fuzzy_set, start=1):
        fn.name = f"A{i}"
    log.info("Сгенерировано %d нечётких величин (%d групп, n_uniform=%d)",
             len(fuzzy_set), groups, n_uniform)
    return fuzzy_set

def generate_uniform_fuzzy_set(count: int = 150,
                               x_range: Tuple[float, float] = (0.0, 100.0),
                               sigma_range: Tuple[float, float] = (0.5, 3.5),
                               seed: int | None = 42) -> List[FuzzyNumber]:
    """Альтернативный режим: равномерный разброс параметров (без структуры)."""
    rng = random.Random(seed)
    return [FuzzyNumber(rng.uniform(*x_range), rng.uniform(*sigma_range),
                        name=f"A{i + 1}") for i in range(count)]

def save_to_tsv(fuzzy_set: Sequence[FuzzyNumber], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("x0\tsigma\n")
        for fn in fuzzy_set:
            f.write(f"{fn.x0:.10g}\t{fn.sigma:.10g}\n")
    log.info("Сохранено %d величин в файл %s", len(fuzzy_set), path)
    return path

def load_from_tsv(path: str | Path) -> List[FuzzyNumber]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    fuzzy_set: List[FuzzyNumber] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(";", "\t").replace(",", ".").split()
            if len(parts) >= 2:
                try:
                    fuzzy_set.append(FuzzyNumber(float(parts[0]), float(parts[1]),
                                                 name=f"A{len(fuzzy_set) + 1}"))
                except ValueError:
                    if line_no > 1:
                        log.warning("Строка %d пропущена: %s", line_no, line)
    log.info("Загружено %d величин из файла %s", len(fuzzy_set), path)
    return fuzzy_set