"""
Способ представления одной нечёткой величины (п.2 задания).

Математический аппарат:
  μ(x) = (1 / (σ·√(2π))) · exp( -(x - x₀)² / (2σ²) )

Это функция плотности нормального распределения, используемая как
функция принадлежности НЕЧЁТКОГО множества (по Заде, 1965).
ВАЖНО: максимум μ(x₀) = 1/(σ·√(2π)) НЕ равен 1 — это осознанное решение,
позволяющее использовать α-уровень как ПОРОГ ОТБОРА величин из множества.

Источники:
  - Заде Л. Нечёткие множества // Information and Control, 1965.
  - Леоненков А.В. Нечёткое моделирование в среде MATLAB, БХВ-Петербург.
"""
from __future__ import annotations
import math
from typing import Tuple

class FuzzyNumber:
    """Нечёткое число с гауссовской функцией принадлежности (с коэффициентом)."""
    __slots__ = ("x0", "sigma", "name")

    def __init__(self, x0: float, sigma: float, name: str = ""):
        x0, sigma = float(x0), float(sigma)
        if not math.isfinite(x0):
            raise ValueError("x0 должен быть конечным числом.")
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"sigma должна быть > 0 (получено {sigma}).")
        self.x0 = x0
        self.sigma = sigma
        self.name = name

    # ------------------------------------------------------------------
    # Функция принадлежности (ПРЯМАЯ формула)
    # ------------------------------------------------------------------
    def membership(self, x: float) -> float:
        """μ(x) = (1/(σ·√(2π))) · exp(-(x-x₀)²/(2σ²))"""
        z = (float(x) - self.x0) / self.sigma
        return (1.0 / (self.sigma * math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * z * z)

    __call__ = membership

    def mu_max(self) -> float:
        """Максимум функции принадлежности: μ(x₀) = 1/(σ·√(2π))."""
        return 1.0 / (self.sigma * math.sqrt(2.0 * math.pi))

    def passes_alpha_threshold(self, alpha: float) -> bool:
        """
        Проверяет, проходит ли величина α-порог отбора.
        Величина включается в «чёткое» подмножество, если μ(x₀) ≥ α.
        Это — критерий определённой принадлежности (по преподавателю).
        """
        return self.mu_max() >= alpha

    # ------------------------------------------------------------------
    # α-срез: ЧИСЛЕННО (метод бисекции) — ключевое требование
    # ------------------------------------------------------------------
    def alpha_cut(self, alpha: float, tolerance: float = 1e-9) -> Tuple[float, float]:
        """
        Численное нахождение границ α-среза методом бисекции (дихотомии).
        Решает уравнение μ(x) = α относительно x.
        Обоснование применимости бисекции: μ(x) непрерывна, строго унимодальна,
        знаки на концах полуинтервалов различны.
        """
        max_mu = self.mu_max()
        if alpha <= 0.0 or alpha > max_mu:
            raise ValueError(
                f"α={alpha:.4f} превышает максимум μ(x₀)={max_mu:.4f}. "
                f"Срез невозможен (величина не проходит порог отбора)."
            )
        if abs(alpha - max_mu) < 1e-12:
            return self.x0, self.x0

        def f(x: float) -> float:
            return self.membership(x) - alpha

        # --- Левая граница ---
        span, a, b = 4.0, self.x0 - 4.0 * self.sigma, self.x0
        for _ in range(60):
            if f(a) <= 0.0:
                break
            span *= 2.0
            a = self.x0 - span * self.sigma
        fa = f(a)
        while (b - a) > tolerance:
            mid = 0.5 * (a + b)
            fm = f(mid)
            if fm == 0.0:
                a = b = mid
                break
            if (fa < 0.0) != (fm < 0.0):
                b = mid
            else:
                a, fa = mid, fm
        x_left = 0.5 * (a + b)

        # --- Правая граница ---
        span, a, b = 4.0, self.x0, self.x0 + 4.0 * self.sigma
        for _ in range(60):
            if f(b) <= 0.0:
                break
            span *= 2.0
            b = self.x0 + span * self.sigma
        fb = f(b)
        while (b - a) > tolerance:
            mid = 0.5 * (a + b)
            fm = f(mid)
            if fm == 0.0:
                a = b = mid
                break
            if (fb < 0.0) != (fm < 0.0):
                a = mid
            else:
                b, fb = mid, fm
        x_right = 0.5 * (a + b)

        return x_left, x_right

    def alpha_cut_analytic(self, alpha: float) -> Tuple[float, float]:
        """
        Аналитическое решение μ(x) = α (для верификации численного метода).
        x_{L,R} = x₀ ∓ σ·√(-2·ln(α·σ·√(2π)))
        """
        max_mu = self.mu_max()
        if alpha > max_mu:
            raise ValueError(f"α={alpha:.4f} > μ_max={max_mu:.4f}.")
        arg = alpha * self.sigma * math.sqrt(2.0 * math.pi)
        half = self.sigma * math.sqrt(-2.0 * math.log(arg))
        return self.x0 - half, self.x0 + half

    # ------------------------------------------------------------------
    # Сравнение (правило зафиксировано)
    # ------------------------------------------------------------------
    def _cmp_key(self) -> Tuple[float, float]:
        return (self.x0, -self.sigma)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FuzzyNumber):
            return NotImplemented
        return (math.isclose(self.x0, other.x0, abs_tol=1e-12) and
                math.isclose(self.sigma, other.sigma, abs_tol=1e-12))

    def __lt__(self, other: "FuzzyNumber") -> bool:
        return self._cmp_key() < other._cmp_key()

    def __hash__(self) -> int:
        return hash((round(self.x0, 12), round(self.sigma, 12)))

    def __str__(self) -> str:
        label = f"{self.name}: " if self.name else ""
        return f"{label}(x₀={self.x0:.4f}; σ={self.sigma:.4f}; μ_max={self.mu_max():.4f})"

    def __repr__(self) -> str:
        return f"FuzzyNumber(x0={self.x0!r}, sigma={self.sigma!r}, name={self.name!r})"

    # ------------------------------------------------------------------
    # График
    # ------------------------------------------------------------------
    def plot(self, alpha: float | None = None, ax=None, label: str | None = None,
             span: float = 4.0, n_points: int = 400):
        """Рисует функцию принадлежности; при заданном alpha показывает срез."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4.5))

        left, right = self.x0 - span * self.sigma, self.x0 + span * self.sigma
        step = (right - left) / (n_points - 1)
        xs = [left + i * step for i in range(n_points)]
        ys = [self.membership(x) for x in xs]

        ax.plot(xs, ys, label=label or f"x₀={self.x0:.3f}, σ={self.sigma:.3f}")
        ax.axvline(self.x0, color="gray", linestyle=":", linewidth=0.8)

        if alpha is not None:
            try:
                xl, xr = self.alpha_cut(alpha)
                ax.axhline(alpha, color="red", linestyle="--", linewidth=1.0, label=f"α = {alpha}")
                ax.plot([xl, xr], [alpha, alpha], color="red", linewidth=3.0)
                ax.fill_between(xs, ys, 0, where=[y >= alpha for y in ys], alpha=0.2, color="red")
                ax.annotate(f"[{xl:.3f}; {xr:.3f}]", xy=((xl + xr) / 2, alpha),
                            xytext=(0, 10), textcoords="offset points",
                            ha="center", color="red", fontsize=9)
            except ValueError:
                ax.text(0.5, 0.5, f"α={alpha} > μ_max={self.mu_max():.3f}\nСрез невозможен",
                        ha='center', va='center', color='red', fontsize=12, transform=ax.transAxes)

        ax.set_xlabel("x"); ax.set_ylabel("μ(x)")
        ax.set_ylim(-0.03, max(1.08, 1.2 * self.mu_max()))
        ax.set_title("Нечёткая величина: μ(x) с коэффициентом плотности")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        return ax