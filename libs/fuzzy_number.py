import numpy as np
import math
from typing import Tuple, List

class FuzzyNumber:
    def __init__(self, x0: float, sigma: float):
        if sigma <= 0:
            raise ValueError("sigma должно быть больше 0")
        self.x0 = x0
        self.sigma = sigma
    
    def membership_function(self, x: float) -> float:
        coefficient = 1.0 / math.sqrt(2 * math.pi * self.sigma)
        exponent = -((x - self.x0) ** 2) / (2 * self.sigma ** 2)
        return coefficient * math.exp(exponent)
    
    def get_alpha_cut(self, alpha: float) -> Tuple[float, float]:
        """
        Получение α-среза нечеткой величины.
        Возвращает интервал [x_left, x_right], где μ(x) >= alpha
        """
        if alpha <= 0 or alpha > self.membership_function(self.x0):
            raise ValueError(f"alpha должно быть в диапазоне (0, {self.membership_function(self.x0)}]")
        
        # Для нормального распределения: (x - x0)² = -2σ² × ln(α × √(2πσ))
        max_membership = self.membership_function(self.x0)
        ratio = alpha / max_membership
        
        if ratio >= 1:
            return (self.x0, self.x0)
        
        delta = self.sigma * math.sqrt(-2 * math.log(ratio))
        return (self.x0 - delta, self.x0 + delta)
    
    def __repr__(self) -> str:
        return f"FuzzyNumber(x0={self.x0:.4f}, sigma={self.sigma:.4f})"
    
    def __str__(self) -> str:
        return f"N({self.x0:.4f}, {self.sigma:.4f})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, FuzzyNumber):
            return False
        return math.isclose(self.x0, other.x0, abs_tol=1e-6) and \
               math.isclose(self.sigma, other.sigma, abs_tol=1e-6)
    
    def distance_to(self, other: 'FuzzyNumber') -> float:
        """
        Расстояние между нечеткими величинами (евклидово расстояние между центрами).
        """
        return abs(self.x0 - other.x0)
    
    def to_dict(self) -> dict:
        """Сериализация в словарь"""
        return {'x0': self.x0, 'sigma': self.sigma}
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FuzzyNumber':
        """Десериализация из словаря"""
        return cls(data['x0'], data['sigma'])