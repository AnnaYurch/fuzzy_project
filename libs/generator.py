import random
import math
from typing import List
from .fuzzy_number import FuzzyNumber

class FuzzyGenerator:
    """Генератор набора нечетких величин"""
    
    @staticmethod
    def generate_normal_fuzzy_numbers(
        count: int,
        x0_range: tuple = (-10, 10),
        sigma_range: tuple = (0.5, 3.0),
        seed: int = None
    ) -> List[FuzzyNumber]:
        """
        Генерация набора нечетких величин с нормальным распределением.
        
        Параметры:
        - count: количество величин (100-300)
        - x0_range: диапазон для x0 (центр)
        - sigma_range: диапазон для sigma (разброс)
        - seed: seed для воспроизводимости
        """
        if not (100 <= count <= 300):
            raise ValueError("count должно быть в диапазоне от 100 до 300")
        
        if seed is not None:
            random.seed(seed)
        
        fuzzy_numbers = []
        for _ in range(count):
            x0 = random.uniform(*x0_range)
            sigma = random.uniform(*sigma_range)
            fuzzy_numbers.append(FuzzyNumber(x0, sigma))
        
        return fuzzy_numbers
    
    @staticmethod
    def generate_with_formula(
        count: int,
        n: int = 10,
        seed: int = None
    ) -> List[FuzzyNumber]:
        """
        Генерация по формулам из задания (для x0=0, sigma=1).
        Использует формулы для получения случайной величины с нормальным распределением.
        """
        if seed is not None:
            random.seed(seed)
        
        fuzzy_numbers = []
        for _ in range(count):
            # Генерация случайной величины по формуле 1 (n > 5)
            omega = [random.uniform(0, 1) for _ in range(n)]
            omega_n = sum(omega) / n
            
            # Формула 1: η = ω_n - (1/(20n)) * (ω_n³ - 3ω_n)
            eta = omega_n - (1.0 / (20 * n)) * (omega_n ** 3 - 3 * omega_n)
            
            # Преобразование для x0 != 0, sigma != 1
            # x' = sigma * eta + x0
            x0_target = random.uniform(-10, 10)
            sigma_target = random.uniform(0.5, 3.0)
            
            x0 = sigma_target * eta + x0_target
            
            fuzzy_numbers.append(FuzzyNumber(x0, sigma_target))
        
        return fuzzy_numbers