import math
from .fuzzy_number import FuzzyNumber

class FuzzyOperations:
    """Операции над нечеткими величинами с нормальным распределением"""
    
    @staticmethod
    def addition(f1: FuzzyNumber, f2: FuzzyNumber) -> FuzzyNumber:
        """
        Сложение нечетких величин.
        Для нормального распределения:
        x0 = x0_1 + x0_2
        sigma = sqrt(sigma_1² + sigma_2²)
        """
        new_x0 = f1.x0 + f2.x0
        new_sigma = math.sqrt(f1.sigma ** 2 + f2.sigma ** 2)
        return FuzzyNumber(new_x0, new_sigma)
    
    @staticmethod
    def subtraction(f1: FuzzyNumber, f2: FuzzyNumber) -> FuzzyNumber:
        """
        Вычитание нечетких величин.
        x0 = x0_1 - x0_2
        sigma = sqrt(sigma_1² + sigma_2²)
        """
        new_x0 = f1.x0 - f2.x0
        new_sigma = math.sqrt(f1.sigma ** 2 + f2.sigma ** 2)
        return FuzzyNumber(new_x0, new_sigma)
    
    @staticmethod
    def multiplication_by_constant(f: FuzzyNumber, c: float) -> FuzzyNumber:
        """
        Умножение нечеткой величины на константу.
        x0 = c * x0
        sigma = |c| * sigma
        """
        new_x0 = c * f.x0
        new_sigma = abs(c) * f.sigma
        return FuzzyNumber(new_x0, new_sigma)
    
    @staticmethod
    def division_by_constant(f: FuzzyNumber, c: float) -> FuzzyNumber:
        """
        Деление нечеткой величины на константу.
        x0 = x0 / c
        sigma = sigma / |c|
        """
        if c == 0:
            raise ValueError("Деление на ноль")
        new_x0 = f.x0 / c
        new_sigma = f.sigma / abs(c)
        return FuzzyNumber(new_x0, new_sigma)
    
    @staticmethod
    def comparison(f1: FuzzyNumber, f2: FuzzyNumber) -> str:
        """
        Сравнение нечетких величин по центрам.
        Возвращает: 'greater', 'less', 'equal'
        """
        diff = f1.x0 - f2.x0
        if abs(diff) < 1e-6:
            return 'equal'
        elif diff > 0:
            return 'greater'
        else:
            return 'less'
    
    @staticmethod
    def negation(f: FuzzyNumber) -> FuzzyNumber:
        """
        Отрицание нечеткой величины.
        x0 = -x0
        sigma остается неизменным
        """
        return FuzzyNumber(-f.x0, f.sigma)