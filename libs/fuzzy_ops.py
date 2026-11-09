"""
Манипулирование нечёткими величинами (п.4 задания).

Математический аппарат:
  Сложение/вычитание — точный результат по принципу расширения Заде:
    x₀(A±B) = x₀A ± x₀B
    σ(A±B) = √(σA² + σB²)

  Умножение/деление — линеаризация (ряд Тейлора 1-го порядка):
    σ_f² = (∂f/∂A)²·σA² + (∂f/∂B)²·σB²

Правило сравнения:
  1) больше та, у которой больше x₀;
  2) при равных x₀ «больше» та, у которой МЕНЬШЕ σ.
"""
from __future__ import annotations
import math
from typing import Union
from .fuzzy_number import FuzzyNumber
from .logger import get_logger

log = get_logger("ops")
Scalar = Union[int, float]
REL_LIMIT, EPS = 3.0, 1e-12

def _check(value: FuzzyNumber, argname: str) -> FuzzyNumber:
    if not isinstance(value, FuzzyNumber):
        raise TypeError(f"{argname} должно быть FuzzyNumber.")
    return value

def add(a: FuzzyNumber, b: FuzzyNumber) -> FuzzyNumber:
    _check(a, "A"); _check(b, "B")
    res = FuzzyNumber(a.x0 + b.x0, math.hypot(a.sigma, b.sigma))
    log.debug("ADD: %s + %s = %s", a, b, res)
    return res

def subtract(a: FuzzyNumber, b: FuzzyNumber) -> FuzzyNumber:
    _check(a, "A"); _check(b, "B")
    res = FuzzyNumber(a.x0 - b.x0, math.hypot(a.sigma, b.sigma))
    log.debug("SUB: %s - %s = %s", a, b, res)
    return res

def multiply(a: FuzzyNumber, b: FuzzyNumber) -> FuzzyNumber:
    _check(a, "A"); _check(b, "B")
    sigma = math.hypot(b.x0 * a.sigma, a.x0 * b.sigma)
    res = FuzzyNumber(a.x0 * b.x0, sigma if sigma > 0.0 else EPS)
    log.debug("MUL: %s * %s = %s (линеаризация)", a, b, res)
    return res

def divide(a: FuzzyNumber, b: FuzzyNumber) -> FuzzyNumber:
    _check(a, "A"); _check(b, "B")
    if abs(b.x0) < EPS:
        raise ZeroDivisionError("Деление невозможно: центр делителя x₀ = 0.")
    if abs(b.x0) < REL_LIMIT * b.sigma:
        log.warning("Деление: носитель делителя захватывает 0, линеаризация приближённая.")
    sigma = math.hypot(a.sigma / b.x0, a.x0 * b.sigma / (b.x0 ** 2))
    res = FuzzyNumber(a.x0 / b.x0, sigma if sigma > 0.0 else EPS)
    log.debug("DIV: %s / %s = %s (линеаризация)", a, b, res)
    return res

def multiply_by_scalar(a: FuzzyNumber, k: Scalar) -> FuzzyNumber:
    _check(a, "A")
    k = float(k)
    if k == 0.0:
        raise ValueError("Умножение на 0 даёт вырожденную величину.")
    res = FuzzyNumber(k * a.x0, abs(k) * a.sigma)
    log.debug("SCALE: %s * %.4f = %s", a, k, res)
    return res

def compare(a: FuzzyNumber, b: FuzzyNumber) -> int:
    _check(a, "A"); _check(b, "B")
    if not math.isclose(a.x0, b.x0, rel_tol=0.0, abs_tol=EPS):
        return 1 if a.x0 > b.x0 else -1
    if math.isclose(a.sigma, b.sigma, rel_tol=0.0, abs_tol=EPS):
        return 0
    return 1 if a.sigma < b.sigma else -1

def compare_str(a: FuzzyNumber, b: FuzzyNumber) -> str:
    sign = compare(a, b)
    symbol = {1: ">", 0: "==", -1: "<"}[sign]
    if sign != 0 and math.isclose(a.x0, b.x0, abs_tol=EPS):
        reason = "центры равны → больше та, у которой меньше σ"
    elif sign == 0:
        reason = "совпадают и центр, и разброс"
    else:
        reason = "сравнение по центру x₀"
    return f"{a} {symbol} {b}   ({reason})"