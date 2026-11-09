"""
Точка входа. Консольная экранная форма + графики matplotlib.

Демонстрационный сценарий (по требованиям преподавателя):
  1. Сгенерировать множество (100-300 величин).
  2. Кластеризация на полном множестве (например, 5 кластеров).
  3. Отбор подмножества по α-порогу (например, α=0.43).
  4. Кластеризация на подмножестве — показать «схлопывание» кластеров.
  5. Вывод: программа работает с ЛЮБЫМИ предельно допустимыми множествами.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from libs import (
    FuzzyNumber, add, subtract, multiply, divide, multiply_by_scalar, compare_str,
    generate_fuzzy_set, generate_uniform_fuzzy_set, check_generator_moments,
    save_to_tsv, load_from_tsv, calculate_distance, perform_clustering,
    elbow_method, gmm_bic, cluster_by_alpha_overlap, alpha_sweep,
    filter_by_alpha_threshold, clustering_on_subset, get_logger
)

log = get_logger("main")
DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DATASET = DATA_DIR / "dataset.tsv"
LINE, THIN = "=" * 78, "-" * 78

def pause() -> None:
    input("\n  Нажмите Enter, чтобы вернуться в меню...")

def ask_float(prompt: str, default: float | None = None) -> float:
    while True:
        raw = input(f"  {prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if not raw and default is not None:
            return float(default)
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            print("  ! Нужно ввести число.")

def ask_int(prompt: str, default: int | None = None) -> int:
    while True:
        raw = input(f"  {prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if not raw and default is not None:
            return int(default)
        try:
            return int(raw)
        except ValueError:
            print("  ! Нужно ввести целое число.")

def require_data(state: dict, minimum: int = 1) -> bool:
    if len(state["data"]) < minimum:
        print(f"\n  ! Нужно минимум {minimum} величин. Сейчас: {len(state['data'])}.")
        return False
    return True

def header(state: dict) -> None:
    print("\n" + LINE)
    print(" КОМПЛЕКС РАБОТЫ С НЕЧЁТКИМИ ВЕЛИЧИНАМИ И ИХ КЛАСТЕРНОГО АНАЛИЗА")
    print(" μ(x) = (1/(σ√(2π)))·exp(-(x-x₀)²/(2σ²))   |   α — порог отбора")
    print(LINE)
    print(f" Величин: {len(state['data'])}   |   α = {state['alpha']}   |   k = {state['k']}")
    print(LINE)

def menu() -> None:
    print("""
  --- ЭКРАННАЯ ФОРМА / ГЛАВНОЕ МЕНЮ ---
   1. Сгенерировать множество нечётких величин (100-300)
   2. Загрузить множество из файла TSV
   3. Сохранить множество в файл TSV
   4. Показать список нечётких величин (с μ_max)
   5. Добавить нечёткую величину вручную
   6. α-срез одной величины (численно, метод бисекции)
   7. Манипулирование: +, −, ×, ÷, ×скаляр, сравнение
   8. Кластерный анализ: K-Means на заданное k
   9. Подбор числа кластеров: метод локтя + GMM/BIC
  10. ★ КЛЮЧЕВОЕ: α-отбор подмножества + кластеризация (демо «схлопывания»)
  11. Кластеризация по перекрытию α-срезов
  12. Таблица влияния α на число кластеров
  13. Графики (функции принадлежности, кластеры, α-отбор)
  14. Проверка генератора: среднее, дисперсия, асимметрия, эксцесс
   0. Выход
""")

# --- Действия ---
def action_generate(state):
    print("\n  --- Генерация множества ---")
    mode = ask_int("Режим (1: сгруппированные, 2: равномерные)", 1)
    count = ask_int("Количество (100-300)", 150)
    seed = ask_int("Зерно ГСЧ (seed)", 42)
    n_uniform = ask_int("Число равномерных слагаемых n", 12)
    if mode == 2:
        state["data"] = generate_uniform_fuzzy_set(count, seed=seed)
    else:
        groups = ask_int("Число «истинных» групп", 3)
        state["data"] = generate_fuzzy_set(count, groups=groups,
                                           n_uniform=n_uniform, seed=seed)
    state["result"] = None
    print(f"\n  ✓ Готово. Сгенерировано {len(state['data'])} нечётких величин.")
    print("  μ_max варьируется от ≈0.1 до ≈0.8 — можно демонстрировать α-отбор.")

def action_load(state):
    path = input(f"  Путь к файлу [{DEFAULT_DATASET}]: ").strip() or str(DEFAULT_DATASET)
    try:
        state["data"] = load_from_tsv(path)
        state["result"] = None
        print(f"  ✓ Загружено {len(state['data'])} величин.")
    except Exception as exc:
        print(f"  ! Ошибка: {exc}")

def action_save(state):
    if not require_data(state): return
    path = input(f"  Путь к файлу [{DEFAULT_DATASET}]: ").strip() or str(DEFAULT_DATASET)
    try:
        save_to_tsv(state["data"], path)
        print(f"  ✓ Сохранено: {path}")
    except Exception as exc:
        print(f"  ! Ошибка: {exc}")

def action_list(state):
    if not require_data(state): return
    data, limit = state["data"], ask_int("Сколько строк показать (0 — все)", 20)
    shown = data if limit == 0 else data[:limit]
    labels = state["result"]["labels"] if state.get("result") else None
    print(f"\n  --- Множество нечётких величин (всего {len(data)}) ---")
    print(f"  {'№':<5}{'Имя':<8}{'x₀':>10}{'σ':>10}{'μ_max':>10}" + (f"{'Кластер':>10}" if labels else ""))
    print("  " + THIN)
    for i, fn in enumerate(shown):
        row = f"  {i + 1:<5}{fn.name:<8}{fn.x0:>10.4f}{fn.sigma:>10.4f}{fn.mu_max():>10.4f}"
        if labels: row += f"{labels[i]:>10}"
        print(row)

def action_add(state):
    print("\n  --- Экранная форма ввода нечёткой величины ---")
    x0, sigma = ask_float("Центр x₀"), ask_float("Разброс σ (> 0)")
    try:
        fn = FuzzyNumber(x0, sigma, name=f"A{len(state['data']) + 1}")
        state["data"].append(fn)
        state["result"] = None
        print(f"  ✓ Добавлено: {fn}")
    except ValueError as exc:
        print(f"  ! Ошибка: {exc}")

def action_alpha_one(state):
    if not require_data(state): return
    idx = ask_int(f"Номер величины (1-{len(state['data'])})", 1) - 1
    if not (0 <= idx < len(state["data"])): return
    alpha = ask_float("Уровень α (0 < α ≤ μ_max)", state["alpha"])
    try:
        fn = state["data"][idx]
        left, right = fn.alpha_cut(alpha)
        a_left, a_right = fn.alpha_cut_analytic(alpha)
        state["alpha"] = alpha
        print(f"\n  Величина: {fn}")
        print(f"  Уровень α = {alpha}")
        print("  " + THIN)
        print(f"  Численно (бисекция):   [{left:.8f}; {right:.8f}]")
        print(f"  Аналитически (проверка):[{a_left:.8f}; {a_right:.8f}]")
        print(f"  Расхождение:            {max(abs(left - a_left), abs(right - a_right)):.2e}")
        if input("\n  Показать график? (д/н): ").strip().lower() in ("д", "y", "да", "yes"):
            fn.plot(alpha=alpha); plt.tight_layout(); plt.show()
    except ValueError as exc:
        print(f"  ! Ошибка: {exc}")

def action_arithmetic(state):
    if not require_data(state, 2): return
    ia, ib = ask_int(f"Номер A (1-{len(state['data'])})", 1) - 1, ask_int(f"Номер B (1-{len(state['data'])})", 2) - 1
    if not (0 <= ia < len(state["data"]) and 0 <= ib < len(state["data"])): return
    a, b, k = state["data"][ia], state["data"][ib], ask_float("Скаляр k для умножения", 2.0)
    print(f"\n  A = {a}\n  B = {b}\n  " + THIN)
    print(f"  A + B      = {add(a, b)}")
    print(f"  A − B      = {subtract(a, b)}")
    print(f"  A × B      = {multiply(a, b)}    (линеаризация)")
    try: print(f"  A / B      = {divide(a, b)}    (линеаризация)")
    except ZeroDivisionError as exc: print(f"  A / B      → ошибка: {exc}")
    print(f"  A × {k:<6} = {multiply_by_scalar(a, k)}")
    print("  " + THIN + f"\n  Сравнение: {compare_str(a, b)}\n  Расстояние d(A,B) = {calculate_distance(a, b):.6f}")

def action_kmeans(state):
    if not require_data(state, 2): return
    k = ask_int(f"Число кластеров k (1-{len(state['data'])})", state["k"])
    try:
        res = perform_clustering(state["data"], k)
        state["k"], state["result"] = k, res
        print(f"\n  ✓ {res['method']}\n  Инерция J = {res['inertia']:.4f}\n\n  Сводка по кластерам:")
        print(f"  {'Кластер':<10}{'Размер':>8}{'средн. x₀':>14}{'средн. σ':>14}{'диапазон x₀':>26}")
        print("  " + THIN)
        for row in res["summary"]:
            print(f"  {row['cluster']:<10}{row['size']:>8}{row['mean_x0']:>14.4f}"
                  f"{row['mean_sigma']:>14.4f}[{row['min_x0']:.2f}; {row['max_x0']:.2f}]")
    except ValueError as exc:
        print(f"  ! Ошибка: {exc}")

def action_choose_k(state):
    if not require_data(state, 2): return
    max_k = ask_int("Максимальное проверяемое k", 10)
    try:
        ks, inertias = elbow_method(state["data"], max_k)
        ks2, bics, best_k = gmm_bic(state["data"], max_k)
        state["k"] = best_k
        print(f"\n  {'k':<6}{'Инерция J (локоть)':>24}{'BIC (GMM)':>20}\n  " + THIN)
        for i, k in enumerate(ks):
            bic_val = bics[i] if i < len(bics) else float("nan")
            mark = "  ← минимум BIC" if k == best_k else ""
            print(f"  {k:<6}{inertias[i]:>24.4f}{bic_val:>20.4f}{mark}")
        print("  " + THIN + f"\n  Оптимальное число кластеров по BIC: k = {best_k}")
        if input("\n  Показать графики? (д/н): ").strip().lower() in ("д", "y", "да", "yes"):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            axes[0].plot(ks, inertias, "o-"); axes[0].set_title("Метод локтя")
            axes[0].set_xlabel("k"); axes[0].set_ylabel("Инерция J"); axes[0].grid(alpha=0.3)
            axes[1].plot(ks2, bics, "s-", color="darkorange")
            axes[1].axvline(best_k, color="red", linestyle="--", label=f"k={best_k}")
            axes[1].set_title("BIC (GMM)"); axes[1].set_xlabel("k"); axes[1].set_ylabel("BIC")
            axes[1].grid(alpha=0.3); axes[1].legend()
            plt.tight_layout(); plt.show()
    except ValueError as exc:
        print(f"  ! Ошибка: {exc}")

def action_alpha_subset_clustering(state):
    """
    ★ КЛЮЧЕВОЕ ДЕЙСТВИЕ — демонстрация по требованиям преподавателя:
      1. Кластеризация полного множества на k кластеров.
      2. Отбор подмножества по α-порогу.
      3. Кластеризация подмножества на те же k кластеров.
      4. Сравнение: «схлопывание» кластеров.
    """
    if not require_data(state, 5): return
    print("\n  " + "=" * 70)
    print("  ★ ДЕМО: α-отбор подмножества + кластеризация")
    print("  " + "=" * 70)

    k = ask_int(f"Число кластеров k (1-{len(state['data'])})", state["k"])
    alpha = ask_float("α-порог отбора (0 < α ≤ μ_max)", state["alpha"])
    state["k"], state["alpha"] = k, alpha

    try:
        result = clustering_on_subset(state["data"], alpha, k)
        state["result"] = result["full"]

        print(f"\n  {'='*70}")
        print(f"  ПОЛНОЕ МНОЖЕСТВО: {len(state['data'])} величин")
        print(f"  Кластеризация на k = {k}")
        print(f"  {'='*70}")
        print(f"  {'Кластер':<10}{'Размер':>8}{'средн. x₀':>14}{'средн. σ':>14}")
        print("  " + THIN)
        for row in result["full"]["summary"]:
            print(f"  {row['cluster']:<10}{row['size']:>8}{row['mean_x0']:>14.4f}{row['mean_sigma']:>14.4f}")

        print(f"\n  {'='*70}")
        print(f"  α-ОТБОР: порог α = {alpha}")
        print(f"  Критерий: μ(x₀) = 1/(σ·√(2π)) ≥ α")
        print(f"  Было: {len(state['data'])} → Стало: {result['subset_size']}")
        print(f"  {'='*70}")

        if result["subset_result"] is None:
            print(f"\n  ! {result['message']}")
        else:
            print(f"\n  ПОДМНОЖЕСТВО: {result['subset_size']} величин")
            print(f"  Кластеризация на те же k = {k}")
            print(f"  {'Кластер':<10}{'Размер':>8}{'средн. x₀':>14}{'средн. σ':>14}")
            print("  " + THIN)
            for row in result["subset_result"]["summary"]:
                print(f"  {row['cluster']:<10}{row['size']:>8}{row['mean_x0']:>14.4f}{row['mean_sigma']:>14.4f}")

            print(f"\n  {'='*70}")
            print(f"  ВЫВОД: при α = {alpha} из {len(state['data'])} величин")
            print(f"  осталось {result['subset_size']} — те, у кого μ_max ≥ {alpha}.")
            print(f"  Кластеры перестроились: часть величин «растворилась».")
            print(f"  Программа работает с ЛЮБЫМИ предельно допустимыми множествами.")
            print(f"  {'='*70}")

        if input("\n  Показать графики? (д/н): ").strip().lower() in ("д", "y", "да", "yes"):
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            X_full = np.array([[fn.x0, fn.sigma] for fn in state["data"]])
            axes[0].scatter(X_full[:, 0], X_full[:, 1],
                            c=result["full"]["labels"], cmap="tab10", s=40, alpha=0.8)
            axes[0].set_title(f"Полное множество ({len(state['data'])} вел.), k={k}")
            axes[0].set_xlabel("x₀"); axes[0].set_ylabel("σ"); axes[0].grid(alpha=0.3)

            if result["subset_result"]:
                subset = filter_by_alpha_threshold(state["data"], alpha)
                X_sub = np.array([[fn.x0, fn.sigma] for fn in subset])
                axes[1].scatter(X_sub[:, 0], X_sub[:, 1],
                                c=result["subset_result"]["labels"], cmap="tab10", s=60, alpha=0.9, edgecolors='k')
                axes[1].set_title(f"Подмножество ({result['subset_size']} вел.), α={alpha}")
                axes[1].set_xlabel("x₀"); axes[1].set_ylabel("σ"); axes[1].grid(alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, result["message"], ha='center', va='center',
                             transform=axes[1].transAxes, fontsize=12, color='red')
            plt.tight_layout(); plt.show()
    except ValueError as exc:
        print(f"  ! Ошибка: {exc}")

def action_alpha_cluster(state):
    if not require_data(state, 2): return
    alpha = ask_float("Уровень α (0 < α ≤ μ_max)", state["alpha"])
    try:
        res = cluster_by_alpha_overlap(state["data"], alpha)
        state["alpha"], state["result"] = alpha, res
        print(f"\n  ✓ {res['method']}\n  Получилось кластеров: {res['k']}")
        print(f"\n  {'Кластер':<10}{'Размер':>8}{'средн. x₀':>14}{'средн. σ':>14}\n  " + THIN)
        for row in res["summary"][:25]:
            print(f"  {row['cluster']:<10}{row['size']:>8}{row['mean_x0']:>14.4f}{row['mean_sigma']:>14.4f}")
    except ValueError as exc:
        print(f"  ! Ошибка: {exc}")

def action_alpha_sweep(state):
    if not require_data(state, 2): return
    rows = alpha_sweep(state["data"])
    print(f"\n  --- Влияние α на отбор и кластеризацию ---")
    print(f"  {'α':<10}{'Прошло отбор':>16}{'Кластеров':>14}\n  " + THIN)
    for row in rows:
        print(f"  {row['alpha']:<10.2f}{row['passed']:>16}{row['k']:>14}")
    print("  " + THIN + "\n  Чем больше α, тем строже отбор → меньше величин → меньше кластеров.")

def action_plots(state):
    if not require_data(state): return
    mode = ask_int("Что показать (1: μ(x), 2: кластеры, 3: α-отбор)", 2)
    data, alpha = state["data"], state["alpha"]
    if mode == 1:
        count = min(ask_int("Сколько функций показать", 3), 6)
        fig, axes = plt.subplots(count, 1, figsize=(8, 3 * count), squeeze=False)
        for i in range(count): data[i].plot(alpha=alpha, ax=axes[i][0])
        plt.tight_layout(); plt.show()
    elif mode == 2:
        labels = state["result"]["labels"] if state.get("result") else None
        X = np.array([[fn.x0, fn.sigma] for fn in data])
        plt.figure(figsize=(9, 6))
        if labels:
            plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=40, alpha=0.85)
            plt.colorbar(label="Кластер")
        else:
            plt.scatter(X[:, 0], X[:, 1], s=40, alpha=0.8)
        plt.xlabel("x₀"); plt.ylabel("σ"); plt.grid(alpha=0.3)
        plt.title("Множество нечётких величин в пространстве (x₀, σ)")
        plt.tight_layout(); plt.show()
    else:
        passed = [fn for fn in data if fn.passes_alpha_threshold(alpha)]
        failed = [fn for fn in data if not fn.passes_alpha_threshold(alpha)]
        plt.figure(figsize=(9, 6))
        if passed:
            X_p = np.array([[fn.x0, fn.sigma] for fn in passed])
            plt.scatter(X_p[:, 0], X_p[:, 1], c="green", s=60, alpha=0.8, label=f"Прошли α={alpha} ({len(passed)})")
        if failed:
            X_f = np.array([[fn.x0, fn.sigma] for fn in failed])
            plt.scatter(X_f[:, 0], X_f[:, 1], c="red", s=60, alpha=0.5, marker="x", label=f"Отсеяны ({len(failed)})")
        plt.xlabel("x₀"); plt.ylabel("σ"); plt.grid(alpha=0.3); plt.legend()
        plt.title(f"α-отбор: порог α = {alpha}")
        plt.tight_layout(); plt.show()

def action_generator_check(state):
    print("\n  --- Проверка генератора нормальных чисел ---")
    n, size = ask_int("Число равномерных слагаемых n", 12), ask_int("Размер выборки", 20000)
    stats = check_generator_moments(size, n)
    print(f"\n  Формула: ω = (Σξ − n/2) / √(n/12),  ξ ~ U[0,1]")
    print(f"  Поправка: {'H₃ (n≥5)' if n >= 5 else 'H₅ (n<5)'}\n  " + THIN)
    print(f"  {'Размер выборки':<28}{stats['sample_size']:>14}")
    print(f"  {'Среднее (ожидается 0)':<28}{stats['mean']:>14.5f}")
    print(f"  {'СКО (ожидается 1)':<28}{stats['std']:>14.5f}")
    print(f"  {'Дисперсия (ожидается 1)':<28}{stats['variance']:>14.5f}")
    print(f"  {'Асимметрия (ожидается 0)':<28}{stats['skewness']:>14.5f}")
    print(f"  {'Эксцесс (ожидается 0)':<28}{stats['excess_kurtosis']:>14.5f}\n  " + THIN)

ACTIONS = {
    "1": action_generate, "2": action_load, "3": action_save, "4": action_list,
    "5": action_add, "6": action_alpha_one, "7": action_arithmetic,
    "8": action_kmeans, "9": action_choose_k, "10": action_alpha_subset_clustering,
    "11": action_alpha_cluster, "12": action_alpha_sweep,
    "13": action_plots, "14": action_generator_check,
}

def run_demo():
    print(LINE + "\n АВТОМАТИЧЕСКАЯ ДЕМОНСТРАЦИЯ (сценарий преподавателя)\n" + LINE)
    print("\n[1] Генерация 150 нечётких величин...")
    data = generate_fuzzy_set(150, groups=3, n_uniform=12, seed=42)
    print(f"  Сгенерировано: {len(data)}")
    print(f"  μ_max варьируется: min={min(fn.mu_max() for fn in data):.4f}, max={max(fn.mu_max() for fn in data):.4f}")

    print("\n[2] Кластеризация полного множества на k=5...")
    res_full = perform_clustering(data, 5)
    print(f"  Кластеров: {len(res_full['summary'])}")
    for row in res_full["summary"]:
        print(f"    Кластер {row['cluster']}: {row['size']} вел.")

    print("\n[3] α-отбор подмножества при α=0.43...")
    alpha = 0.43
    subset = filter_by_alpha_threshold(data, alpha)
    print(f"  Было: {len(data)} → Стало: {len(subset)} (по критерию μ_max ≥ {alpha})")

    print("\n[4] Кластеризация подмножества на те же k=5...")
    if len(subset) >= 5:
        res_sub = perform_clustering(subset, 5)
        print(f"  Кластеров: {len(res_sub['summary'])}")
        for row in res_sub["summary"]:
            print(f"    Кластер {row['cluster']}: {row['size']} вел.")
        print(f"\n  ВЫВОД: программа работает с ЛЮБЫМИ предельно допустимыми множествами.")
    else:
        print(f"  Подмножество слишком мало ({len(subset)} < 5)")

    save_to_tsv(data, DEFAULT_DATASET)
    print(f"\n  Набор сохранён в {DEFAULT_DATASET}\n" + LINE)

def main() -> None:
    if "--demo" in sys.argv:
        run_demo()
        return
    state = {"data": [], "alpha": 0.43, "k": 5, "result": None}
    log.info("Запуск комплекса в интерактивном режиме")
    while True:
        header(state); menu()
        choice = input("  Выберите действие (0-14): ").strip()
        if choice == "0":
            print("\n  Завершение работы. Лог: logs/app.log\n")
            log.info("Завершение работы программы")
            break
        action = ACTIONS.get(choice)
        if action is None:
            print("  ! Неверный пункт меню."); pause(); continue
        try:
            action(state)
        except Exception as exc:
            log.exception("Ошибка при выполнении пункта %s", choice)
            print(f"\n  ! Ошибка: {exc}")
        pause()

if __name__ == "__main__":
    main()