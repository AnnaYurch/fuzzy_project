import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from libs.fuzzy_number import FuzzyNumber
from libs.fuzzy_ops import FuzzyOperations
from libs.generator import FuzzyGenerator
from libs.clustering import FuzzyClustering
from libs.logger import FuzzyLogger
import matplotlib.pyplot as plt
import numpy as np

class FuzzyApp:
    """Главное приложение для работы с нечеткими величинами"""
    
    def __init__(self):
        self.logger = FuzzyLogger()
        self.fuzzy_numbers = []
        self.clusters = {}
        self.ops = FuzzyOperations()
    
    def menu(self):
        """Главное меню программы"""
        while True:
            print("\n" + "="*60)
            print("СИСТЕМА РАБОТЫ С НЕЧЕТКИМИ ВЕЛИЧИНАМИ")
            print("="*60)
            print("1. Сгенерировать набор нечетких величин")
            print("2. Показать статистику набора")
            print("3. Выполнить арифметические операции")
            print("4. Провести кластерный анализ")
            print("5. Визуализировать результаты")
            print("6. Показать функцию принадлежности")
            print("0. Выход")
            print("="*60)
            
            choice = input("Выберите действие: ").strip()
            
            if choice == '1':
                self.generate_fuzzy_numbers()
            elif choice == '2':
                self.show_statistics()
            elif choice == '3':
                self.perform_operations()
            elif choice == '4':
                self.perform_clustering()
            elif choice == '5':
                self.visualize_results()
            elif choice == '6':
                self.show_membership_function()
            elif choice == '0':
                self.logger.info("Программа завершена")
                print("До свидания!")
                break
            else:
                print("Неверный выбор. Попробуйте снова.")
    
    def generate_fuzzy_numbers(self):
        """Генерация набора нечетких величин"""
        print("\nГенерация нечетких величин")
        print("-" * 40)
        
        try:
            count = int(input("Количество величин (100-300): "))
            if not (100 <= count <= 300):
                print("Ошибка: количество должно быть от 100 до 300")
                return
            
            method = input("Метод генерации (1 - случайный, 2 - по формуле): ").strip()
            
            if method == '1':
                self.fuzzy_numbers = FuzzyGenerator.generate_normal_fuzzy_numbers(
                    count, seed=42
                )
            elif method == '2':
                n = int(input("Параметр n для формулы (>5): "))
                if n <= 5:
                    print("Ошибка: n должно быть больше 5")
                    return
                self.fuzzy_numbers = FuzzyGenerator.generate_with_formula(
                    count, n=n, seed=42
                )
            else:
                print("Неверный метод")
                return
            
            self.logger.info(f"Сгенерировано {count} нечетких величин")
            print(f"✓ Сгенерировано {count} нечетких величин")
            print(f"Примеры: {self.fuzzy_numbers[:5]}")
            
        except ValueError as e:
            print(f"Ошибка: {e}")
    
    def show_statistics(self):
        """Показать статистику набора"""
        if not self.fuzzy_numbers:
            print("Сначала сгенерируйте нечеткие величины")
            return
        
        print("\nСтатистика набора нечетких величин")
        print("-" * 40)
        
        centers = [fn.x0 for fn in self.fuzzy_numbers]
        sigmas = [fn.sigma for fn in self.fuzzy_numbers]
        
        print(f"Количество: {len(self.fuzzy_numbers)}")
        print(f"Центры (x0):")
        print(f"  Мин: {min(centers):.4f}")
        print(f"  Макс: {max(centers):.4f}")
        print(f"  Среднее: {sum(centers)/len(centers):.4f}")
        print(f"Разбросы (sigma):")
        print(f"  Мин: {min(sigmas):.4f}")
        print(f"  Макс: {max(sigmas):.4f}")
        print(f"  Среднее: {sum(sigmas)/len(sigmas):.4f}")
    
    def perform_operations(self):
        """Выполнение арифметических операций"""
        if len(self.fuzzy_numbers) < 2:
            print("Нужно минимум 2 нечеткие величины")
            return
        
        print("\nАрифметические операции")
        print("-" * 40)
        
        f1 = self.fuzzy_numbers[0]
        f2 = self.fuzzy_numbers[1]
        
        print(f"f1 = {f1}")
        print(f"f2 = {f2}")
        print()
        
        # Сложение
        result = self.ops.addition(f1, f2)
        print(f"f1 + f2 = {result}")
        self.logger.log_fuzzy_number(result, "Сложение")
        
        # Вычитание
        result = self.ops.subtraction(f1, f2)
        print(f"f1 - f2 = {result}")
        self.logger.log_fuzzy_number(result, "Вычитание")
        
        # Умножение на константу
        c = 2.5
        result = self.ops.multiplication_by_constant(f1, c)
        print(f"{c} * f1 = {result}")
        self.logger.log_fuzzy_number(result, f"Умножение на {c}")
        
        # Деление на константу
        result = self.ops.division_by_constant(f1, c)
        print(f"f1 / {c} = {result}")
        self.logger.log_fuzzy_number(result, f"Деление на {c}")
        
        # Сравнение
        comp = self.ops.comparison(f1, f2)
        print(f"Сравнение f1 и f2: {comp}")
    
    def perform_clustering(self):
        """Проведение кластерного анализа"""
        if not self.fuzzy_numbers:
            print("Сначала сгенерируйте нечеткие величины")
            return
        
        print("\nКластерный анализ")
        print("-" * 40)
        
        try:
            k = int(input("Количество кластеров: "))
            method = input("Метод (1 - K-средних, 2 - иерархический): ").strip()
            
            if method == '1':
                self.clusters = FuzzyClustering.k_means_clustering(
                    self.fuzzy_numbers, k
                )
            elif method == '2':
                self.clusters = FuzzyClustering.hierarchical_clustering(
                    self.fuzzy_numbers, k
                )
            else:
                print("Неверный метод")
                return
            
            self.logger.info(f"Кластеризация завершена: {k} кластеров")
            
            # Статистика по кластерам
            stats = FuzzyClustering.get_cluster_statistics(self.clusters)
            
            print(f"\nРезультаты кластеризации ({k} кластеров):")
            for cluster_id, items in sorted(self.clusters.items()):
                print(f"\nКластер {cluster_id}: {len(items)} элементов")
                if cluster_id in stats:
                    s = stats[cluster_id]
                    print(f"  Средний x0: {s['mean_x0']:.4f}")
                    print(f"  Средний sigma: {s['mean_sigma']:.4f}")
                    print(f"  Диапазон x0: [{s['min_x0']:.4f}, {s['max_x0']:.4f}]")
                    self.logger.log_cluster_result(cluster_id, len(items), s)
            
        except ValueError as e:
            print(f"Ошибка: {e}")
    
    def visualize_results(self):
        """Визуализация результатов"""
        if not self.fuzzy_numbers:
            print("Сначала сгенерируйте нечеткие величины")
            return
        
        print("\nВизуализация")
        print("-" * 40)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Распределение центров
        centers = [fn.x0 for fn in self.fuzzy_numbers]
        axes[0, 0].hist(centers, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Распределение центров (x0)')
        axes[0, 0].set_xlabel('x0')
        axes[0, 0].set_ylabel('Частота')
        
        # 2. Распределение разбросов
        sigmas = [fn.sigma for fn in self.fuzzy_numbers]
        axes[0, 1].hist(sigmas, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_title('Распределение разбросов (sigma)')
        axes[0, 1].set_xlabel('sigma')
        axes[0, 1].set_ylabel('Частота')
        
        # 3. Кластеры (если есть)
        if self.clusters:
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
            for cluster_id, items in sorted(self.clusters.items()):
                cluster_centers = [fn.x0 for fn in items]
                color = colors[cluster_id % len(colors)]
                axes[1, 0].scatter(
                    [cluster_id] * len(cluster_centers),
                    cluster_centers,
                    alpha=0.6,
                    color=color,
                    label=f'Кластер {cluster_id}'
                )
            axes[1, 0].set_title('Кластеры нечетких величин')
            axes[1, 0].set_xlabel('Кластер')
            axes[1, 0].set_ylabel('Центр (x0)')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'Сначала выполните кластеризацию',
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Кластеры')
        
        # 4. Примеры функций принадлежности
        x = np.linspace(-15, 15, 1000)
        for i, fn in enumerate(self.fuzzy_numbers[:5]):
            y = [fn.membership_function(xi) for xi in x]
            axes[1, 1].plot(x, y, label=f'N({fn.x0:.1f}, {fn.sigma:.1f})', alpha=0.7)
        axes[1, 1].set_title('Функции принадлежности (примеры)')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('μ(x)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('logs/clustering_results.png', dpi=150, bbox_inches='tight')
        print("✓ Графики сохранены в logs/clustering_results.png")
        plt.show()
    
    def show_membership_function(self):
        """Показать функцию принадлежности для конкретной величины"""
        if not self.fuzzy_numbers:
            print("Сначала сгенерируйте нечеткие величины")
            return
        
        print("\nФункция принадлежности")
        print("-" * 40)
        
        try:
            idx = int(input(f"Индекс величины (0-{len(self.fuzzy_numbers)-1}): "))
            if idx < 0 or idx >= len(self.fuzzy_numbers):
                print("Неверный индекс")
                return
            
            fn = self.fuzzy_numbers[idx]
            print(f"Выбрана величина: {fn}")
            
            # Построение графика
            x = np.linspace(fn.x0 - 4*fn.sigma, fn.x0 + 4*fn.sigma, 1000)
            y = [fn.membership_function(xi) for xi in x]
            
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, 'b-', linewidth=2, label=f'N({fn.x0:.2f}, {fn.sigma:.2f})')
            plt.axvline(x=fn.x0, color='r', linestyle='--', label=f'x0 = {fn.x0:.2f}')
            plt.fill_between(x, y, alpha=0.3)
            plt.title('Функция принадлежности нечеткой величины')
            plt.xlabel('x')
            plt.ylabel('μ(x)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('logs/membership_function.png', dpi=150, bbox_inches='tight')
            print("✓ График сохранен в logs/membership_function.png")
            plt.show()
            
        except ValueError as e:
            print(f"Ошибка: {e}")

def main():
    """Точка входа в программу"""
    print("Запуск системы работы с нечеткими величинами...")
    print("Тип нечеткой величины: нормальное распределение")
    print()
    
    app = FuzzyApp()
    app.logger.info("Программа запущена")
    app.menu()

if __name__ == '__main__':
    main()