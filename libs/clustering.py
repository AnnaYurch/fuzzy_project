import math
from typing import List, Dict, Tuple
from .fuzzy_number import FuzzyNumber

class FuzzyClustering:
    """Кластерный анализ множества нечетких величин"""
    
    @staticmethod
    def calculate_distance_matrix(fuzzy_numbers: List[FuzzyNumber]) -> List[List[float]]:
        """
        Вычисление матрицы расстояний между нечеткими величинами.
        Используется евклидово расстояние между центрами.
        """
        n = len(fuzzy_numbers)
        distance_matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = fuzzy_numbers[i].distance_to(fuzzy_numbers[j])
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist
        
        return distance_matrix
    
    @staticmethod
    def k_means_clustering(
        fuzzy_numbers: List[FuzzyNumber],
        k: int,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Dict[int, List[FuzzyNumber]]:
        """
        Алгоритм K-средних для кластеризации нечетких величин.
        
        Параметры:
        - fuzzy_numbers: список нечетких величин
        - k: количество кластеров
        - max_iterations: максимальное количество итераций
        - tolerance: порог сходимости
        """
        import random
        
        n = len(fuzzy_numbers)
        if k > n:
            raise ValueError("k не может быть больше количества элементов")
        
        # Инициализация центроидов (случайный выбор k элементов)
        indices = random.sample(range(n), k)
        centroids = [fuzzy_numbers[i].x0 for i in indices]
        
        clusters = {i: [] for i in range(k)}
        
        for iteration in range(max_iterations):
            # Назначение элементов кластерам
            new_clusters = {i: [] for i in range(k)}
            
            for fuzzy_num in fuzzy_numbers:
                # Находим ближайший центроид
                distances = [abs(fuzzy_num.x0 - centroid) for centroid in centroids]
                cluster_id = distances.index(min(distances))
                new_clusters[cluster_id].append(fuzzy_num)
            
            # Проверка сходимости
            if new_clusters == clusters and iteration > 0:
                break
            
            clusters = new_clusters
            
            # Обновление центроидов
            new_centroids = []
            for i in range(k):
                if clusters[i]:
                    new_centroid = sum(fn.x0 for fn in clusters[i]) / len(clusters[i])
                    new_centroids.append(new_centroid)
                else:
                    new_centroids.append(centroids[i])
            
            # Проверка изменения центроидов
            max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(k))
            centroids = new_centroids
            
            if max_shift < tolerance:
                break
        
        return clusters
    
    @staticmethod
    def hierarchical_clustering(
        fuzzy_numbers: List[FuzzyNumber],
        k: int
    ) -> Dict[int, List[FuzzyNumber]]:
        """
        Иерархическая кластеризация (агломеративная).
        """
        n = len(fuzzy_numbers)
        
        # Инициализация: каждый элемент в своем кластере
        clusters = {i: [fuzzy_numbers[i]] for i in range(n)}
        cluster_centers = {i: fuzzy_numbers[i].x0 for i in range(n)}
        
        # Повторяем пока не получим k кластеров
        while len(clusters) > k:
            min_dist = float('inf')
            merge_i, merge_j = -1, -1
            
            # Находим два ближайших кластера
            cluster_ids = list(clusters.keys())
            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    ci, cj = cluster_ids[i], cluster_ids[j]
                    dist = abs(cluster_centers[ci] - cluster_centers[cj])
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = ci, cj
            
            # Объединяем кластеры
            clusters[merge_i].extend(clusters[merge_j])
            del clusters[merge_j]
            
            # Обновляем центр объединенного кластера
            cluster_centers[merge_i] = sum(fn.x0 for fn in clusters[merge_i]) / len(clusters[merge_i])
        
        # Переиндексация кластеров
        result = {}
        for i, (key, items) in enumerate(clusters.items()):
            result[i] = items
        
        return result
    
    @staticmethod
    def get_cluster_statistics(clusters: Dict[int, List[FuzzyNumber]]) -> Dict[int, Dict]:
        """
        Получение статистики по кластерам.
        """
        stats = {}
        for cluster_id, items in clusters.items():
            if items:
                centers = [fn.x0 for fn in items]
                sigmas = [fn.sigma for fn in items]
                stats[cluster_id] = {
                    'count': len(items),
                    'mean_x0': sum(centers) / len(centers),
                    'mean_sigma': sum(sigmas) / len(sigmas),
                    'min_x0': min(centers),
                    'max_x0': max(centers),
                    'std_x0': (sum((x - sum(centers)/len(centers))**2 for x in centers) / len(centers)) ** 0.5
                }
        return stats