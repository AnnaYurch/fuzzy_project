import logging
import os
from datetime import datetime
from typing import Optional

class FuzzyLogger:
    """Независимый модуль логирования"""
    
    def __init__(self, log_dir: str = 'logs', log_file: str = 'app.log'):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, log_file)
        
        # Создаем директорию для логов если её нет
        os.makedirs(log_dir, exist_ok=True)
        
        # Настраиваем логгер
        self.logger = logging.getLogger('fuzzy_project')
        self.logger.setLevel(logging.DEBUG)
        
        # Очищаем предыдущие обработчики
        self.logger.handlers.clear()
        
        # Файловый обработчик
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Форматирование
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def log_fuzzy_number(self, fuzzy_num, operation: str = "created"):
        self.info(f"{operation}: {fuzzy_num}")
    
    def log_cluster_result(self, cluster_id: int, count: int, stats: dict):
        self.info(f"Кластер {cluster_id}: {count} элементов, "
                 f"средний x0={stats['mean_x0']:.4f}, "
                 f"средний sigma={stats['mean_sigma']:.4f}")