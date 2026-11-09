"""
Модуль логирования. Пишет одновременно в консоль (INFO) и в файл logs/app.log (DEBUG).
"""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"

_LOGGER_NAME = "fuzzy_project"
_configured = False

def _configure() -> logging.Logger:
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    if _configured:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Файл: подробный лог
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)

    # Консоль: только INFO и выше, кратко
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(fmt="  [LOG] %(message)s"))
    logger.addHandler(console_handler)

    _configured = True
    logger.debug("=" * 70)
    logger.debug("Сессия логирования начата")
    return logger

def get_logger(name: str | None = None) -> logging.Logger:
    base = _configure()
    return base.getChild(name) if name else base