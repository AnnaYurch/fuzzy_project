"""
Точка входа. GUI (tkinter + matplotlib).
Вся математика — в libs/, весь интерфейс — в gui/. Здесь только запуск.

    python main.py          — обычное окно
    python main.py --demo   — окно + автоматический сценарий преподавателя
"""
from __future__ import annotations
import sys


def _hidpi_windows() -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def main() -> None:
    _hidpi_windows()
    import tkinter as tk
    from gui.theme import setup_ttk, setup_matplotlib
    from gui.app import App

    setup_matplotlib()
    root = tk.Tk()
    setup_ttk(root)
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    if "--demo" in sys.argv:
        root.after(700, app.action_demo)
    root.mainloop()


if __name__ == "__main__":
    main()