"""Главное окно (светлая тема). Оптимизировано: 12 действий, только равномерная генерация."""
from __future__ import annotations
import math
import time
import random
import logging
import functools
import traceback
from contextlib import contextmanager
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from libs import (
    FuzzyNumber, add, subtract, multiply, divide, multiply_by_scalar, compare_str, calculate_distance,
    generate_uniform_fuzzy_set, save_to_tsv, load_from_tsv,
    perform_clustering, elbow_method, gmm_bic,
    cluster_by_alpha_overlap, filter_by_alpha_threshold, clustering_on_subset, get_logger,
)
from .theme import T, F_TITLE, F_SUB, F_SECTION, F_CONS, F_SMALL, F_STAT, BASE_FONT, MONO_FONT
from .widgets import Chip, MiniBtn, SideButton, ParamDialog
from . import plots

log = get_logger("gui")
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATASET = DATA_DIR / "dataset.tsv"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SQRT_2PI = math.sqrt(2.0 * math.pi)

def safe(name):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(self, *a, **k):
            try:
                return fn(self, *a, **k)
            except Exception as exc:
                self.log(f"✗ Ошибка ({name}): {exc}", "err")
                for ln in traceback.format_exc(limit=3).strip().splitlines()[-3:]:
                    self.log("   " + ln, "dim")
                self._set_status(f"Ошибка: {exc}", "err")
                log.exception(name)
        return wrap
    return deco

class _GuiLogHandler(logging.Handler):
    def __init__(self, app):
        super().__init__(level=logging.INFO)
        self.app = app

    def emit(self, record):
        try:
            tag = {"WARNING": "warn", "ERROR": "err", "CRITICAL": "err"}.get(record.levelname, "lib")
            self.app.log(record.getMessage(), tag, stamp=True)
        except Exception:
            pass

BUTTON_SPECS = [
    ("sec", "Данные", T["accent"]),
    (1,   "", "Генерация множества",       "",          "action_generate"),
    (2,   "", "Загрузить из TSV",          "",             "action_load"),
    (3,   "", "Сохранить в TSV",           "",              "action_save"),
    (4,   "", "Список величин",            "",          "action_list"),
    (5,   "", "Добавить вручную",          "",                           "action_add"),
    ("sec", "Анализ и Визуализация", T["green"]),
    (6,   "", "Визуализация α-порога",     "", "action_alpha_all"),
    (7,   "", "Арифметика A и B",          "",         "action_arithmetic"),
    ("sec", "Кластерный анализ", T["accent_2"]),
    (8,   "", "K-Means (заданное k)",      "",              "action_kmeans"),
    (9,   "", "Подбор числа кластеров",    "",                 "action_choose_k"),
    (10,  "", "★ α-отбор + кластеризация", "",       "action_alpha_subset"),
    (11,  "", "Перекрытие α-срезов",       "",     "action_alpha_overlap"),
    ("sec", "Графики", T["amber"]),
    (12,  "", "Купола выбранных величин",  "",       "action_bells_selected"),
]

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.data = []
        self.alpha = 0.43
        self.k = 5
        self.result = None
        self._resize_job = None
        self._build_ui()
        get_logger().addHandler(_GuiLogHandler(self))
        self._startup_log()

    def _build_ui(self):
        r = self.root
        r.title("Комплекс работы с нечёткими величинами")
        r.configure(bg=T["bg"])
        r.geometry("1360x860")
        r.minsize(1180, 740)
        self._build_header()
        self._build_statusbar()
        body = tk.Frame(r, bg=T["bg"])
        body.pack(side="top", fill="both", expand=True)
        self._build_sidebar(body)
        tk.Frame(body, bg=T["border"], width=1).pack(side="left", fill="y")
        self._build_right(body)

    def _build_header(self):
        hd = tk.Frame(self.root, bg=T["panel"])
        hd.pack(side="top", fill="x")
        row = tk.Frame(hd, bg=T["panel"])
        row.pack(fill="x", padx=16, pady=(12, 10))
        # БЕЗ бейджа μ, БЕЗ подзаголовка, БЕЗ формулы и чипа libs — только заголовок
        ttl = tk.Frame(row, bg=T["panel"])
        ttl.pack(side="left")
        tk.Label(ttl, text="Комплекс работы с нечёткими величинами", font=F_TITLE, fg=T["text"], bg=T["panel"]).pack(anchor="w")
        tk.Frame(hd, bg=T["border"], height=1).pack(fill="x")

    def _build_statusbar(self):
        sb = tk.Frame(self.root, bg=T["panel"])
        sb.pack(side="bottom", fill="x")
        tk.Frame(sb, bg=T["border"], height=1).pack(side="top", fill="x")
        row = tk.Frame(sb, bg=T["panel"])
        row.pack(fill="x", padx=12, pady=5)
        # БЕЗ точки и БЕЗ текста «Готов к работе»: строка пустая до первого действия
        self.status_lbl = tk.Label(row, text="", fg=T["muted"], bg=T["panel"], font=F_STAT)
        self.status_lbl.pack(side="left")
        self.chip_k = Chip(row, f"k = {self.k}", fg=T["accent_2"])
        self.chip_alpha = Chip(row, f"α = {self.alpha}", fg=T["amber"])
        self.chip_n = Chip(row, f"N = {len(self.data)}", fg=T["accent"])
        for c in (self.chip_k, self.chip_alpha, self.chip_n):
            c.pack(side="right", padx=(8, 0))

    def _build_sidebar(self, parent):
        side = tk.Frame(parent, bg=T["panel"], width=300)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        cap = tk.Frame(side, bg=T["panel"])
        cap.pack(fill="x", padx=14, pady=(14, 2))
        tk.Label(cap, text="ПАНЕЛЬ ОПЕРАЦИЙ", font=F_SECTION, fg=T["muted"], bg=T["panel"]).pack(side="left")
        tk.Label(cap, text="12 действий", font=F_SMALL, fg=T["faint"], bg=T["panel"]).pack(side="right")

        cv = tk.Canvas(side, bg=T["panel"], highlightthickness=0, bd=0)
        cv.pack(fill="both", expand=True)
        inner = tk.Frame(cv, bg=T["panel"])
        win = cv.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: cv.itemconfigure(win, width=e.width))
        cv.bind("<Configure>", lambda e: cv.itemconfigure(win, width=e.width))

        def wheel(ev):
            dy = -1 if (getattr(ev, "delta", 0) > 0 or getattr(ev, "num", 0) == 4) else 1
            cv.yview_scroll(dy, "units")

        side.bind("<Enter>", lambda e: (cv.bind_all("<MouseWheel>", wheel), cv.bind_all("<Button-4>", wheel), cv.bind_all("<Button-5>", wheel)))
        side.bind("<Leave>", lambda e: (cv.unbind_all("<MouseWheel>"), cv.unbind_all("<Button-4>"), cv.unbind_all("<Button-5>")))

        #SideButton(inner, None, "▶", "Автодемо: сценарий защиты", "150 вел. → k=5 → α=0.43 → «схлопывание»", T["green"], self.action_demo).pack(fill="x", padx=10, pady=(6, 2))

        self._actions = {}
        for spec in BUTTON_SPECS:
            if spec[0] == "sec":
                _, text, color = spec
                wrap = tk.Frame(inner, bg=T["panel"])
                wrap.pack(fill="x", padx=14, pady=(16, 6))
                line = tk.Frame(wrap, bg=T["panel"])
                line.pack(fill="x")
                tk.Label(line, text=text.upper(), font=F_SECTION, fg=color, bg=T["panel"]).pack(side="left")
                tk.Frame(line, bg=T["border"], height=1).pack(side="left", fill="x", expand=True, padx=(8, 0))
            else:
                num, icon, title, sub, method = spec
                fn = getattr(self, method)
                self._actions[num] = fn
                accent = T["accent"] if num <= 5 else T["green"] if num <= 7 else T["accent_2"] if num <= 11 else T["amber"]
                SideButton(inner, num, icon, title, sub, accent, fn).pack(fill="x", padx=10, pady=2)

        tk.Label(inner, text="libs/ · tkinter · matplotlib", font=F_SMALL, fg=T["faint"], bg=T["panel"]).pack(pady=(18, 12))

        for num in range(1, 10):
            self.root.bind_all(f"<Alt-{num}>", lambda e, m=self._actions[num]: m())
        self.root.bind_all("<Alt-0>", lambda e: self._actions[10]())
        self.root.bind_all("<Alt-d>", lambda e: self.action_demo())

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=T["bg"])
        right.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(right, bg=T["bg"])
        inner.pack(fill="both", expand=True, padx=12, pady=10)
        self.pane = tk.PanedWindow(inner, orient="vertical", bg=T["bg"], bd=0, sashwidth=10, sashrelief="flat", showhandle=False)
        self.pane.pack(fill="both", expand=True)

        cp = tk.Frame(self.pane, bg=T["panel"], highlightbackground=T["border"], highlightthickness=1)
        chd = tk.Frame(cp, bg=T["panel"])
        chd.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(chd, text="КОНСОЛЬ ПРОГРАММЫ", font=F_SECTION, fg=T["muted"], bg=T["panel"]).pack(side="left")
        MiniBtn(chd, "очистить", self._clear_console).pack(side="right")
        MiniBtn(chd, "копировать", self._copy_console).pack(side="right", padx=(0, 10))
        cwrap = tk.Frame(cp, bg=T["panel"])
        cwrap.pack(fill="both", expand=True, padx=10, pady=(2, 10))
        csb = ttk.Scrollbar(cwrap, style="Dark.Vertical.TScrollbar")
        csb.pack(side="right", fill="y")
        self.console = tk.Text(cwrap, yscrollcommand=csb.set, bg="#ffffff", fg=T["text"], font=F_CONS, insertbackground=T["text"], relief="flat", padx=12, pady=10, wrap="word", state="disabled", highlightthickness=0, bd=0)
        self.console.pack(side="left", fill="both", expand=True)
        csb.config(command=self.console.yview)
        for tag, kw in {"h": dict(foreground=T["accent"], font=(MONO_FONT, 10, "bold")), "h2": dict(foreground=T["accent_2"], font=(MONO_FONT, 10, "bold")), "ok": dict(foreground=T["green"]), "warn": dict(foreground=T["amber"]), "err": dict(foreground=T["red"], font=(MONO_FONT, 10, "bold")), "dim": dict(foreground=T["faint"]), "math": dict(foreground=T["cyan"]), "lib": dict(foreground=T["muted"]), "stamp": dict(foreground="#9aa3b2")}.items():
            self.console.tag_configure(tag, **kw)

        pp = tk.Frame(self.pane, bg=T["panel"], highlightbackground=T["border"], highlightthickness=1)
        phd = tk.Frame(pp, bg=T["panel"])
        phd.pack(fill="x", padx=10, pady=(8, 2))
        tk.Label(phd, text="ВИЗУАЛИЗАЦИЯ", font=F_SECTION, fg=T["muted"], bg=T["panel"]).pack(side="left")
        self.plot_hint = tk.Label(phd, text="", font=F_SMALL, fg=T["faint"], bg=T["panel"])
        self.plot_hint.pack(side="right")
        self.plot_body = tk.Frame(pp, bg=T["panel"])
        self.plot_body.pack(fill="both", expand=True, padx=10, pady=(2, 8))
        self.fig = Figure(figsize=(8, 4.2), dpi=100, facecolor="#ffffff", constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_body)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.config(bg="#ffffff", highlightthickness=0, bd=0)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_body)
        try:
            if not self.toolbar.winfo_manager():
                self.toolbar.pack(side="bottom", fill="x")
        except Exception:
            pass
        self._style_toolbar()
        self.canvas_widget.pack(side="top", fill="both", expand=True)
        self.pane.add(cp, minsize=150, stretch="always")
        self.pane.add(pp, minsize=220, stretch="always")
        self.plot_body.bind("<Configure>", self._on_plot_resize)
        self.root.after(120, self._place_sash)
        self.root.after(260, self._initial_plot)

    def _style_toolbar(self):
        try:
            self.toolbar.configure(bg=T["panel"], highlightthickness=0, bd=0)
            for w in self.toolbar.winfo_children():
                try:
                    w.configure(bg=T["panel"], fg=T["muted"], relief="flat", bd=0, activebackground=T["panel_2"], activeforeground=T["text"], highlightthickness=0, disabledforeground=T["faint"])
                except Exception:
                    pass
        except Exception:
            pass

    def _place_sash(self):
        try:
            h = self.pane.winfo_height()
            if h > 250:
                self.pane.sash_place(0, 1, int(h * 0.44))
        except Exception:
            pass

    def log(self, text, tag=None, stamp=False):
        self.console.configure(state="normal")
        if stamp:
            self.console.insert("end", time.strftime("[%H:%M:%S] "), "stamp")
        self.console.insert("end", text + "\n", tag)
        self.console.see("end")
        self.console.configure(state="disabled")

    def _log_header(self, title):
        self.log("")
        self.log("─" * 68, "dim")
        self.log(" " + title, "h", stamp=True)
        self.log("─" * 68, "dim")

    def _clear_console(self):
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")
        self._set_status("Консоль очищена", "info")

    def _copy_console(self):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.console.get("1.0", "end").strip())
            self._set_status("Консоль скопирована в буфер обмена", "ok")
        except Exception:
            pass

    def _set_status(self, msg, kind="info"):
        self.status_lbl.config(text=msg)

    def _refresh_chips(self):
        self.chip_n["text"] = f"N = {len(self.data)}"
        self.chip_alpha["text"] = f"α = {self.alpha:.2f}"
        self.chip_k["text"] = f"k = {self.k}"

    @contextmanager
    def _busy(self, msg):
        self._set_status(msg + "…", "work")
        self.root.config(cursor="wait")
        self.root.update_idletasks()
        try:
            yield
        finally:
            self.root.config(cursor="")

    def _require(self, minimum=1):
        if len(self.data) < minimum:
            self.log(f"Нужно минимум {minimum} величин. Сейчас: {len(self.data)}. Сначала пункт 1 или 2.", "warn")
            self._set_status("Нет данных", "warn")
            return False
        return True

    def _redraw(self):
        self.canvas.draw_idle()

    def _on_plot_resize(self, ev):
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(140, self._do_resize, ev.width, ev.height)

    def _do_resize(self, w, h):
        if w < 90 or h < 70:
            return
        self.fig.set_size_inches(w / self.fig.dpi, h / self.fig.dpi)
        self.fig.canvas.draw_idle()

    def _initial_plot(self):
        w, h = self.plot_body.winfo_width(), self.plot_body.winfo_height()
        if w > 90 and h > 70:
            self.fig.set_size_inches(w / self.fig.dpi, h / self.fig.dpi)
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#ffffff")
        ax.axis("off")
        # БЕЗ водяного знака μ — только подсказка по центру
        ax.text(0.5, 0.5, "Графики появятся здесь — выполните действие на панели слева", ha="center", va="center", fontsize=10, color=T["faint"])
        self._redraw()

    def _print_summary(self, rows, with_range=False):
        if not rows:
            self.log("  (кластеров нет)", "dim")
            return
        has_range = with_range and "min_x0" in rows[0]
        head = f"  {'Кластер':<9}{'Размер':>7}{'средн. x₀':>13}{'средн. σ':>13}"
        if has_range:
            head += f"   {'диапазон x₀':<13}"
        self.log(head, "h2")
        self.log("  " + "─" * 66, "dim")
        for r in rows:
            line = f"  {r['cluster']:<9}{r['size']:>7}{r['mean_x0']:>13.4f}{r['mean_sigma']:>13.4f}"
            if has_range:
                line += f"   [{r['min_x0']:.2f}; {r['max_x0']:.2f}]"
            self.log(line)

    def _startup_log(self):
        self.log("╔" + "═" * 62 + "╗", "dim")
        self.log("║  КОМПЛЕКС РАБОТЫ С НЕЧЁТКИМИ ВЕЛИЧИНАМИ".ljust(63) + "║", "h")
        self.log("╚" + "═" * 62 + "╝", "dim")

    # ── ДЕЙСТВИЯ ──
    @safe("Генерация")
    def action_generate(self):
        # УБРАН ВЫБОР РЕЖИМА. ТЕПЕРЬ ТОЛЬКО РАВНОМЕРНАЯ ГЕНЕРАЦИЯ.
        d = ParamDialog(self.root, "1 · Генерация множества", [
            ("count", "Количество (100–300)", 150, "int", (100, 300)),
            ("seed", "Зерно ГСЧ (seed)", 42, "int", (0, 10_000_000)),
        ], hint="Равномерный разброс параметров x₀ и σ (без группировки).").show()
        if not d: return
        with self._busy("Генерация множества"):
            self.data = generate_uniform_fuzzy_set(d["count"], seed=d["seed"])
        self.result = None
        mm = [f.mu_max() for f in self.data]
        self._log_header("1 · ГЕНЕРАЦИЯ МНОЖЕСТВА")
        self.log(f"✓ Сгенерировано {len(self.data)} нечётких величин (равномерно).", "ok")
        self.log(f"seed={d['seed']}")
        self.log(f"μ_max ∈ [{min(mm):.4f}; {max(mm):.4f}] — разброс позволяет демонстрировать α-отбор.", "math")
        plots.overview(self.fig, self.data, f"Сгенерированное множество · N={len(self.data)} (цвет = μmax)")
        self.plot_hint.config(text="пространство (x₀, σ)")
        self._redraw(); self._refresh_chips()
        self._set_status(f"Сгенерировано {len(self.data)} величин", "ok")

    @safe("Загрузка TSV")
    def action_load(self):
        path = filedialog.askopenfilename(title="Загрузить множество из TSV", initialdir=str(DATA_DIR), filetypes=[("TSV-файлы", "*.tsv *.txt *.csv"), ("Все файлы", "*.*")])
        if not path: return
        with self._busy("Чтение файла"):
            self.data = load_from_tsv(path)
        self.result = None
        self._log_header("2 · ЗАГРУЗКА ИЗ TSV")
        self.log(f"✓ Загружено {len(self.data)} величин из {path}", "ok")
        plots.overview(self.fig, self.data, f"Загруженное множество · N={len(self.data)}")
        self._redraw(); self._refresh_chips()
        self._set_status(f"Загружено {len(self.data)} величин", "ok")

    @safe("Сохранение TSV")
    def action_save(self):
        if not self._require(): return
        path = filedialog.asksaveasfilename(title="Сохранить множество в TSV", defaultextension=".tsv", initialfile="dataset.tsv", initialdir=str(DATA_DIR), filetypes=[("TSV-файлы", "*.tsv")])
        if not path: return
        save_to_tsv(self.data, path)
        self.log(f"✓ Сохранено: {path}", "ok")
        self._set_status("Множество сохранено", "ok")

    @safe("Список величин")
    def action_list(self):
        if not self._require(): return
        n = len(self.data)
        d = ParamDialog(self.root, "4 · Список величин", [("limit", "Сколько строк показать (0 — все)", 20, "int", (0, n))]).show()
        if not d: return
        shown = self.data if d["limit"] == 0 else self.data[:d["limit"]]
        labels = self.result["labels"] if self.result else None
        self._log_header(f"4 · МНОЖЕСТВО НЕЧЁТКИХ ВЕЛИЧИН (всего {n})")
        head = f"  {'№':<5}{'Имя':<8}{'x₀':>10}{'σ':>10}{'μ_max':>10}"
        if labels: head += f"{'Кластер':>10}"
        self.log(head, "h2")
        self.log("  " + "─" * 60, "dim")
        
        for i, fn in enumerate(shown):
            row = f"  {i + 1:<5}{fn.name:<8}{fn.x0:>10.4f}{fn.sigma:>10.4f}{fn.mu_max():>10.4f}"
            if labels: row += f"{labels[i]:>10}"
            self.log(row)
        self._set_status(f"Показано {len(shown)} из {n}", "info")

    @safe("Добавление величины")
    def action_add(self):
        d = ParamDialog(self.root, "5 · Добавить нечёткую величину", [
            ("x0", "Центр x₀", 50.0, "float", (-1e6, 1e6)),
            ("sigma", "Разброс σ (> 0)", 1.0, "float", (0.001, 100.0)),
        ], hint="μmax = 1/(σ√2π). Чем меньше σ, тем выше купол и тем строже α величина пройдёт.").show()
        if not d: return
        fn = FuzzyNumber(d["x0"], d["sigma"], name=f"A{len(self.data) + 1}")
        self.data.append(fn)
        self.result = None
        self._log_header("5 · ДОБАВЛЕНИЕ ВЕЛИЧИНЫ")
        self.log(f"✓ Добавлено: {fn}", "ok")
        plots.overview(self.fig, self.data, f"Множество · N={len(self.data)}")
        self._redraw(); self._refresh_chips()
        self._set_status("Величина добавлена", "ok")

    @safe("Визуализация α-порога")
    def action_alpha_all(self):
        if not self._require(): return
        d = ParamDialog(self.root, "6 · Визуализация α-порога для ВСЕХ величин", [
            ("alpha", "Уровень α-порога", self.alpha, "float", (0.001, 10.0)),
        ], hint="Синие: проходят порог (μ_max ≥ α). Красные: не проходят. Чёрная линия: уровень α.").show()
        if not d: return
        self.alpha = d["alpha"]
        self._log_header("6 · ВИЗУАЛИЗАЦИЯ α-ПОРОГА (ВСЕ ВЕЛИЧИНЫ)")
        passed = sum(1 for fn in self.data if fn.passes_alpha_threshold(self.alpha))
        self.log(f"α = {self.alpha:g}. Прошли порог: {passed} из {len(self.data)}.", "ok")
        plots.alpha_threshold_all(self.fig, self.data, self.alpha)
        self.plot_hint.config(text="")
        self._redraw(); self._refresh_chips()
        self._set_status("Визуализация α-порога построена", "ok")

    @safe("Арифметика")
    def action_arithmetic(self):
        if not self._require(2): return
        n = len(self.data)
        d = ParamDialog(self.root, "7 · Манипулирование: A и B", [
            ("ia", f"Номер A (1…{n})", 1, "int", (1, n)),
            ("ib", f"Номер B (1…{n})", 2, "int", (1, n)),
            ("k", "Скаляр k для умножения", 2.0, "float", (-1000.0, 1000.0)),
        ]).show()
        if not d: return
        a, b, k = self.data[d["ia"] - 1], self.data[d["ib"] - 1], d["k"]
        self._log_header("7 · МАНИПУЛИРОВАНИЕ НЕЧЁТКИМИ ВЕЛИЧИНАМИ")
        self.log(f"A = {a}")
        self.log(f"B = {b}")

        r_add, r_sub, r_mul = add(a, b), subtract(a, b), multiply(a, b)
        self.log(f"A + B = {r_add}")
        self.log(f"A − B = {r_sub}")
        self.log(f"A × B = {r_mul}")

        r_div = None
        try:
            r_div = divide(a, b)
            self.log(f"A / B = {r_div}")
        except ZeroDivisionError as exc:
            self.log(f"A / B → ошибка: {exc}", "warn")

        r_scal = None
        try:
            r_scal = multiply_by_scalar(a, k)
            self.log(f"A × {k:g} = {r_scal}")
        except ValueError as exc:
            self.log(f"A × {k:g} → ошибка: {exc}", "warn")

        self.log(f"Расстояние d(A,B) = {calculate_distance(a, b):.6f}", "math")

        # ТЕПЕРЬ НА ГРАФИКЕ ВСЕ ОПЕРАЦИИ: ± слева, ×÷·k справа
        plots.bells_ops(self.fig, a, b, r_add, r_sub, r_mul, r_div, r_scal, k)
        self.plot_hint.config(text="")
        self._redraw()
        self._set_status("Операции выполнены", "ok")

    @safe("K-Means")
    def action_kmeans(self):
        if not self._require(2): return
        n = len(self.data)
        d = ParamDialog(self.root, "8 · K-Means на заданное k", [("k", f"Число кластеров k (1…{n})", self.k, "int", (1, n))]).show()
        if not d: return
        k = d["k"]
        with self._busy("Кластеризация K-Means"):
            res = perform_clustering(self.data, k)
        self.k, self.result = k, res
        self._log_header("8 · K-MEANS (КРИТЕРИЙ WCSS)")
        self.log(f"Инерция J = {res['inertia']:.4f}", "math")
        self._print_summary(res["summary"], with_range=True)
        plots.clusters(self.fig, self.data, res, f"K-Means · N={n}, k={k}")
        self.plot_hint.config(text="центры кластеров — кресты")
        self._redraw(); self._refresh_chips()
        self._set_status(f"K-Means: k={k}", "ok")

    @safe("Подбор k")
    def action_choose_k(self):
        if not self._require(2): return
        n = len(self.data)
        d = ParamDialog(self.root, "9 · Подбор числа кластеров", [("max_k", "Максимальное проверяемое k", min(10, n), "int", (2, min(15, n)))], hint="Локоть по инерции J(k); BIC = −2·ln(L) + p·ln(n), минимум BIC = оптимальное k.").show()
        if not d: return
        with self._busy("Локоть + GMM/BIC"):
            ks, inertias = elbow_method(self.data, d["max_k"])
            ks2, bics, best_k = gmm_bic(self.data, d["max_k"])
        self.k = best_k
        self._log_header("9 · ПОДБОР k: ЛОКОТЬ + GMM/BIC")
        self.log(f"  {'k':<6}{'Инерция J (локоть)':>24}{'BIC (GMM)':>20}", "h2")
        self.log("  " + "─" * 60, "dim")
        for i, kk in enumerate(ks):
            bic = bics[i] if i < len(bics) else float("nan")
            mark = "  ← минимум BIC" if kk == best_k else ""
            self.log(f"  {kk:<6}{inertias[i]:>24.4f}{bic:>20.4f}{mark}", "ok" if kk == best_k else None)
        self.log(f"✓ Оптимальное число кластеров по BIC: k = {best_k}", "ok")
        plots.elbow_bic(self.fig, ks, inertias, ks2, bics, best_k)
        self.plot_hint.config(text=f"оптимум BIC: k={best_k}")
        self._redraw(); self._refresh_chips()
        self._set_status(f"BIC: оптимальное k={best_k}", "ok")

    @safe("★ α-отбор + кластеризация")
    def action_alpha_subset(self):
        if not self._require(2): return
        n = len(self.data)
        d = ParamDialog(self.root, "10 · ★ α-отбор подмножества + кластеризация", [
            ("k", f"Число кластеров k (1…{n})", self.k, "int", (1, n)),
            ("alpha", "α-порог отбора", self.alpha, "float", (0.001, 10.0)),
        ], hint="Критерий: μ(x₀) = 1/(σ√2π) ≥ α  ⇔  σ ≤ 1/(α√2π). Кластеризация полного множества и подмножества на те же k.").show()
        if not d: return
        k, alpha = d["k"], d["alpha"]
        self.k, self.alpha = k, alpha
        with self._busy("α-отбор и кластеризация"):
            demo = clustering_on_subset(self.data, alpha, k)
        self.result = demo["full"]
        thr = 1.0 / (alpha * SQRT_2PI)
        self._log_header("10 · ★ α-ОТБОР ПОДМНОЖЕСТВА + КЛАСТЕРИЗАЦИЯ")
        self.log(f"ПОЛНОЕ МНОЖЕСТВО: {n} величин, k = {k}", "h2")
        self._print_summary(demo["full"]["summary"])
        self.log(f"α-ОТБОР: порог α = {alpha:g};  критерий μ(x₀) ≥ α  ⇔  σ ≤ {thr:.4f}", "math")
        self.log(f"Было: {n} → Стало: {demo['subset_size']}", "ok")
        if demo["subset_result"] is None:
            self.log(f"! {demo['message']}", "warn")
        else:
            self.log(f"ПОДМНОЖЕСТВО: {demo['subset_size']} величин, те же k = {k}", "h2")
            self._print_summary(demo["subset_result"]["summary"])
            self.log(f"ВЫВОД: при α = {alpha:g} из {n} величин осталось {demo['subset_size']} — те, у кого μ_max ≥ {alpha:g}.", "ok")
            self.log("Кластеры перестроились: часть величин «растворилась».", "dim")
            self.log("Программа работает с ЛЮБЫМИ предельно допустимыми множествами.", "dim")
        plots.subset_demo(self.fig, self.data, demo)
        self.plot_hint.config(text=f"«схлопывание»: {n} → {demo['subset_size']} при α={alpha:g}")
        self._redraw(); self._refresh_chips()
        self._set_status(f"α-отбор: {n} → {demo['subset_size']}", "ok")

    @safe("α-перекрытие")
    def action_alpha_overlap(self):
        if not self._require(2): return
        d = ParamDialog(self.root, "11 · Перекрытие α-срезов (single-linkage)", [("alpha", "Уровень α", self.alpha, "float", (0.001, 10.0))], hint="Величины в одном кластере, если их α-срезы пересекаются. Это НЕ классическая дендрограмма, а алгоритм связности интервалов.").show()
        if not d: return
        alpha = d["alpha"]
        with self._busy("α-перекрытие"):
            res = cluster_by_alpha_overlap(self.data, alpha)
        self.alpha, self.result = alpha, res
        self._log_header("11 · ПЕРЕКРЫТИЕ α-СРЕЗОВ (НЕ ДЕНДРОГРАММА)")
        self.log(f"✓ {res['method']}")
        self.log(f"Получилось кластеров: {res['k']}", "ok")
        self._print_summary(res["summary"][:25])
        plots.overlap(self.fig, self.data, res, alpha)
        self.plot_hint.config(text="полосы — α-срезы, цвет — кластер")
        self._redraw(); self._refresh_chips()
        self._set_status(f"α-перекрытие: {res['k']} кластер(ов)", "ok")

    @safe("Купола выбранных величин")
    def action_bells_selected(self):
        if not self._require(1): return
        n = len(self.data)
        d = ParamDialog(self.root, "12 · Купола выбранных величин", [
            ("indices", f"Номера через запятую (макс. 4, напр: 1, 5, 12)", "1, 2, 3", "str"),
            ("alpha", "Показать линию α (0 для скрытия)", 0.0, "float", (0.0, 10.0)),
        ], hint="Программа построит графики μ(x) ТОЛЬКО для указанных вами величин, чтобы вы могли сравнить их форму и ширину.").show()
        if not d: return

        try:
            idx_list = [int(x.strip()) - 1 for x in d["indices"].split(",")]
            idx_list = [i for i in idx_list if 0 <= i < n]
            if not idx_list:
                self.log("Ошибка: некорректные номера.", "err")
                return
            if len(idx_list) > 4:
                idx_list = idx_list[:4]
        except ValueError:
            self.log("Ошибка: введите числа через запятую.", "err")
            return

        items = [self.data[i] for i in idx_list]
        names = [f"A{i+1}" for i in idx_list]

        self._log_header("12 · КУПОЛА ВЫБРАННЫХ ВЕЛИЧИН")
        self.log(f"Отображение величин: {', '.join(names)}", "ok")

        plots.bells_selected(self.fig, items, names, d["alpha"] if d["alpha"] > 0 else None)
        self.plot_hint.config(text="сравнение форм μ(x) для выбранных величин")
        self._redraw()
        self._set_status("Графики куполов построены", "ok")

    @safe("Автодемо")
    def action_demo(self):
        with self._busy("Автодемо"):
            self.data = generate_uniform_fuzzy_set(150, seed=42)
            demo = clustering_on_subset(self.data, 0.43, 5)
            save_to_tsv(self.data, DEFAULT_DATASET)
        self.alpha, self.k, self.result = 0.43, 5, demo["full"]
        mm = [f.mu_max() for f in self.data]
        self._log_header("▶ АВТОДЕМО · СЦЕНАРИЙ ПРЕПОДАВАТЕЛЯ")
        self.log(f"[1] Генерация: {len(self.data)} величин (равномерно); μ_max ∈ [{min(mm):.4f}; {max(mm):.4f}]", "ok")
        self.log("[2] Кластеризация полного множества, k=5:", "h2")
        self._print_summary(demo["full"]["summary"])
        self.log(f"[3] α-отбор при α=0.43: было {len(self.data)} → стало {demo['subset_size']}", "ok")
        if demo["subset_result"]:
            self.log("[4] Кластеризация подмножества, те же k=5:", "h2")
            self._print_summary(demo["subset_result"]["summary"])
            self.log("ВЫВОД: программа работает с ЛЮБЫМИ предельно допустимыми множествами.", "ok")
        else:
            self.log(f"! {demo['message']}", "warn")
        self.log(f"Набор сохранён в {DEFAULT_DATASET}", "dim")
        plots.subset_demo(self.fig, self.data, demo)
        self.plot_hint.config(text=f"демо: {len(self.data)} → {demo['subset_size']} при α=0.43")
        self._redraw(); self._refresh_chips()
        self._set_status("Автодемо завершено", "ok")

    def on_close(self):
        log.info("Завершение работы программы")
        self.root.destroy()