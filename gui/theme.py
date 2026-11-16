"""Светлая (белая) тема: классический академический вид, без «модного» тёмного дашборда."""
from __future__ import annotations
import sys
import tkinter as tk
from tkinter import ttk
import matplotlib

T = {
    "bg":       "#f4f6f9",   # фон окна — очень светлый серый
    "panel":    "#ffffff",   # панели — белые
    "panel_2":  "#eef1f6",   # hover / плашки
    "panel_3":  "#ffffff",   # поля ввода
    "border":   "#d3d9e2",
    "text":     "#1f2430",
    "muted":    "#5a6472",
    "faint":    "#8a93a2",
    "accent":   "#2f6fb2",   # спокойный синий
    "accent_2": "#6b4fa1",   # приглушённый фиолетовый
    "green":    "#2e8b57",
    "amber":    "#b07d1a",
    "red":      "#c0392b",
    "cyan":     "#1a7f8c",
}

# 12 цветов кластеров (tab10 + extended) — хорошо читаются на белом
PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#393b79", "#ad494a",
]


def cluster_color(label: int) -> str:
    return PALETTE[label % len(PALETTE)] if label >= 0 else T["faint"]


if sys.platform.startswith("win"):
    BASE_FONT, MONO_FONT = "Segoe UI", "Consolas"
elif sys.platform == "darwin":
    BASE_FONT, MONO_FONT = "Helvetica", "Menlo"
else:
    BASE_FONT, MONO_FONT = "DejaVu Sans", "DejaVu Sans Mono"

F_TITLE   = (BASE_FONT, 14, "bold")
F_SUB     = (BASE_FONT, 9)
F_SECTION = (BASE_FONT, 8, "bold")
F_BTN     = (BASE_FONT, 9, "bold")
F_BTNSUB  = (BASE_FONT, 8)
F_CONS    = (MONO_FONT, 10)
F_CONS_S  = (MONO_FONT, 9)
F_SMALL   = (BASE_FONT, 8)
F_STAT    = (BASE_FONT, 9)
F_CHIP    = (MONO_FONT, 9, "bold")
F_DLG_T   = (BASE_FONT, 12, "bold")


def setup_matplotlib() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [BASE_FONT, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
    })


def setup_ttk(root: tk.Tk) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    root.option_add("*TCombobox*Listbox.background", "#ffffff")
    root.option_add("*TCombobox*Listbox.foreground", T["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", T["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    root.option_add("*TCombobox*Listbox.font", (MONO_FONT, 9))
    style.configure("Dark.TCombobox",
                    fieldbackground="#ffffff", background=T["panel_2"],
                    foreground=T["text"], arrowcolor=T["muted"],
                    bordercolor=T["border"], lightcolor="#ffffff",
                    darkcolor="#ffffff", insertcolor=T["text"], padding=4)
    style.map("Dark.TCombobox",
              fieldbackground=[("readonly", "#ffffff")],
              foreground=[("readonly", T["text"])],
              bordercolor=[("focus", T["accent"])],
              lightcolor=[("focus", T["accent"])],
              darkcolor=[("focus", T["accent"])])
    style.configure("Dark.Vertical.TScrollbar",
                    background=T["panel_2"], troughcolor=T["bg"],
                    bordercolor=T["bg"], arrowcolor=T["muted"])
    style.map("Dark.Vertical.TScrollbar", background=[("active", T["border"])])


def beautify_ax(ax, title=None, xl=None, yl=None, grid=True) -> None:
    ax.set_facecolor("#ffffff")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b6bfca")
    ax.tick_params(colors=T["muted"], labelsize=8, length=3)
    if grid:
        ax.grid(True, color="#e6eaf0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=T["text"], fontsize=10, fontweight="bold", pad=8, loc="left")
    if xl:
        ax.set_xlabel(xl, color=T["muted"], fontsize=8.5)
    if yl:
        ax.set_ylabel(yl, color=T["muted"], fontsize=8.5)


def style_legend(ax, **kw) -> None:
    ax.legend(facecolor="#ffffff", edgecolor=T["border"], labelcolor=T["text"],
              fontsize=7.5, framealpha=0.97, borderpad=0.7,
              handlelength=1.4, labelspacing=0.45, **kw)