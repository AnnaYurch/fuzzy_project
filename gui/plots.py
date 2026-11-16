"""
Графики (светлая тема). Кривые строятся через fn.membership(),
срезы — через fn.alpha_cut() / fn.alpha_cut_analytic().
Общий график распределения плотности всего множества: density_overview().
"""
from __future__ import annotations
import math
import numpy as np
from matplotlib import cm, colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from libs import filter_by_alpha_threshold
from .theme import T, PALETTE, cluster_color, beautify_ax, style_legend

SQRT_2PI = math.sqrt(2.0 * math.pi)
EDGE = "#ffffff"          # обводка точек на белом фоне


def _mu_curve(fn, xs):
    return np.array([fn.membership(float(x)) for x in xs], dtype=float)


def _colorbar(fig, ax, sc, label):
    cb = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label(label, color=T["muted"], fontsize=8.5)
    cb.ax.yaxis.set_tick_params(colors=T["muted"], labelsize=7.5)
    cb.outline.set_edgecolor(T["border"])


# ── Обзор множества: (x₀, σ), цвет = μmax ─────────────────────────
def overview(fig, data, title):
    fig.clf()
    ax = fig.add_subplot(111)
    xs = np.array([f.x0 for f in data])
    ss = np.array([f.sigma for f in data])
    mm = np.array([f.mu_max() for f in data])
    sc = ax.scatter(xs, ss, c=mm, cmap="viridis", s=30,
                    edgecolors=EDGE, linewidths=0.5, zorder=3)
    _colorbar(fig, ax, sc, "μmax")
    beautify_ax(ax, title=title, xl="x₀ — центр", yl="σ — разброс")


# ── ★ ОБЩИЙ ГРАФИК РАСПРЕДЕЛЕНИЯ ПЛОТНОСТИ μ(x) ВСЕГО МНОЖЕСТВА ──
def density_overview(fig, data, alpha=None, max_curves=80, title=None):
    """
    Тонкие линии  — плотности μ(x) отдельных величин (цвет = μmax).
    Жирная линия  — средняя плотность по всему множеству.
    Пунктир       — верхняя огибающая (максимум μ в каждой точке x).
    Горизонталь   — уровень α.
    """
    fig.clf()
    ax = fig.add_subplot(111)
    n = len(data)
    lo = min(f.x0 - 4.0 * f.sigma for f in data)
    hi = max(f.x0 + 4.0 * f.sigma for f in data)
    xs = np.linspace(lo, hi, 600)

    curves = np.array([_mu_curve(f, xs) for f in data], dtype=float)
    mm = np.array([f.mu_max() for f in data], dtype=float)
    mean_curve = curves.mean(axis=0)
    envelope = curves.max(axis=0)

    cmap = colormaps["viridis"]
    vmin, vmax = float(mm.min()), float(mm.max())
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    idx = (np.unique(np.linspace(0, n - 1, max_curves).astype(int))
           if n > max_curves else np.arange(n))
    for i in idx:
        ax.plot(xs, curves[i], color=cmap(norm(mm[i])), lw=0.8, alpha=0.40, zorder=2)

    ax.plot(xs, envelope, color=T["accent"], lw=1.3, ls=":", zorder=3,
            label="верхняя огибающая (max μ)")
    ax.plot(xs, mean_curve, color=T["text"], lw=2.4, zorder=4,
            label="средняя плотность множества")
    if alpha is not None:
        ax.axhline(alpha, color=T["red"], ls="--", lw=1.3, zorder=3, label=f"α = {alpha:g}")
        passed = int((mm >= alpha).sum())
        ax.text(0.99, 0.97, f"прошли α: {passed} из {n}", transform=ax.transAxes,
                ha="right", va="top", color=T["red"], fontsize=8.5)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("μmax величины", color=T["muted"], fontsize=8.5)
    cb.ax.yaxis.set_tick_params(colors=T["muted"], labelsize=7.5)
    cb.outline.set_edgecolor(T["border"])

    ax.set_ylim(bottom=0)
    beautify_ax(ax, title=title or f"Общее распределение плотности μ(x) · N={n}",
                xl="x", yl="μ(x)")
    style_legend(ax, loc="upper right")


# ── Scatter по кластерам ──────────────────────────────────────────
def _scatter_clusters(ax, data, labels, centers, title):
    X = np.array([[f.x0, f.sigma] for f in data], dtype=float)
    labels = np.asarray(labels)
    for c in sorted(set(labels.tolist())):
        m = labels == c
        lbl = f"C{c} · {int(m.sum())} шт." if c >= 0 else f"не прошли α · {int(m.sum())}"
        ax.scatter(X[m, 0], X[m, 1], s=30, color=cluster_color(c),
                   edgecolors=EDGE, linewidths=0.5, alpha=0.92, zorder=3, label=lbl)
    if centers is None:
        centers = [[X[labels == c, 0].mean(), X[labels == c, 1].mean()]
                   for c in sorted(set(labels.tolist())) if c >= 0]
    centers = np.asarray(centers, dtype=float)
    if centers.size:
        ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=90,
                   facecolors=T["text"], edgecolors=EDGE, linewidths=0.9, zorder=4)
    beautify_ax(ax, title=title, xl="x₀", yl="σ")
    style_legend(ax)


def clusters(fig, data, result, title):
    fig.clf()
    ax = fig.add_subplot(111)
    _scatter_clusters(ax, data, result["labels"], result.get("centers"), title)


# ── Купола μ(x) ───────────────────────────────────────────────────
def bells(fig, items, names, alpha=None, title="", cut=None):
    fig.clf()
    ax = fig.add_subplot(111)
    cols = [T["accent"], T["accent_2"], T["green"], T["amber"], T["cyan"], T["red"]]
    lo = min(f.x0 - 4.5 * f.sigma for f in items)
    hi = max(f.x0 + 4.5 * f.sigma for f in items)
    xs = np.linspace(lo, hi, 500)
    ymax = 0.0
    for fn, nm, c in zip(items, names, cols):
        ys = _mu_curve(fn, xs)
        ymax = max(ymax, float(ys.max()))
        ax.plot(xs, ys, color=c, lw=1.9, label=f"{nm}: x₀={fn.x0:.2f}, σ={fn.sigma:.2f}")
    if alpha is not None:
        ax.axhline(alpha, color=T["red"], ls="--", lw=1.3, label=f"α = {alpha:g}")
        ymax = max(ymax, alpha)
    if cut is not None and alpha is not None:
        try:
            xl, xr = cut.alpha_cut(alpha)
            seg = (xs >= xl) & (xs <= xr)
            ax.fill_between(xs[seg], alpha, _mu_curve(cut, xs[seg]),
                            color=T["green"], alpha=0.25, zorder=4)
            ax.axvline(xl, color=T["amber"], ls=":", lw=1.2)
            ax.axvline(xr, color=T["amber"], ls=":", lw=1.2)
            ax.text((xl + xr) / 2, alpha * 1.03, f"[{xl:.2f}; {xr:.2f}]",
                    ha="center", color=T["amber"], fontsize=8)
        except ValueError:
            pass
    ax.set_ylim(-0.03 * ymax, ymax * 1.18)
    beautify_ax(ax, title=title, xl="x", yl="μ(x)")
    style_legend(ax)


def bells_grid(fig, items, alpha):
    fig.clf()
    n = len(items)
    for i, fn in enumerate(items):
        ax = fig.add_subplot(n, 1, i + 1)
        xs = np.linspace(fn.x0 - 4.5 * fn.sigma, fn.x0 + 4.5 * fn.sigma, 400)
        ys = _mu_curve(fn, xs)
        ax.plot(xs, ys, color=PALETTE[i % len(PALETTE)], lw=1.8)
        ax.axhline(alpha, color=T["red"], ls="--", lw=1.1)
        try:
            xl, xr = fn.alpha_cut(alpha)
            seg = (xs >= xl) & (xs <= xr)
            ax.fill_between(xs[seg], alpha, ys[seg], color=T["green"], alpha=0.25)
            ax.axvline(xl, color=T["amber"], ls=":", lw=1.0)
            ax.axvline(xr, color=T["amber"], ls=":", lw=1.0)
        except ValueError:
            ax.text(0.98, 0.85, "α > μmax — срез пуст", transform=ax.transAxes,
                    ha="right", color=T["red"], fontsize=8)
        beautify_ax(ax, title=f"{fn.name}: x₀={fn.x0:.2f}, σ={fn.sigma:.2f}, μmax={fn.mu_max():.3f}",
                    xl="x", yl="μ(x)")


def alpha_cut_single(fig, fn, alpha):
    fig.clf()
    ax = fig.add_subplot(111)
    xs = np.linspace(fn.x0 - 4.5 * fn.sigma, fn.x0 + 4.5 * fn.sigma, 500)
    ys = _mu_curve(fn, xs)
    ax.plot(xs, ys, color=T["accent"], lw=2.0,
            label=f"{fn.name}: x₀={fn.x0:.2f}, σ={fn.sigma:.2f}")
    ax.axhline(alpha, color=T["red"], ls="--", lw=1.3, label=f"α = {alpha:g}")
    try:
        xl, xr = fn.alpha_cut(alpha)
        seg = (xs >= xl) & (xs <= xr)
        ax.fill_between(xs[seg], alpha, ys[seg], color=T["green"], alpha=0.25,
                        label=f"срез (бисекция): [{xl:.3f}; {xr:.3f}]")
        ax.axvline(xl, color=T["amber"], ls=":", lw=1.2)
        ax.axvline(xr, color=T["amber"], ls=":", lw=1.2)
        al, ar = fn.alpha_cut_analytic(alpha)
        ax.plot([al, ar], [alpha, alpha], "o", ms=5, color=T["amber"],
                label="аналитика (проверка)")
    except ValueError:
        ax.text(0.5, 0.5, f"α = {alpha:g} > μmax = {fn.mu_max():.3f}\nсрез пуст",
                transform=ax.transAxes, ha="center", va="center", color=T["red"], fontsize=11)
    ax.set_ylim(-0.03, max(fn.mu_max(), alpha) * 1.2)
    beautify_ax(ax, title=f"α-срез величины {fn.name}", xl="x", yl="μ(x)")
    style_legend(ax)


# ── Демо «схлопывания» ────────────────────────────────────────────
def subset_demo(fig, data, demo):
    fig.clf()
    alpha, full = demo["alpha"], demo["full"]
    ax1 = fig.add_subplot(121)
    _scatter_clusters(ax1, data, full["labels"], full.get("centers"),
                      f"До α-отбора · N={len(data)}, k={full['k']}")
    ax2 = fig.add_subplot(122)
    sub = demo["subset_result"]
    if sub:
        subset = filter_by_alpha_threshold(data, alpha)
        _scatter_clusters(ax2, subset, sub["labels"], sub.get("centers"),
                          f"После α-отбора · N={len(subset)}, α={alpha:g}")
        thr = 1.0 / (alpha * SQRT_2PI)
        ax2.axhline(thr, color=T["red"], ls="--", lw=1.0, alpha=0.85)
        ax2.text(0.98, 0.04, f"порог σ = {thr:.3f}", transform=ax2.transAxes,
                 ha="right", color=T["red"], fontsize=7.5)
    else:
        beautify_ax(ax2, title="Подмножество слишком мало", xl="x₀", yl="σ")
        ax2.text(0.5, 0.5, demo["message"], transform=ax2.transAxes,
                 ha="center", va="center", color=T["red"], fontsize=10, wrap=True)


def elbow_bic(fig, ks, inertias, ks2, bics, best_k):
    fig.clf()
    ax1 = fig.add_subplot(121)
    ax1.plot(ks, inertias, "o-", color=T["accent"], lw=1.8, ms=5,
             markeredgecolor=EDGE, zorder=3)
    beautify_ax(ax1, title="Метод локтя: инерция J(k)", xl="k", yl="J (WCSS)")
    ax2 = fig.add_subplot(122)
    ax2.plot(ks2, bics, "s-", color=T["amber"], lw=1.8, ms=5,
             markeredgecolor=EDGE, zorder=3)
    bb = bics[ks2.index(best_k)]
    ax2.axvline(best_k, color=T["red"], ls="--", lw=1.2)
    ax2.scatter([best_k], [bb], s=130, facecolors="none",
                edgecolors=T["green"], linewidths=2, zorder=4)
    ax2.annotate(f"k* = {best_k}", (best_k, bb), textcoords="offset points",
                 xytext=(10, 8), color=T["green"], fontsize=9, fontweight="bold")
    beautify_ax(ax2, title="BIC (GMM): минимум = оптимальное k", xl="k", yl="BIC")


def alpha_selection(fig, data, alpha):
    fig.clf()
    ax = fig.add_subplot(111)
    passed = [f for f in data if f.passes_alpha_threshold(alpha)]
    failed = [f for f in data if not f.passes_alpha_threshold(alpha)]
    if passed:
        X = np.array([[f.x0, f.sigma] for f in passed])
        ax.scatter(X[:, 0], X[:, 1], c=T["green"], s=55, alpha=0.85,
                   edgecolors=EDGE, linewidths=0.5, zorder=3,
                   label=f"прошли α={alpha:g} ({len(passed)})")
    if failed:
        X = np.array([[f.x0, f.sigma] for f in failed])
        ax.scatter(X[:, 0], X[:, 1], c=T["red"], s=55, alpha=0.6, marker="x",
                   zorder=3, label=f"отсеяны ({len(failed)})")
    thr = 1.0 / (alpha * SQRT_2PI)
    ax.axhline(thr, color=T["amber"], ls="--", lw=1.2,
               label=f"граница σ = 1/(α√2π) = {thr:.3f}")
    beautify_ax(ax, title=f"α-отбор: μmax ≥ {alpha:g}", xl="x₀", yl="σ")
    style_legend(ax)


def overlap(fig, data, result, alpha):
    fig.clf()
    ax = fig.add_subplot(111)
    labels = result["labels"]
    rows = []
    for i, fn in enumerate(data):
        if labels[i] < 0:
            continue
        try:
            xl, xr = fn.alpha_cut(alpha)
        except ValueError:
            continue
        rows.append((labels[i], xl, xr))
    rows.sort(key=lambda r: (r[0], r[1]))
    for y, (c, xl, xr) in enumerate(rows):
        ax.barh(y, xr - xl, left=xl, height=0.7, color=cluster_color(c),
                edgecolor=EDGE, linewidth=0.3, zorder=3)
    if rows:
        ax.set_ylim(-1, len(rows))
        ax.invert_yaxis()
        ax.set_yticks([])
        k = result["k"]
        if k <= 12:
            handles = [Patch(facecolor=cluster_color(c), label=f"C{c}") for c in range(k)]
            ax.legend(handles=handles, facecolor="#ffffff", edgecolor=T["border"],
                      labelcolor=T["text"], fontsize=7.5, loc="lower right")
    else:
        ax.text(0.5, 0.5, "Ни одна величина не прошла порог α", transform=ax.transAxes,
                ha="center", va="center", color=T["red"], fontsize=11)
    rejected = sum(1 for l in labels if l < 0)
    beautify_ax(ax, title=f"α-перекрытие: {result['k']} кластер(ов), α={alpha:g}, "
                          f"отсеяно {rejected}", xl="x (границы α-среза)", yl=None)


def sweep(fig, rows):
    fig.clf()
    ax = fig.add_subplot(111)
    alphas = [r["alpha"] for r in rows]
    passed = [r["passed"] for r in rows]
    ks = [r["k"] for r in rows]
    ax.plot(alphas, passed, "o-", color=T["accent"], lw=1.8, ms=5,
            markeredgecolor=EDGE, zorder=3, label="прошли α-отбор")
    beautify_ax(ax, title="Влияние α на отбор и кластеризацию", xl="α", yl="величин прошло")
    ax2 = ax.twinx()
    ax2.plot(alphas, ks, "s-", color=T["amber"], lw=1.8, ms=5,
             markeredgecolor=EDGE, zorder=3, label="кластеров (α-перекрытие)")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(T["amber"])
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(colors=T["amber"], labelsize=8)
    ax2.set_ylabel("кластеров", color=T["amber"], fontsize=8.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, facecolor="#ffffff", edgecolor=T["border"],
              labelcolor=T["text"], fontsize=7.5, loc="upper right")


def generator_check(fig, sample, n, size):
    fig.clf()
    ax = fig.add_subplot(111)
    ax.hist(np.asarray(sample), bins=70, density=True, color=T["accent"],
            alpha=0.5, edgecolor="none", zorder=3, label="выборка η")
    xx = np.linspace(-4.5, 4.5, 300)
    ax.plot(xx, np.exp(-xx ** 2 / 2.0) / SQRT_2PI, color=T["red"], lw=1.8,
            label="N(0,1) — эталон")
    beautify_ax(ax, title=f"Распределение η · n={n}, выборка {size}", xl="η", yl="плотность")
    style_legend(ax)