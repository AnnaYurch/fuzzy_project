"""Виджеты: чипы, мини-кнопки, кнопки сайдбара, диалог ввода (светлая тема)."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk

from .theme import (T, F_CHIP, F_SMALL, F_BTN, F_BTNSUB, F_CONS_S,
                    F_DLG_T, BASE_FONT)


class Chip(tk.Label):
    def __init__(self, master, text="", fg=None):
        super().__init__(master, text=text, font=F_CHIP, bg=T["panel_2"],
                         fg=fg or T["accent"], padx=9, pady=3)


class MiniBtn(tk.Label):
    def __init__(self, master, text, command):
        super().__init__(master, text=text, font=F_SMALL, fg=T["muted"],
                         bg=T["panel"], padx=8, pady=2, cursor="hand2")
        self._cmd = command
        self.bind("<Enter>", lambda e: self.config(fg=T["text"], bg=T["panel_2"]))
        self.bind("<Leave>", lambda e: self.config(fg=T["muted"], bg=T["panel"]))
        self.bind("<Button-1>", lambda e: self._cmd())


class SideButton(tk.Frame):
    def __init__(self, master, num, icon, title, sub, accent, command):
        super().__init__(master, bg=T["panel"], cursor="hand2")
        self.accent = accent
        self.command = command
        self.bar = tk.Frame(self, bg=T["panel"], width=3)
        self.bar.pack(side="left", fill="y")
        self.icon_lbl = tk.Label(self, text=icon, font=(BASE_FONT, 12, "bold"),
                                 fg=accent, bg=T["panel"], width=2)
        self.icon_lbl.pack(side="left", padx=(10, 4))
        txt = tk.Frame(self, bg=T["panel"])
        txt.pack(side="left", fill="both", expand=True, pady=6)
        head = f"{num:02d} · {title}" if isinstance(num, int) else title
        tk.Label(txt, text=head, font=F_BTN, fg=T["text"], bg=T["panel"], anchor="w").pack(fill="x")
        tk.Label(txt, text=sub, font=F_BTNSUB, fg=T["muted"], bg=T["panel"], anchor="w").pack(fill="x")
        self._bind_tree("<Enter>", self._enter)
        self._bind_tree("<Leave>", self._leave)
        self._bind_tree("<Button-1>", self._click)

    def _bind_tree(self, seq, func):
        self.bind(seq, func)
        for ch in self.winfo_children():
            ch.bind(seq, func)
            for gch in ch.winfo_children():
                gch.bind(seq, func)

    def _paint(self, bg):
        self.config(bg=bg)
        for ch in self.winfo_children():
            try:
                ch.config(bg=bg)
            except Exception:
                pass
            for gch in ch.winfo_children():
                try:
                    gch.config(bg=bg)
                except Exception:
                    pass

    def _enter(self, _):
        self._paint(T["panel_2"])
        self.bar.config(bg=self.accent)

    def _leave(self, _):
        self._paint(T["panel"])
        self.bar.config(bg=T["panel"])

    def _click(self, _):
        self.bar.config(bg=T["accent"])
        self.after(140, lambda: self.bar.config(
            bg=self.accent if self.cget("bg") == T["panel_2"] else T["panel"]))
        self.command()


class ParamDialog(tk.Toplevel):
    """fields: (ключ, подпись, значение, тип, доп.); int/float → (мин,макс), choice → варианты."""

    def __init__(self, parent, title, fields, hint=None, ok_text="Выполнить"):
        super().__init__(parent)
        self.result = None
        self.entries = {}
        self._widgets = []
        self.configure(bg=T["panel"])
        self.transient(parent)
        self.resizable(False, False)

        head = tk.Frame(self, bg=T["panel"])
        head.pack(fill="x", padx=18, pady=(16, 6))
        tk.Label(head, text=title, font=F_DLG_T, fg=T["text"], bg=T["panel"]).pack(anchor="w")
        if hint:
            tk.Label(head, text=hint, font=F_SMALL, fg=T["muted"], bg=T["panel"],
                     justify="left", wraplength=440).pack(anchor="w", pady=(3, 0))

        body = tk.Frame(self, bg=T["panel"])
        body.pack(fill="both", padx=18, pady=(6, 4))
        for key, label, default, kind, *rest in fields:
            extra = rest[0] if rest else None
            row = tk.Frame(body, bg=T["panel"])
            row.pack(fill="x", pady=4)
            tk.Label(row, text=label, width=30, anchor="w", font=F_BTNSUB,
                     fg=T["muted"], bg=T["panel"]).pack(side="left")
            if kind == "choice":
                var = tk.StringVar(value=default)
                cb = ttk.Combobox(row, textvariable=var, values=list(extra),
                                  state="readonly", width=26, style="Dark.TCombobox", font=F_CONS_S)
                cb.pack(side="left")
                self.entries[key] = (kind, var, None, None)
                self._widgets.append(cb)
            else:
                var = tk.StringVar(value=str(default))
                mn, mx = extra if extra else (None, None)
                en = tk.Entry(row, textvariable=var, width=14, font=F_CONS_S,
                              bg="#ffffff", fg=T["text"], insertbackground=T["text"],
                              relief="flat", highlightthickness=1,
                              highlightbackground=T["border"], highlightcolor=T["accent"])
                en.pack(side="left")
                try:
                    en.selection_range(0, "end")
                except Exception:
                    pass
                self.entries[key] = (kind, var, mn, mx)
                self._widgets.append(en)
                if extra:
                    tk.Label(row, text=f" [{extra[0]:g} … {extra[1]:g}]", font=F_SMALL,
                             fg=T["faint"], bg=T["panel"]).pack(side="left", padx=(6, 0))

        self.err_lbl = tk.Label(body, text="", fg=T["red"], font=F_SMALL, bg=T["panel"], anchor="w")
        self.err_lbl.pack(fill="x", pady=(4, 0))

        btns = tk.Frame(self, bg=T["panel"])
        btns.pack(fill="x", padx=18, pady=(6, 16))
        tk.Button(btns, text=ok_text, command=self._ok, bg=T["accent"], fg="#ffffff",
                  activebackground="#3f7fc4", activeforeground="#ffffff",
                  relief="flat", bd=0, font=(BASE_FONT, 10, "bold"),
                  padx=18, pady=6, cursor="hand2").pack(side="right")
        tk.Button(btns, text="Отмена", command=self._cancel, bg=T["panel"], fg=T["muted"],
                  activebackground=T["panel_2"], activeforeground=T["text"],
                  relief="flat", bd=0, font=F_BTN, padx=14, pady=6,
                  cursor="hand2").pack(side="right", padx=(0, 10))

        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self._cancel())
        if self._widgets:
            self._widgets[0].focus_set()

    def _fail(self, msg):
        self.err_lbl.config(text="⚠ " + msg)
        self.bell()

    def _ok(self):
        res = {}
        for key, (kind, var, mn, mx) in self.entries.items():
            raw = var.get().strip()
            # ИСПРАВЛЕНИЕ: заменяем запятую на точку ТОЛЬКО для чисел!
            if kind in ("int", "float"):
                raw = raw.replace(",", ".")
            try:
                val = int(float(raw)) if kind == "int" else (float(raw) if kind == "float" else raw)
            except ValueError:
                err_msg = f"Поле «{key}»: введите число" if kind in ("int", "float") else f"Поле «{key}»: некорректный ввод"
                return self._fail(err_msg)
            if kind in ("int", "float"):
                if (mn is not None and val < mn) or (mx is not None and val > mx):
                    return self._fail(f"Поле «{key}»: диапазон {mn:g} … {mx:g}")
            res[key] = val
        self.result = res
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

    def show(self):
        self.update_idletasks()
        try:
            pw, ph = self.master.winfo_width(), self.master.winfo_height()
            px, py = self.master.winfo_rootx(), self.master.winfo_rooty()
            x = px + max((pw - self.winfo_reqwidth()) // 2, 0)
            y = py + max((ph - self.winfo_reqheight()) // 3, 0)
            self.geometry(f"+{x}+{y}")
        except Exception:
            pass
        self.grab_set()
        self.wait_window(self)
        return self.result