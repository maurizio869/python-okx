# price_jump_visualize.py
"""Загружает файл viz_data.npz, сформированный скриптом price_jump_eval_colab.py,
и строит свечной график с отметками прогнозируемых скачков.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import mplfinance as mpf

# ─── НАСТРОЙКА ──────────────────────────────────────────────────────
DATA_FILE = Path("viz_data.npz")   # путь к файлу с сохранёнными данными
# ───────────────────────────────────────────────────────────────────

print("Читаем", DATA_FILE)
npz = np.load(DATA_FILE)

# Восстанавливаем DataFrame
idx = pd.to_datetime(npz["index"], utc=True)

df = pd.DataFrame({
    "o": npz["o"],
    "h": npz["h"],
    "l": npz["l"],
    "c": npz["c"],
}, index=idx)

preds = pd.Series(npz["preds"], index=idx[20:20 + len(npz["preds"])] )

# Подготовка для mplfinance
jumps = preds[preds == 1]
dfp = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close"})

kw = dict(type="candle", style="charles", volume=False,
          show_nontrading=True, datetime_format="%m-%d %H:%M", xrotation=15)

# Сохраняем даты скачков без таймзоны для последующего рисования
vdates = list(jumps.index.tz_localize(None)) if not jumps.empty else []

print("Рисуем график…")

# Рисуем график и получаем фигуру для последующего добавления линий
fig, axlist = mpf.plot(dfp, **kw, returnfig=True)

# Добавляем короткие вертикальные линии (0–10 % высоты оси цены)
if vdates:
    price_ax = axlist[0]  # основная ось с ценой
    for vd in vdates:
        price_ax.axvline(vd, color="blue", ymin=0.0, ymax=0.12,
                         linewidth=1.2, alpha=0.8)

import matplotlib.pyplot as plt
plt.show()