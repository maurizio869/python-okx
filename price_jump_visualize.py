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

preds = pd.Series(npz["preds"], index=idx[20:20 + len(npz["preds")]])

# Подготовка для mplfinance
jumps = preds[preds == 1]
dfp = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close"})

kw = dict(type="candle", style="charles", volume=False,
          show_nontrading=True, datetime_format="%m-%d %H:%M", xrotation=15)
if not jumps.empty:
    vdates = list(jumps.index.tz_localize(None))
    kw["vlines"] = vdates

print("Рисуем график…")
mpf.plot(dfp, **kw)
import matplotlib.pyplot as plt
plt.show()