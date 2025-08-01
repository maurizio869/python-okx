# price_jump_visualize_dual.py
"""Загружает два файла viz_data.npz и строит их на одном графике.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

# ─── НАСТРОЙКА ──────────────────────────────────────────────────────
DATA_FILE1 = Path("viz_data1.npz")   # первый файл с данными
DATA_FILE2 = Path("viz_data2.npz")   # второй файл с данными
# ───────────────────────────────────────────────────────────────────

def load_viz_data(file_path: Path):
    """Загружает данные из .npz файла"""
    print(f"Читаем {file_path}")
    npz = np.load(file_path)
    
    # Восстанавливаем DataFrame
    idx = pd.to_datetime(npz["index"], utc=True)
    
    df = pd.DataFrame({
        "o": npz["o"],
        "h": npz["h"],
        "l": npz["l"],
        "c": npz["c"],
    }, index=idx)
    
    preds = pd.Series(npz["preds"], index=idx[20:20 + len(npz["preds"])])
    
    return df, preds

# Загружаем данные
df1, preds1 = load_viz_data(DATA_FILE1)
df2, preds2 = load_viz_data(DATA_FILE2)

# Подготовка для mplfinance
jumps1 = preds1[preds1 == 1]
jumps2 = preds2[preds2 == 1]

dfp1 = df1.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close"})
dfp2 = df2.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close"})

# Создаём фигуру с двумя subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Первый график
kw1 = dict(type="candle", style="charles", volume=False,
           show_nontrading=True, datetime_format="%m-%d %H:%M", xrotation=15,
           ax=ax1, title="Данные 1")
if not jumps1.empty:
    vdates1 = list(jumps1.index.tz_localize(None))
    kw1["vlines"] = vdates1

mpf.plot(dfp1, **kw1)

# Второй график
kw2 = dict(type="candle", style="charles", volume=False,
           show_nontrading=True, datetime_format="%m-%d %H:%M", xrotation=15,
           ax=ax2, title="Данные 2")
if not jumps2.empty:
    vdates2 = list(jumps2.index.tz_localize(None))
    kw2["vlines"] = vdates2

mpf.plot(dfp2, **kw2)

plt.tight_layout()
plt.show()