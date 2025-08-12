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

seq_len = int(npz.get("seq_len", 60)) if hasattr(npz, "get") else int(npz["seq_len"]) if "seq_len" in npz.files else 60
threshold = float(npz["threshold"]) if "threshold" in npz.files else None

df = pd.DataFrame({
    "o": npz["o"],
    "h": npz["h"],
    "l": npz["l"],
    "c": npz["c"],
    "v": npz["v"],
}, index=idx)

preds = pd.Series(npz["preds"], index=idx[seq_len:seq_len + len(npz["preds"])])

# Подготовка для mplfinance
jumps = preds[preds == 1]
dfp = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})

kw = dict(type="candle", style="charles", volume=True,
          show_nontrading=True, datetime_format="%m-%d %H:%M", xrotation=15)

# Сохраняем даты скачков без таймзоны для последующего рисования
vdates = list(jumps.index.tz_localize(None)) if not jumps.empty else []

print("Рисуем график…")
if threshold is not None:
    print(f"Порог классификации (LR+): {threshold:.4f}")

# Рисуем график и получаем фигуру для последующего добавления линий
fig, axlist = mpf.plot(dfp, **kw, returnfig=True)

# Добавляем короткие вертикальные линии (0–10 % высоты оси цены)
if vdates:
    price_ax = axlist[0]  # основная ось с ценой
    y_min, y_max = price_ax.get_ylim()
    margin = (y_max - y_min) * 0.02  # отступ 2 % от цены

    # Индекс без таймзоны для удобного поиска цен
    df_no_tz = dfp.copy()
    df_no_tz.index = df_no_tz.index.tz_localize(None)

    for vd in vdates:
        # Цена минимума текущей свечи
        low_price = df_no_tz.loc[vd, "Low"]
        top_y = max(y_min, low_price - margin)
        price_ax.vlines(vd, y_min, top_y,
                        colors="blue", linewidth=1.2, alpha=0.8)

import matplotlib.pyplot as plt
plt.show()