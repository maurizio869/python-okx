# visualize.py
# Last modified (MSK): 2025-01-03 18:56 — правка номер 2
# Changes:
# - правка 1: Добавлена поддержка отображения jump (синие) и drop (оранжевые) предсказаний
# - правка 2: Переименован из price_jump_visualize.py в visualize.py
"""Загружает файл viz_data.npz и строит свечной график с отметками 
прогнозируемых скачков (синие линии снизу) и падений (оранжевые линии сверху).
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

seq_len = int(npz.get("seq_len", 30)) if hasattr(npz, "get") else int(npz["seq_len"]) if "seq_len" in npz.files else 30

df = pd.DataFrame({
    "o": npz["o"],
    "h": npz["h"],
    "l": npz["l"],
    "c": npz["c"],
    "v": npz["v"],
}, index=idx)

# Проверяем формат данных - новый (jump+drop) или старый (только jump)
if "preds_jump" in npz.files and "preds_drop" in npz.files:
    # Новый формат с двумя моделями
    print("Обнаружен новый формат с jump и drop предсказаниями")
    preds_jump = pd.Series(npz["preds_jump"], index=idx[seq_len:seq_len + len(npz["preds_jump"])])
    preds_drop = pd.Series(npz["preds_drop"], index=idx[seq_len:seq_len + len(npz["preds_drop"])])
    jumps = preds_jump[preds_jump == 1]
    drops = preds_drop[preds_drop == 1]
    threshold_jump = float(npz["threshold_jump"]) if "threshold_jump" in npz.files else None
    threshold_drop = float(npz["threshold_drop"]) if "threshold_drop" in npz.files else None
elif "preds" in npz.files:
    # Старый формат с одной моделью (обратная совместимость)
    print("Обнаружен старый формат с одной моделью")
    preds = pd.Series(npz["preds"], index=idx[seq_len:seq_len + len(npz["preds"])])
    jumps = preds[preds == 1]
    drops = None
    threshold_jump = float(npz["threshold"]) if "threshold" in npz.files else None
    threshold_drop = None
else:
    print("⚠️ Не найдены предсказания в файле")
    jumps = pd.Series([], dtype=int)
    drops = None
    threshold_jump = None
    threshold_drop = None
dfp = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})

kw = dict(type="candle", style="charles", volume=True,
          show_nontrading=True, datetime_format="%m-%d %H:%M", xrotation=15)

# Сохраняем даты без таймзоны для последующего рисования
jump_dates = list(jumps.index.tz_localize(None)) if not jumps.empty else []
drop_dates = list(drops.index.tz_localize(None)) if drops is not None and not drops.empty else []

print("Рисуем график…")
if threshold_jump is not None:
    print(f"Порог для jump: {threshold_jump:.4f}")
if threshold_drop is not None:
    print(f"Порог для drop: {threshold_drop:.4f}")
print(f"Найдено jump предсказаний: {len(jump_dates)}")
print(f"Найдено drop предсказаний: {len(drop_dates)}")

# Рисуем график и получаем фигуру для последующего добавления линий
fig, axlist = mpf.plot(dfp, **kw, returnfig=True)

price_ax = axlist[0]  # основная ось с ценой
y_min, y_max = price_ax.get_ylim()
margin = (y_max - y_min) * 0.02  # отступ 2 % от цены

# Индекс без таймзоны для удобного поиска цен
df_no_tz = dfp.copy()
df_no_tz.index = df_no_tz.index.tz_localize(None)

# Добавляем синие линии для jump (снизу от свечи)
if jump_dates:
    for vd in jump_dates:
        # Цена минимума текущей свечи
        low_price = df_no_tz.loc[vd, "Low"]
        top_y = max(y_min, low_price - margin)
        price_ax.vlines(vd, y_min, top_y,
                        colors="blue", linewidth=1.2, alpha=0.8)

# Добавляем оранжевые линии для drop (сверху от свечи)
if drop_dates:
    for vd in drop_dates:
        # Цена максимума текущей свечи
        high_price = df_no_tz.loc[vd, "High"]
        bottom_y = min(y_max, high_price + margin)
        price_ax.vlines(vd, bottom_y, y_max,
                        colors="orange", linewidth=1.2, alpha=0.8)

import matplotlib.pyplot as plt
plt.show()