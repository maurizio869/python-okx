# visualize.py
# Last modified (MSK): 2025-09-03 23:40 — правка номер 6
# Changes:
# - правка 1: Добавлена поддержка отображения jump (синие) и drop (оранжевые) предсказаний
# - правка 2: Переименован из price_jump_visualize.py в visualize.py
# - правка 3: Добавлена визуализация сделок PnL VAS (зеленые прямоугольники для LONG, красные для SHORT) и аннотация с результатом
# - правка 4: Переименована метрика PnL VAS в PnL VAS2 (двойная стратегия)
# - правка 5: Добавлен allow_pickle=True для загрузки trades (object array)
# - правка 6: Улучшена визуализация сделок - добавлены слои комиссий и вывод всех метрик
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
npz = np.load(DATA_FILE, allow_pickle=True)

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
          show_nontrading=True, datetime_format="%m-%d %H:%M", xrotation=15,
          warn_too_much_data=10000)

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

# Визуализация сделок PnL VAS2 если есть
if "trades" in npz.files:
    trades = npz["trades"]
    if hasattr(trades, 'tolist'):
        trades = trades.tolist()  # Конвертируем из numpy если нужно
    
    if trades:
        print(f"\nВизуализация {len(trades)} сделок PnL VAS2")
        
        for trade in trades:
            entry_idx = trade['entry_idx']
            exit_idx = trade['exit_idx']
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            trade_type = trade['type']
            
            # Находим даты для входа и выхода
            if entry_idx < len(idx) and exit_idx < len(idx):
                entry_date = idx[entry_idx].tz_localize(None)
                exit_date = idx[exit_idx].tz_localize(None)
                
                # Получаем комиссии
                entry_fee = float(npz.get('entry_fee', 0.0005))
                exit_fee = float(npz.get('exit_fee', 0.0005))
                
                # Рисуем слоеный прямоугольник с комиссиями
                from matplotlib.patches import Rectangle
                import matplotlib.dates as mdates
                
                # Конвертируем даты в числовой формат matplotlib
                x_start = mdates.date2num(entry_date)
                x_end = mdates.date2num(exit_date)
                width = x_end - x_start
                
                if trade_type == 'long':
                    # LONG: покупаем по entry_price, продаем по exit_price
                    # Комиссия входа увеличивает цену покупки
                    actual_entry = entry_price * (1 + entry_fee)
                    # Комиссия выхода уменьшает цену продажи
                    actual_exit = exit_price * (1 - exit_fee)
                    
                    # Слой 1: Комиссия входа (желтая, снизу)
                    fee_entry_height = entry_price * entry_fee
                    rect_fee_entry = Rectangle((x_start, entry_price), width, fee_entry_height,
                                              linewidth=0, facecolor='yellow', alpha=0.3)
                    price_ax.add_patch(rect_fee_entry)
                    
                    # Слой 2: Основное тело сделки (зеленое)
                    body_bottom = actual_entry
                    body_height = actual_exit - actual_entry
                    if body_height > 0:  # Прибыльная сделка
                        rect_body = Rectangle((x_start, body_bottom), width, body_height,
                                             linewidth=0, facecolor='green', alpha=0.15)
                    else:  # Убыточная сделка
                        rect_body = Rectangle((x_start, actual_exit), width, abs(body_height),
                                             linewidth=0, facecolor='red', alpha=0.15)
                    price_ax.add_patch(rect_body)
                    
                    # Слой 3: Комиссия выхода (желтая, сверху)
                    fee_exit_height = exit_price * exit_fee
                    rect_fee_exit = Rectangle((x_start, actual_exit), width, fee_exit_height,
                                             linewidth=0, facecolor='yellow', alpha=0.3)
                    price_ax.add_patch(rect_fee_exit)
                    
                else:  # SHORT
                    # SHORT: продаем по entry_price, откупаем по exit_price
                    # Комиссия входа уменьшает цену продажи
                    actual_entry = entry_price * (1 - entry_fee)
                    # Комиссия выхода увеличивает цену откупа
                    actual_exit = exit_price * (1 + exit_fee)
                    
                    # Слой 1: Комиссия входа (желтая, сверху от entry)
                    fee_entry_height = entry_price * entry_fee
                    rect_fee_entry = Rectangle((x_start, actual_entry), width, fee_entry_height,
                                              linewidth=0, facecolor='yellow', alpha=0.3)
                    price_ax.add_patch(rect_fee_entry)
                    
                    # Слой 2: Основное тело сделки
                    body_bottom = min(actual_entry, actual_exit)
                    body_height = abs(actual_entry - actual_exit)
                    if actual_entry > actual_exit:  # Прибыльная SHORT
                        rect_body = Rectangle((x_start, body_bottom), width, body_height,
                                             linewidth=0, facecolor='green', alpha=0.15)
                    else:  # Убыточная SHORT
                        rect_body = Rectangle((x_start, body_bottom), width, body_height,
                                             linewidth=0, facecolor='red', alpha=0.15)
                    price_ax.add_patch(rect_body)
                    
                    # Слой 3: Комиссия выхода (желтая, снизу от exit)
                    fee_exit_height = exit_price * exit_fee
                    rect_fee_exit = Rectangle((x_start, exit_price), width, fee_exit_height,
                                             linewidth=0, facecolor='yellow', alpha=0.3)
                    price_ax.add_patch(rect_fee_exit)

# Добавляем аннотацию со всеми метриками под графиком
metrics_text_parts = []

# Основные метрики
if "pnl_vas2" in npz.files:
    pnl_vas2 = float(npz["pnl_vas2"])
    stop_loss = float(npz["pnl_vas2_stop_loss"])
    metrics_text_parts.append(f"PnL VAS2: {pnl_vas2:.2f}% (SL: {stop_loss*100:.2f}%)")

if "pnl_sum" in npz.files:
    pnl_sum = float(npz["pnl_sum"])
    metrics_text_parts.append(f"PnL Sum: {pnl_sum:.2f}%")

if "mean_ret" in npz.files and "median_ret" in npz.files:
    mean_ret = float(npz["mean_ret"])
    median_ret = float(npz["median_ret"])
    metrics_text_parts.append(f"Mean/Median: {mean_ret:.3f}%/{median_ret:.3f}%")

if "sharpe" in npz.files:
    sharpe = float(npz["sharpe"])
    metrics_text_parts.append(f"Sharpe: {sharpe:.2f}")

if "win_rate" in npz.files:
    win_rate = float(npz["win_rate"])
    metrics_text_parts.append(f"Win Rate: {win_rate:.1f}%")

if "profit_factor" in npz.files:
    profit_factor = float(npz["profit_factor"])
    if profit_factor < 1000:  # Не inf
        metrics_text_parts.append(f"PF: {profit_factor:.2f}")

# Drawdown метрики
dd_parts = []
if "max_equity_dd" in npz.files:
    max_equity_dd = float(npz["max_equity_dd"])
    dd_parts.append(f"Max Eq DD: {max_equity_dd:.2f}%")

if "avg_equity_dd" in npz.files:
    avg_equity_dd = float(npz["avg_equity_dd"])
    dd_parts.append(f"Avg Eq DD: {avg_equity_dd:.2f}%")

if "max_price_dd" in npz.files:
    max_price_dd = float(npz["max_price_dd"])
    dd_parts.append(f"Max Price DD: {max_price_dd:.2f}%")

if "avg_price_dd" in npz.files:
    avg_price_dd = float(npz["avg_price_dd"])
    dd_parts.append(f"Avg Price DD: {avg_price_dd:.2f}%")

# Информация о сделках
trade_info = []
if "trades_count" in npz.files:
    trades_count = int(npz["trades_count"])
    trade_info.append(f"Trades: {trades_count}")

if "long_trades" in npz.files and "short_trades" in npz.files:
    long_trades = int(npz["long_trades"])
    short_trades = int(npz["short_trades"])
    trade_info.append(f"Long/Short: {long_trades}/{short_trades}")

# Собираем весь текст
if metrics_text_parts or dd_parts or trade_info:
    # Первая строка - основные метрики
    line1 = " | ".join(metrics_text_parts) if metrics_text_parts else ""
    # Вторая строка - drawdown
    line2 = " | ".join(dd_parts) if dd_parts else ""
    # Третья строка - информация о сделках
    line3 = " | ".join(trade_info) if trade_info else ""
    
    # Объединяем строки
    full_text = "\n".join([l for l in [line1, line2, line3] if l])
    
    # Добавляем текст под графиком
    fig.text(0.5, 0.01, full_text, 
             horizontalalignment='center',
             fontsize=9, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.2),
             verticalalignment='bottom',
             multialignment='center')
    
    print(f"\n📊 Метрики:\n{full_text}")

import matplotlib.pyplot as plt
plt.show()