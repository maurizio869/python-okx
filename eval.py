# eval.py
# Last modified (MSK): 2025-09-03 21:20 — правка номер 7
# Changes:
# - правка 1: Создан единый eval скрипт для обеих моделей (jump и drop)
# - правка 2: Переименован из price_jump_drop_eval_OneCFocalL.py в eval.py
# - правка 3: Исправлена загрузка модели - параметры архитектуры берутся из checkpoint
# - правка 4: Исправлены дата и время в шапке на правильные из системы Linux
# - правка 5: Добавлен флаг USE_CONSTANT_THRESHOLD для использования фиксированных порогов вместо подбираемых
# - правка 6: Добавлен расчет PnL VAS для комбинированных сигналов jump и drop с сохранением сделок для визуализации
# - правка 7: Переименована метрика PnL VAS в PnL VAS2 (двойная стратегия)
"""Единый eval скрипт для jump и drop моделей OneCFocalL"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve

# ─── ПАРАМЕТРЫ ────────────────────────────────────────────────────
EVAL_JSON = Path("candles_eval.json")       # файл свечей для теста
JUMP_MODEL_PATH = Path("lstm_jump.pt")      # модель для jump
DROP_MODEL_PATH = Path("lstm_drop.pt")      # модель для drop
OUT_DATA = Path("viz_data.npz")             # куда сохранить данные
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Комиссии
MAKER_FEE = 0.0002
TAKER_FEE = 0.0005
USE_MAKER_FEES = False
ENTRY_FEE = MAKER_FEE if USE_MAKER_FEES else TAKER_FEE
EXIT_FEE = MAKER_FEE if USE_MAKER_FEES else TAKER_FEE

# Defaults (будут перезаписаны из моделей если есть)
SEQ_LEN = 30
PRED_WINDOW = 5
JUMP_THRESHOLD = 0.0035
DROP_THRESHOLD = 0.0035

# Флаг для использования фиксированных порогов
USE_CONSTANT_THRESHOLD = False
CONSTANT_JUMP_THRESHOLD = 0.5
CONSTANT_DROP_THRESHOLD = 0.5

# PnL_VAS parameters (sequential, dynamic exit)
PNL_VAS_THRESH_PCT = 0.0025   # +0.25% above/below entry open
PNL_VAS_MAX_HOLD_MIN = 10     # fallback hold minutes if no early exit
PNL_VAS_SWEEP_THR = 0.55      # fixed threshold for SL sweep
PNL_VAS_SL_MIN = -0.015       # -1.5%
PNL_VAS_SL_MAX = -0.0001      # -0.01%
PNL_VAS_SL_STEP = 0.0001      # 0.01%

def load_df(path: Path) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)
    df = pd.DataFrame(list(raw.values()))
    df["datetime"] = pd.to_datetime(df["x"], unit="s")
    return df.set_index("datetime").sort_index()

class EvalDataset(Dataset):
    """Dataset для OneCFocalL моделей (7 фичей, без нормализации)"""
    def __init__(self, df: pd.DataFrame, seq_len: int):
        self.seq_len = seq_len
        self.df = df
        
        # Извлекаем данные
        self.closes = df['c'].astype(np.float32).values
        self.opens = df['o'].astype(np.float32).values
        self.highs = df['h'].astype(np.float32).values
        self.lows = df['l'].astype(np.float32).values
        self.volumes = df['v'].astype(np.float32).values
        
        # Количество возможных окон
        self.n_samples = len(df) - seq_len - PRED_WINDOW
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Индексы для окна
        start_idx = idx
        end_idx = idx + self.seq_len
        
        # Извлекаем окно данных
        closes_w = self.closes[start_idx:end_idx]
        opens_w = self.opens[start_idx:end_idx]
        highs_w = self.highs[start_idx:end_idx]
        lows_w = self.lows[start_idx:end_idx]
        vols_w = self.volumes[start_idx:end_idx]
        
        # Вычисляем дополнительные фичи (КАК В ОБУЧЕНИИ!)
        body = np.abs(closes_w - opens_w) + 1e-12
        
        # Обрезка отрицательных значений для хвостов
        upper_w = np.clip(highs_w - np.maximum(opens_w, closes_w), 0.0, None)
        lower_w = np.clip(np.minimum(opens_w, closes_w) - lows_w, 0.0, None)
        
        # Отношения
        ratio_up = upper_w / body
        ratio_dn = lower_w / body
        
        # Собираем фичи в правильном порядке (как в обучении)
        x_seq = np.stack([
            closes_w,    # 0
            opens_w,     # 1
            highs_w,     # 2
            lows_w,      # 3
            vols_w,      # 4
            ratio_up,    # 5
            ratio_dn     # 6
        ], axis=0).astype(np.float32)
        
        # Транспонируем для модели
        x_seq = x_seq.T  # (seq_len, 7)
        
        return torch.from_numpy(x_seq)

class LSTMClassifier(nn.Module):
    """Модель LSTM для OneCFocalL"""
    def __init__(self, input_size=7, hidden_size=128, num_layers=3, dropout=0.35):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 2)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

def load_model_and_predict(model_path: Path, df: pd.DataFrame, model_name: str):
    """Загружает модель и делает предсказания"""
    
    if not model_path.exists():
        print(f"⚠️ Модель {model_path} не найдена, пропускаем")
        return None, None, None
    
    print(f"\n📊 Обработка {model_name}:")
    print(f"Загружаем модель {model_path}")
    
    # Загружаем чекпойнт
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    # Извлекаем метаданные
    if isinstance(ckpt, dict) and "meta" in ckpt:
        meta = ckpt["meta"]
    else:
        meta = {}
    
    # Параметры из мета или дефолтные
    seq_len = int(meta.get("seq_len", SEQ_LEN))
    pred_window = int(meta.get("pred_window", PRED_WINDOW))
    
    # Параметры архитектуры модели (должны совпадать с train скриптом)
    hidden_size = int(meta.get("hidden_size", 64))  # По умолчанию как в OneCFocalL
    num_layers = int(meta.get("num_layers", 2))     # По умолчанию как в OneCFocalL
    dropout = float(meta.get("dropout", 0.25))      # По умолчанию как в OneCFocalL
    
    # Создаем модель
    model = LSTMClassifier(input_size=7, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    
    # Загружаем веса
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    
    model.to(DEVICE).eval()
    
    # Создаем DataLoader
    dataset = EvalDataset(df, seq_len)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    # Массивы для результатов
    probs = np.zeros(len(dataset), dtype=np.float32)
    
    print(f"Делаем предсказания для {len(dataset)} окон...")
    with torch.no_grad():
        ptr = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            outputs = model(batch)
            probs_batch = torch.softmax(outputs, dim=1).cpu().numpy()
            probs[ptr:ptr+len(probs_batch)] = probs_batch[:, 1]
            ptr += len(probs_batch)
    
    # Вычисляем оптимальный порог по ROC кривой
    # Для этого нужны истинные метки - берем из целевой функции обучения
    opens_arr = df["o"].astype(np.float32).values
    closes_arr = df["c"].astype(np.float32).values
    
    true_labels = []
    for i in range(seq_len, len(df) - pred_window):
        current_open = opens_arr[i]
        
        if "jump" in model_name.lower():
            # Для jump: максимальная цена в окне предсказания
            max_close = np.max(closes_arr[i+1:i+pred_window+1])
            label = 1 if (max_close / current_open - 1.0) >= JUMP_THRESHOLD else 0
        else:
            # Для drop: минимальная цена в окне предсказания
            min_close = np.min(closes_arr[i+1:i+pred_window+1])
            label = 1 if (min_close / current_open - 1.0) <= -DROP_THRESHOLD else 0
        
        true_labels.append(label)
    
    true_labels = np.array(true_labels)
    
    # Находим оптимальный порог
    fpr, tpr, thresholds = roc_curve(true_labels, probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Оптимальный порог (подобран): {optimal_threshold:.4f}")
    
    # Проверяем флаг USE_CONSTANT_THRESHOLD
    if USE_CONSTANT_THRESHOLD:
        if "jump" in model_name.lower():
            used_threshold = CONSTANT_JUMP_THRESHOLD
            print(f"Используем константу jump_threshold={used_threshold:.4f} для jump предсказаний")
        else:
            used_threshold = CONSTANT_DROP_THRESHOLD
            print(f"Используем константу drop_threshold={used_threshold:.4f} для drop предсказаний")
    else:
        used_threshold = optimal_threshold
    
    # Бинаризация предсказаний
    preds = (probs >= used_threshold).astype(np.int8)
    
    print(f"Предсказаний 0: {np.sum(preds == 0)}")
    print(f"Предсказаний 1: {np.sum(preds == 1)}")
    print(f"Процент предсказаний 1: {np.mean(preds == 1)*100:.2f}%")
    
    return probs, preds, used_threshold

# ─── ОСНОВНОЙ КОД ─────────────────────────────────────────────────
print("=" * 60)
print("Единый eval для Jump и Drop моделей OneCFocalL")
print("=" * 60)

print(f"\n📁 Читаем {EVAL_JSON}")
df = load_df(EVAL_JSON)
print(f"Загружено {len(df)} свечей")

# Обрабатываем модель Jump
probs_jump, preds_jump, threshold_jump = load_model_and_predict(
    JUMP_MODEL_PATH, df, "Jump"
)

# Обрабатываем модель Drop
probs_drop, preds_drop, threshold_drop = load_model_and_predict(
    DROP_MODEL_PATH, df, "Drop"
)

# ─── РАСЧЕТ PNL VAS2 ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("📈 Расчет PnL VAS2 для комбинированных сигналов")

def calculate_pnl_vas2(df, preds_jump, preds_drop, threshold_jump, threshold_drop, stop_loss_pct):
    """Расчет PnL VAS2 для комбинированных LONG и SHORT сигналов"""
    
    trades = []  # Список всех сделок для визуализации
    equity = 1.0
    last_exit = -10**9
    
    # Подготовка данных
    opens = df['o'].values
    highs = df['h'].values
    lows = df['l'].values
    closes = df['c'].values
    
    # Индексы где есть сигналы
    jump_indices = np.where(preds_jump == 1)[0] + SEQ_LEN if preds_jump is not None else np.array([])
    drop_indices = np.where(preds_drop == 1)[0] + SEQ_LEN if preds_drop is not None else np.array([])
    
    # Объединяем и сортируем все сигналы
    all_signals = []
    for idx in jump_indices:
        all_signals.append((idx, 'long'))
    for idx in drop_indices:
        all_signals.append((idx, 'short'))
    all_signals.sort(key=lambda x: x[0])
    
    # Обрабатываем сигналы последовательно
    for signal_idx, signal_type in all_signals:
        if signal_idx < last_exit:
            continue  # Пропускаем если еще в позиции
            
        # Проверяем перекрытие сигналов
        if signal_type == 'long' and signal_idx in drop_indices:
            continue  # Пропускаем при одновременных сигналах
        if signal_type == 'short' and signal_idx in jump_indices:
            continue  # Пропускаем при одновременных сигналах
            
        entry_idx = signal_idx
        entry_open = float(opens[entry_idx])
        
        if not np.isfinite(entry_open) or entry_open <= 0:
            continue
            
        exit_idx = None
        max_hold = max(PNL_VAS_MAX_HOLD_MIN, PRED_WINDOW)
        
        # Сканируем следующие свечи для выхода
        for k in range(1, max_hold + 1):
            j = entry_idx + k
            if j >= len(opens):
                break
                
            if signal_type == 'long':
                # LONG: стоп-лосс на падении
                low_j = float(lows[j])
                if (low_j / entry_open - 1.0) <= stop_loss_pct:
                    exit_idx = j
                    break
                    
                # Динамический выход при росте
                close_j = float(closes[j])
                open_j = float(opens[j])
                if j > 0:
                    close_prev = float(closes[j-1])
                    open_prev = float(opens[j-1])
                    prev_green = (close_prev > open_prev)
                    body_current = close_j - open_j
                    body_prev = close_prev - open_prev
                    body_smaller = (body_current < body_prev)
                    price_up_enough = ((close_j / entry_open - 1.0) >= PNL_VAS_THRESH_PCT)
                    
                    if body_smaller and prev_green and price_up_enough:
                        exit_idx = j
                        break
                        
            else:  # SHORT
                # SHORT: стоп-лосс на росте
                high_j = float(highs[j])
                if (high_j / entry_open - 1.0) >= abs(stop_loss_pct):
                    exit_idx = j
                    break
                    
                # Динамический выход при падении
                close_j = float(closes[j])
                open_j = float(opens[j])
                if j > 0:
                    close_prev = float(closes[j-1])
                    open_prev = float(opens[j-1])
                    prev_red = (close_prev < open_prev)
                    body_current = abs(close_j - open_j)
                    body_prev = abs(close_prev - open_prev)
                    body_smaller = (body_current < body_prev)
                    price_down_enough = ((close_j / entry_open - 1.0) <= -PNL_VAS_THRESH_PCT)
                    
                    if body_smaller and prev_red and price_down_enough:
                        exit_idx = j
                        break
        
        # Выход по таймауту если не сработали другие условия
        if exit_idx is None:
            exit_idx = min(entry_idx + max_hold, len(closes) - 1)
            
        exit_close = float(closes[exit_idx])
        
        # Расчет PnL
        if signal_type == 'long':
            pnl = (exit_close * (1.0 - EXIT_FEE)) / (entry_open * (1.0 + ENTRY_FEE)) - 1.0
        else:  # SHORT
            pnl = (entry_open * (1.0 - EXIT_FEE)) / (exit_close * (1.0 + ENTRY_FEE)) - 1.0
            
        equity *= (1.0 + pnl)
        last_exit = exit_idx
        
        # Сохраняем сделку для визуализации
        trades.append({
            'type': signal_type,
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'entry_price': entry_open,
            'exit_price': exit_close,
            'pnl': pnl
        })
    
    return (equity - 1.0) * 100.0, trades

# Подбор оптимального стоп-лосса
if preds_jump is not None or preds_drop is not None:
    sl_values = np.arange(PNL_VAS_SL_MIN, PNL_VAS_SL_MAX + 1e-12, PNL_VAS_SL_STEP)
    best_sl = PNL_VAS_SL_MIN
    best_pnl_vas2 = -np.inf
    best_trades = []
    
    print(f"Подбор стоп-лосса из {len(sl_values)} вариантов...")
    for sl in sl_values:
        pnl_here, trades_here = calculate_pnl_vas2(df, preds_jump, preds_drop, 
                                                   threshold_jump, threshold_drop, float(sl))
        if pnl_here > best_pnl_vas2:
            best_pnl_vas2 = pnl_here
            best_sl = float(sl)
            best_trades = trades_here
    
    print(f"✅ Выбран стоп-лосс: {best_sl*100:.2f}%")
    print(f"📊 PnL VAS2 (compound): {best_pnl_vas2:.2f}%")
    print(f"📝 Количество сделок: {len(best_trades)}")
    if best_trades:
        long_trades = [t for t in best_trades if t['type'] == 'long']
        short_trades = [t for t in best_trades if t['type'] == 'short']
        print(f"   - LONG сделок: {len(long_trades)}")
        print(f"   - SHORT сделок: {len(short_trades)}")
else:
    best_pnl_vas2 = 0.0
    best_sl = PNL_VAS_SL_MIN
    best_trades = []

# Подготавливаем данные для сохранения
save_data = {
    "index": df.index.astype("int64").values,
    "o": df["o"].values.astype(np.float32),
    "h": df["h"].values.astype(np.float32),
    "l": df["l"].values.astype(np.float32),
    "c": df["c"].values.astype(np.float32),
    "v": df["v"].values.astype(np.float32),
    "seq_len": np.int32(SEQ_LEN),
    "pred_window": np.int32(PRED_WINDOW),
    "maker_fee": np.float32(MAKER_FEE),
    "taker_fee": np.float32(TAKER_FEE),
    "use_maker_fees": np.bool_(USE_MAKER_FEES),
    "entry_fee": np.float32(ENTRY_FEE),
    "exit_fee": np.float32(EXIT_FEE),
    # PnL VAS2 данные
    "pnl_vas2": np.float32(best_pnl_vas2),
    "pnl_vas2_stop_loss": np.float32(best_sl),
    "trades": best_trades,  # Список сделок для визуализации
}

# Добавляем данные jump если есть
if probs_jump is not None:
    save_data.update({
        "probs_jump": probs_jump,
        "preds_jump": preds_jump,
        "threshold_jump": np.float32(threshold_jump),
    })

# Добавляем данные drop если есть
if probs_drop is not None:
    save_data.update({
        "probs_drop": probs_drop,
        "preds_drop": preds_drop,
        "threshold_drop": np.float32(threshold_drop),
    })

# Для обратной совместимости - если есть только jump, дублируем как старые поля
if probs_jump is not None and probs_drop is None:
    save_data.update({
        "probs": probs_jump,
        "preds": preds_jump,
        "threshold": np.float32(threshold_jump),
    })

print(f"\n💾 Сохраняем {OUT_DATA}")
np.savez_compressed(OUT_DATA, **save_data)
print(f"✅ Данные сохранены: {OUT_DATA}")
print("=" * 60)