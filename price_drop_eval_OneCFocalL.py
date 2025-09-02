#!/usr/bin/env python3
# price_drop_eval_OneCFocalL.py
# Last modified (MSK): 2025-09-01 15:10 — правка номер 1
# Changes:
# - Created new eval script for price_drop compatible with OneCFocalL models
# - Uses 7 features, no normalization, same preprocessing as training

"""
Загружает модель из price_drop OneCFocalL и проверяет на candles_eval.json.
Сохраняет результаты в viz_data_drop.npz для визуализации.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, average_precision_score, roc_auc_score

# ─── ПАРАМЕТРЫ ────────────────────────────────────────────────────
EVAL_JSON = Path("candles_eval.json")   # файл свечей для теста
MODEL_PATH = Path("lstm_drop.pt")       # обученная модель (переименованная из lstm_drop_PRAUC.pt)
PNL_MODEL_PATH = Path("lstm_drop_pnl.pt")  # модель, отобранная по PnL
OUT_DATA = Path("viz_data_drop.npz")    # куда сохранить данные
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Параметры из OneCFocalL
SEQ_LEN = 30
PRED_WINDOW = 5
DROP_THRESHOLD = 0.0035  # Падение на 0.35%
DEFAULT_DROPOUT = 0.35

# Комиссии
MAKER_FEE = 0.0002
TAKER_FEE = 0.0005
USE_MAKER_FEES = False
ENTRY_FEE = MAKER_FEE if USE_MAKER_FEES else TAKER_FEE
EXIT_FEE = MAKER_FEE if USE_MAKER_FEES else TAKER_FEE

# ─── МОДЕЛЬ (копия из OneCFocalL) ─────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 2, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# ─── ЗАГРУЗКА ДАННЫХ ───────────────────────────────────────────────
def load_df(path: Path) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)
    df = pd.DataFrame(list(raw.values()))
    df["datetime"] = pd.to_datetime(df["x"], unit="s")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df.set_index("datetime").sort_index()

# ─── DATASET (как в OneCFocalL, но для eval) ──────────────────────
class EvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.closes = df["close"].astype(np.float32).values
        self.opens = df["open"].astype(np.float32).values
        self.highs = df["high"].astype(np.float32).values
        self.lows = df["low"].astype(np.float32).values
        self.volumes = df["volume"].astype(np.float32).values
        
        # Собираем индексы валидных окон
        self.indices = []
        self.labels = []
        
        for i in range(SEQ_LEN, len(self.closes) - PRED_WINDOW):
            self.indices.append(i)
            
            # Вычисляем метку для DROP (падение цены)
            current_open = float(self.opens[i])
            min_close = float(np.min(self.closes[i+1:i+PRED_WINDOW+1]))  # Ищем минимум для падения
            label = 1 if (min_close / max(current_open, 1e-12) - 1.0) <= -DROP_THRESHOLD else 0
            self.labels.append(label)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        
        # Берем окно данных (как в OneCFocalL)
        closes_w = self.closes[i-SEQ_LEN+1:i+1]
        opens_w = self.opens[i-SEQ_LEN+1:i+1]
        highs_w = self.highs[i-SEQ_LEN+1:i+1]
        lows_w = self.lows[i-SEQ_LEN+1:i+1]
        vols_w = self.volumes[i-SEQ_LEN+1:i+1]
        
        # Вычисляем дополнительные фичи (как в OneCFocalL)
        body = np.abs(closes_w - opens_w) + 1e-12
        upper_w = np.clip(highs_w - np.maximum(opens_w, closes_w), 0.0, None)
        lower_w = np.clip(np.minimum(opens_w, closes_w) - lows_w, 0.0, None)
        ratio_up = upper_w / body
        ratio_dn = lower_w / body
        
        # Собираем в том же порядке, что и в OneCFocalL
        x_seq = np.stack([
            closes_w,
            opens_w,
            highs_w,
            lows_w,
            vols_w,
            ratio_up,
            ratio_dn,
        ], axis=0).astype(np.float32)  # (7, SEQ_LEN)
        
        return torch.from_numpy(x_seq), self.labels[idx]

# ─── ОСНОВНОЙ КОД ──────────────────────────────────────────────────
print(f"Читаем {EVAL_JSON}")
df = load_df(EVAL_JSON)
print(f"Загружено {len(df)} свечей")

# Создаем dataset
eval_ds = EvalDataset(df)
eval_loader = DataLoader(eval_ds, batch_size=512, shuffle=False)

print(f"Загружаем модель {MODEL_PATH}")
if not MODEL_PATH.exists():
    print(f"⚠️  Файл {MODEL_PATH} не найден!")
    print("   Переименуйте одну из моделей OneCFocalL:")
    print("   - lstm_drop_PRAUC.pt (лучшая по PR_AUC)")
    print("   - lstm_drop_pnl.pt (лучшая по PnL)")
    print("   - lstm_drop_valacc.pt (лучшая по точности)")
    exit(1)

# Загружаем модель
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = LSTMClassifier()
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE).eval()

print(f"Модель загружена (эпоха {ckpt.get('epoch', 'N/A')})")

# Делаем предсказания
all_probs = []
all_labels = []
all_indices = []

with torch.no_grad():
    for xb, yb in eval_loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)[:, 1]
        
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(yb.numpy())
        
        # Сохраняем индексы для синхронизации с данными
        batch_size = len(yb)
        start_idx = len(all_indices)
        all_indices.extend(eval_ds.indices[start_idx:start_idx+batch_size])

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# Вычисляем метрики
pr_auc = average_precision_score(all_labels, all_probs)
roc_auc = roc_auc_score(all_labels, all_probs)

print(f"\nМетрики на eval:")
print(f"PR_AUC: {pr_auc:.4f}")
print(f"ROC_AUC: {roc_auc:.4f}")

# Находим оптимальный порог
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
optimal_idx = np.argmax(tpr - fpr)
best_threshold = thresholds[optimal_idx]
print(f"Оптимальный порог: {best_threshold:.4f}")

# Применяем порог
preds = (all_probs >= best_threshold).astype(int)
accuracy = np.mean(preds == all_labels)
print(f"Точность: {accuracy:.4f}")

# Подсчет сигналов
n_signals = np.sum(preds)
print(f"Сигналов: {n_signals} из {len(preds)} ({n_signals/len(preds)*100:.1f}%)")

# Проверяем модель PnL если есть
if PNL_MODEL_PATH.exists():
    print(f"\nЗагружаем модель PnL: {PNL_MODEL_PATH}")
    ckpt_pnl = torch.load(PNL_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model_pnl = LSTMClassifier()
    model_pnl.load_state_dict(ckpt_pnl["model_state"])
    model_pnl.to(DEVICE).eval()
    
    # Предсказания от PnL модели
    pnl_probs = []
    with torch.no_grad():
        for xb, _ in eval_loader:
            xb = xb.to(DEVICE)
            logits = model_pnl(xb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            pnl_probs.extend(probs.cpu().numpy())
    
    pnl_probs = np.array(pnl_probs)
    pnl_preds = (pnl_probs >= best_threshold).astype(int)
    pnl_signals = np.sum(pnl_preds)
    print(f"PnL модель - сигналов: {pnl_signals} ({pnl_signals/len(pnl_preds)*100:.1f}%)")

# Подготовка данных для сохранения
# Создаем массивы полной длины с NaN для предсказаний
full_probs = np.full(len(df), np.nan, dtype=np.float32)
full_preds = np.full(len(df), -1, dtype=np.int32)

# Заполняем предсказания в правильных позициях
for idx, prob in zip(all_indices, all_probs):
    full_probs[idx] = prob
    full_preds[idx] = 1 if prob >= best_threshold else 0

# Сохраняем результаты
print(f"\nСохраняем {OUT_DATA}")
np.savez_compressed(
    OUT_DATA,
    # Временные метки и OHLCV
    index=df.index.astype("int64").values,
    o=df["open"].values.astype(np.float32),
    h=df["high"].values.astype(np.float32),
    l=df["low"].values.astype(np.float32),
    c=df["close"].values.astype(np.float32),
    v=df["volume"].values.astype(np.float32),
    # Предсказания
    preds=full_preds,
    probs=full_probs,
    # Параметры
    seq_len=np.int32(SEQ_LEN),
    pred_window=np.int32(PRED_WINDOW),
    threshold=np.float32(best_threshold),
    drop_threshold=np.float32(DROP_THRESHOLD),
    # Комиссии
    maker_fee=np.float32(MAKER_FEE),
    taker_fee=np.float32(TAKER_FEE),
    use_maker_fees=np.bool_(USE_MAKER_FEES),
    entry_fee=np.float32(ENTRY_FEE),
    exit_fee=np.float32(EXIT_FEE),
    # Метрики
    pr_auc=np.float32(pr_auc),
    roc_auc=np.float32(roc_auc),
    accuracy=np.float32(accuracy),
)

print(f"✓ Данные сохранены: {OUT_DATA}")
print(f"  Используйте price_drop_visualize.py для визуализации")