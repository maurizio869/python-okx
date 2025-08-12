# price_jump_eval_colab.py
"""Коллаб-ячейка: загрузка чекпойнта, расчёт предсказаний,
сохранение данных для визуализации (без вывода графика).
"""
from pathlib import Path

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve

# ─── ПАРАМЕТРЫ ────────────────────────────────────────────────────
EVAL_JSON = Path("candles_2d.json")   # файл свечей для теста
MODEL_PATH = Path("lstm_jump.pt")     # обученная модель
MODEL_META_PATH = MODEL_PATH.with_suffix(".meta.json")
OUT_DATA = Path("viz_data.npz")       # куда сохранить данные
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────

# defaults (will be overwritten by meta if present)
SEQ_LEN, PRED_WINDOW = 60, 5
THRESHOLD = 0.8   # порог вероятности для присвоения класса 1

def load_df(path: Path) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)
    df = pd.DataFrame(list(raw.values()))
    df["datetime"] = pd.to_datetime(df["x"], unit="s")
    return df.set_index("datetime").sort_index()

class EvalDS(Dataset):
    def __init__(self, df: pd.DataFrame, scaler: StandardScaler):
        price_feats = df[["o", "h", "l", "c"]].astype(np.float32).values
        volumes     = df["v"].astype(np.float32).values.reshape(-1, 1)

        windows = []
        for i in range(SEQ_LEN, len(df) - PRED_WINDOW):
            window_raw_prices = price_feats[i - SEQ_LEN + 1 : i + 1].copy()
            ref_open          = window_raw_prices[0, 0]
            window_rel_prices = window_raw_prices / ref_open - 1.0

            window_raw_vol = volumes[i - SEQ_LEN + 1 : i + 1].copy()
            ref_vol        = max(float(window_raw_vol[0, 0]), 1e-8)
            window_rel_vol = window_raw_vol / ref_vol - 1.0

            window_rel = np.concatenate([window_rel_prices, window_rel_vol], axis=1)
            windows.append(scaler.transform(window_rel))

        self.samples = windows
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return torch.tensor(self.samples[idx])

class LSTMCls(nn.Module):
    def __init__(self, nfeat: int = 5, hidden: int = 64, layers: int = 2):
        super().__init__(); self.lstm = nn.LSTM(nfeat, hidden, layers, batch_first=True); self.fc = nn.Linear(hidden, 2)
    def forward(self, x): _, (h, _) = self.lstm(x); return self.fc(h[-1])

print("Читаем", EVAL_JSON)
df = load_df(EVAL_JSON)
print(f"Загружено {len(df)} свечей")

print("Загружаем модель", MODEL_PATH)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
# try load meta from checkpoint, otherwise side JSON
meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
if not meta and MODEL_META_PATH.exists():
    try:
        with open(MODEL_META_PATH, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
    except Exception:
        meta = {}
if isinstance(meta, dict):
    SEQ_LEN = int(meta.get("seq_len", SEQ_LEN))
    PRED_WINDOW = int(meta.get("pred_window", PRED_WINDOW))

# choose threshold: from meta if present, else default THRESHOLD
best_threshold = float(meta.get("threshold", THRESHOLD)) if isinstance(meta, dict) else THRESHOLD

model = LSTMCls(nfeat=5); model.load_state_dict(ckpt["model_state"]); model.to(DEVICE).eval()
scaler: StandardScaler = ckpt["scaler"]

loader = DataLoader(EvalDS(df, scaler), batch_size=512)
preds = np.zeros(len(loader.dataset), dtype=np.int8)
probs = np.zeros(len(loader.dataset), dtype=np.float32)

print("Делаем предсказания...")
with torch.no_grad():
    ptr = 0
    for xb in loader:
        outputs = model(xb.to(DEVICE))
        probs_batch = torch.softmax(outputs, dim=1).cpu().numpy()
        probs[ptr:ptr+len(probs_batch)] = probs_batch[:, 1]
        ptr += len(probs_batch)

# binarize by chosen threshold
preds = (probs >= best_threshold).astype(np.int8)

print(f"Сделано {len(preds)} предсказаний")
print(f"Используем порог: {best_threshold:.4f} ({'из meta' if 'threshold' in (meta or {}) else 'по умолчанию'})")
print(f"Предсказаний 0: {np.sum(preds == 0)}")
print(f"Предсказаний 1: {np.sum(preds == 1)}")
print(f"Процент предсказаний 1: {np.mean(preds == 1)*100:.2f}%")
print(f"Средняя уверенность модели: {np.mean(probs):.3f}")
print(f"Мин/макс уверенность: {np.min(probs):.3f} / {np.max(probs):.3f}")

print("Сохраняем", OUT_DATA)
np.savez_compressed(
    OUT_DATA,
    index=df.index.astype("int64").values,
    o=df["o"].values.astype(np.float32),
    h=df["h"].values.astype(np.float32),
    l=df["l"].values.astype(np.float32),
    c=df["c"].values.astype(np.float32),
    v=df["v"].values.astype(np.float32),
    preds=preds,
    probs=probs,
    seq_len=np.int32(SEQ_LEN),
    pred_window=np.int32(PRED_WINDOW),
    threshold=np.float32(best_threshold),
)
print("✓ Данные сохранены:", OUT_DATA)