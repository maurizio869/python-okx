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

# ─── ПАРАМЕТРЫ ────────────────────────────────────────────────────
EVAL_JSON = Path("candles_2d.json")   # файл свечей для теста
MODEL_PATH = Path("lstm_jump.pt")     # обученная модель
OUT_DATA = Path("viz_data.npz")       # куда сохранить данные
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────

SEQ_LEN, PRED_WINDOW = 20, 5
THRESHOLD = 0.8   # порог вероятности для присвоения класса 1

def load_df(path: Path) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)
    df = pd.DataFrame(list(raw.values()))
    df["datetime"] = pd.to_datetime(df["x"], unit="s")
    return df.set_index("datetime").sort_index()

class EvalDS(Dataset):
    def __init__(self, df: pd.DataFrame, scaler: StandardScaler):
        raw_feats = df[["o", "h", "l", "c"]].astype(np.float32).values

        windows = []
        for i in range(SEQ_LEN, len(df) - PRED_WINDOW):
            window_raw = raw_feats[i - SEQ_LEN + 1 : i + 1].copy()
            ref_open   = window_raw[0, 0]
            window_rel = window_raw / ref_open - 1.0
            windows.append(scaler.transform(window_rel))

        self.samples = windows
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return torch.tensor(self.samples[idx])

class LSTMCls(nn.Module):
    def __init__(self, nfeat: int = 4, hidden: int = 64, layers: int = 2):
        super().__init__(); self.lstm = nn.LSTM(nfeat, hidden, layers, batch_first=True); self.fc = nn.Linear(hidden, 2)
    def forward(self, x): _, (h, _) = self.lstm(x); return self.fc(h[-1])

print("Читаем", EVAL_JSON)
df = load_df(EVAL_JSON)
print(f"Загружено {len(df)} свечей")

print("Загружаем модель", MODEL_PATH)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = LSTMCls(); model.load_state_dict(ckpt["model_state"]); model.to(DEVICE).eval()
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
        p = (probs_batch[:, 1] > THRESHOLD).astype(np.int8)
        preds[ptr:ptr+len(p)] = p
        probs[ptr:ptr+len(p)] = probs_batch[:, 1]  # вероятность класса 1
        ptr += len(p)

print(f"Сделано {len(preds)} предсказаний")
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
    preds=preds,
    probs=probs,
)
print("✓ Данные сохранены:", OUT_DATA)