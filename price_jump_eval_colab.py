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

# derive ground-truth labels (same logic as train)
opens  = df["o"].astype(np.float32).values
closes = df["c"].astype(np.float32).values
labels = []
for i in range(SEQ_LEN, len(df) - PRED_WINDOW):
    current_open = opens[i]
    max_close    = closes[i + 1 : i + PRED_WINDOW + 1].max()
    jump         = (max_close / current_open - 1) >= 0.0035
    labels.append(1 if jump else 0)
y_true = np.asarray(labels, dtype=np.int8)

# select threshold by maximizing TPR/FPR (LR+)
fpr, tpr, thr = roc_curve(y_true, probs)
lr_plus = tpr / np.maximum(fpr, 1e-6)
best_idx_lr = int(np.argmax(lr_plus))
best_threshold_lr = float(thr[best_idx_lr])

# PnL-based threshold sweep (entry at current open, exit at close after PRED_WINDOW)
entry_opens = opens[SEQ_LEN : len(df) - PRED_WINDOW]
exit_closes = closes[SEQ_LEN + PRED_WINDOW : len(df)]
ret_per_trade = exit_closes / np.maximum(entry_opens, 1e-12) - 1.0  # percentage returns

thr_min, thr_max, thr_step = 0.05, 0.95, 0.05
print(f"Перебор порога по PnL: min={thr_min:.2f}, max={thr_max:.2f}, step={thr_step:.2f}")
thresholds = np.arange(thr_min, thr_max + 1e-9, thr_step)

best_comp_ret = -np.inf
best_threshold_pnl = thresholds[0]
best_trades = 0

def safe_sharpe(r):
    if r.size < 2:
        return 0.0
    std = np.std(r)
    return float(np.mean(r) / (std + 1e-12))

for t in thresholds:
    mask = (probs >= t)
    n_trades = int(mask.sum())
    if n_trades == 0:
        comp_ret = -np.inf
        sharpe = 0.0
    else:
        r = ret_per_trade[mask]
        # compounded return
        # guard for any r <= -1.0
        if np.any(r <= -0.999999):
            comp_ret = -1.0
        else:
            comp_ret = float(np.exp(np.sum(np.log1p(r))) - 1.0)
        sharpe = safe_sharpe(r)
    print(f"thr={t:.2f} trades={n_trades} comp_ret={comp_ret*100 if np.isfinite(comp_ret) else float('nan'):.2f}% sharpe={sharpe:.2f}")
    if comp_ret > best_comp_ret:
        best_comp_ret = comp_ret
        best_threshold_pnl = float(t)
        best_trades = n_trades

best_threshold = best_threshold_pnl

preds = (probs >= best_threshold).astype(np.int8)

print(f"Сделано {len(preds)} предсказаний")
print(f"Выбран порог по PnL: {best_threshold:.4f}, comp_ret={best_comp_ret*100 if np.isfinite(best_comp_ret) else float('nan'):.2f}% trades={best_trades}")
print(f"(Справочно) Порог по LR+ (TPR/FPR): {best_threshold_lr:.4f}, LR+={lr_plus[best_idx_lr]:.2f}")
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