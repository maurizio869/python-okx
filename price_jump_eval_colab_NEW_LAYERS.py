# price_jump_eval_colab_NEW_LAYERS.py
# Last modified (MSK): 2025-08-17 21:12
"""Инференс модели с расширенными признаками (upper_ratio, lower_ratio, body_sign).
Читает meta (seq_len, pred_window, threshold), считает признаки из OHLCV и выводит PnL.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

EVAL_JSON = Path("candles_2d.json")
MODEL_PATH = Path("lstm_jump.pt")
PNL_MODEL_PATH = Path("lstm_jump_pnl.pt")
MODEL_META_PATH = MODEL_PATH.with_suffix(".meta.json")
OUT_DATA = Path("viz_data.npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# defaults, overwritten by meta
SEQ_LEN, PRED_WINDOW = 30, 5
THRESHOLD = 0.8


def load_df(path: Path) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)
    df = pd.DataFrame(list(raw.values()))
    df["datetime"] = pd.to_datetime(df["x"], unit="s")
    return df.set_index("datetime").sort_index()


class EvalDS(Dataset):
    def __init__(self, df: pd.DataFrame, scaler: StandardScaler):
        o = df["o"].astype(np.float32).values
        h = df["h"].astype(np.float32).values
        l = df["l"].astype(np.float32).values
        c = df["c"].astype(np.float32).values
        v = df["v"].astype(np.float32).values.reshape(-1, 1)

        windows = []
        for i in range(SEQ_LEN, len(df) - PRED_WINDOW):
            sl = slice(i - SEQ_LEN + 1, i + 1)
            o_win = o[sl].copy(); h_win = h[sl].copy(); l_win = l[sl].copy(); c_win = c[sl].copy(); v_win = v[sl].copy()

            ref_open = o_win[0]
            prices_rel = np.stack([
                o_win / ref_open - 1.0,
                h_win / ref_open - 1.0,
                l_win / ref_open - 1.0,
                c_win / ref_open - 1.0,
            ], axis=1).astype(np.float32)

            ref_vol = max(float(v_win[0, 0]), 1e-8)
            vol_rel = (v_win / ref_vol - 1.0).astype(np.float32)

            body = np.abs(c_win - o_win)
            eps = np.maximum(1e-8, 1e-6 * np.maximum(o_win, 1.0))
            denom = np.maximum(body, eps)
            upper = h_win - np.maximum(o_win, c_win)
            lower = np.minimum(o_win, c_win) - l_win
            upper_ratio = (upper / denom).astype(np.float32)
            lower_ratio = (lower / denom).astype(np.float32)
            body_sign = np.sign(c_win - o_win).astype(np.float32)
            extra = np.stack([upper_ratio, lower_ratio, body_sign], axis=1)

            window_rel = np.concatenate([prices_rel, vol_rel, extra], axis=1)  # (seq_len, 8)
            windows.append(scaler.transform(window_rel))
        self.samples = windows

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return torch.tensor(self.samples[idx])


class LSTMCls(nn.Module):
    def __init__(self, nfeat: int = 8, hidden: int = 64, layers: int = 2):
        super().__init__(); self.lstm = nn.LSTM(nfeat, hidden, layers, batch_first=True); self.fc = nn.Linear(hidden, 2)
    def forward(self, x): _, (h, _) = self.lstm(x); return self.fc(h[-1])

print("Читаем", EVAL_JSON)
df = load_df(EVAL_JSON)
print(f"Загружено {len(df)} свечей")

print("Загружаем модель", MODEL_PATH)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
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

best_threshold = float(meta.get("threshold", THRESHOLD)) if isinstance(meta, dict) else THRESHOLD

model = LSTMCls(nfeat=8); model.load_state_dict(ckpt["model_state"]); model.to(DEVICE).eval()
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

preds = (probs >= best_threshold).astype(np.int8)

# PnL stats
opens_arr  = df["o"].astype(np.float32).values
closes_arr = df["c"].astype(np.float32).values
entry_opens = opens_arr[SEQ_LEN : len(df) - PRED_WINDOW]
exit_closes = closes_arr[SEQ_LEN + PRED_WINDOW : len(df)]
ret_per_trade = exit_closes / np.maximum(entry_opens, 1e-12) - 1.0
mask = probs >= best_threshold
num_trades = int(mask.sum())
if num_trades > 0:
    r = ret_per_trade[mask]
    sum_pnl = float(np.sum(r))
    comp_ret = -1.0 if np.any(r <= -0.999999) else float(np.exp(np.sum(np.log1p(r))) - 1.0)
    sharpe = float(np.mean(r) / (np.std(r) + 1e-12)) if r.size >= 2 else 0.0
else:
    sum_pnl, comp_ret, sharpe = 0.0, float('-inf'), 0.0

print(f"Сделано {len(preds)} предсказаний")
print(f"Используем порог: {best_threshold:.4f} ({'из meta' if 'threshold' in (meta or {}) else 'по умолчанию'})")
print(f"PnL: trades={num_trades} pnl_sum={sum_pnl*100:.2f}% comp_ret={comp_ret*100 if np.isfinite(comp_ret) else float('nan'):.2f}% sharpe={sharpe:.2f}")

# Compare with PnL-selected checkpoint if exists
if PNL_MODEL_PATH.exists():
    print("\nСравнение с моделью по PnL:")
    ckpt_pnl = torch.load(PNL_MODEL_PATH, map_location=DEVICE, weights_only=False)
    try:
        model_pnl = LSTMCls(nfeat=8); model_pnl.load_state_dict(ckpt_pnl["model_state"]); model_pnl.to(DEVICE).eval()
        scaler_pnl: StandardScaler = ckpt_pnl["scaler"]
        loader_pnl = DataLoader(EvalDS(df, scaler_pnl), batch_size=512)
        probs_pnl = np.zeros(len(loader_pnl.dataset), dtype=np.float32)
        with torch.no_grad():
            ptr = 0
            for xb in loader_pnl:
                outputs = model_pnl(xb.to(DEVICE))
                probs_batch = torch.softmax(outputs, dim=1).cpu().numpy()
                probs_pnl[ptr:ptr+len(probs_batch)] = probs_batch[:, 1]
                ptr += len(probs_batch)
        mask_pnl = probs_pnl >= best_threshold
        num_trades_pnl = int(mask_pnl.sum())
        if num_trades_pnl > 0:
            r2 = ret_per_trade[mask_pnl]
            sum_pnl2 = float(np.sum(r2))
            comp_ret2 = -1.0 if np.any(r2 <= -0.999999) else float(np.exp(np.sum(np.log1p(r2))) - 1.0)
            sharpe2 = float(np.mean(r2) / (np.std(r2) + 1e-12)) if r2.size >= 2 else 0.0
        else:
            sum_pnl2, comp_ret2, sharpe2 = 0.0, float('-inf'), 0.0
        print(f"PnL (lstm_jump_pnl.pt): trades={num_trades_pnl} pnl_sum={sum_pnl2*100:.2f}% comp_ret={comp_ret2*100 if np.isfinite(comp_ret2) else float('nan'):.2f}% sharpe={sharpe2:.2f}")
    except Exception as ex:
        print(f"! Не удалось сравнить с lstm_jump_pnl.pt (возможно другая архитектура): {ex}")

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