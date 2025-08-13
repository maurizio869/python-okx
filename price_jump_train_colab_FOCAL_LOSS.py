# price_jump_train_colab_FOCAL_LOSS.py
# Last modified (MSK): 2025-08-13 23:14
"""Обучение LSTM с Focal Loss (для усиления влияния редкого класса).
Сохраняет лучшую модель по PR AUC и подбирает порог по PnL на валидации.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader, random_split
import math

# ─── ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ ─────────────────────────────────────────────
SEQ_LEN, PRED_WINDOW, JUMP_THRESHOLD = 30, 5, 0.0035  # 30-мин история, окно 5 мин

# Focal Loss параметры (подбираются по валидации)
ALPHA_NEG, ALPHA_POS = 0.25, 0.75
FOCAL_GAMMA = 2.0

# ─── ДАННЫЕ ───────────────────────────────────────────────────────────
def load_dataframe(path: Path) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)
    df = pd.DataFrame(list(raw.values()))
    df["datetime"] = pd.to_datetime(df["x"], unit="s")
    return df.set_index("datetime").sort_index()

class CandleDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.closes = df["c"].astype(np.float32).values
        self.opens  = df["o"].astype(np.float32).values
        price_feats = df[["o", "h", "l", "c"]].astype(np.float32).values
        volumes     = df["v"].astype(np.float32).values.reshape(-1, 1)

        raw_windows = []  # список (seq_len, 5)
        labels      = []

        for i in range(SEQ_LEN, len(df) - PRED_WINDOW):
            current_open = self.opens[i]
            max_close    = self.closes[i + 1 : i + PRED_WINDOW + 1].max()
            jump         = (max_close / current_open - 1) >= JUMP_THRESHOLD
            label        = 1 if jump else 0

            window_raw_prices = price_feats[i - SEQ_LEN + 1 : i + 1].copy()
            ref_open          = window_raw_prices[0, 0]
            window_rel_prices = window_raw_prices / ref_open - 1.0

            window_raw_vol = volumes[i - SEQ_LEN + 1 : i + 1].copy()
            ref_vol        = max(float(window_raw_vol[0, 0]), 1e-8)
            window_rel_vol = window_raw_vol / ref_vol - 1.0

            window_rel = np.concatenate([window_rel_prices, window_rel_vol], axis=1)
            raw_windows.append(window_rel)
            labels.append(label)

        all_rows = np.vstack(raw_windows)  # shape: (n_samples*seq_len, 5)
        self.scaler = StandardScaler().fit(all_rows)
        self.samples = [(self.scaler.transform(w), lbl) for w, lbl in zip(raw_windows, labels)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

# ─── МОДЕЛЬ ────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, nfeat: int = 5, hidden: int = 64, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=nfeat,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# ─── Focal Loss ────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=(ALPHA_NEG, ALPHA_POS), gamma: float = FOCAL_GAMMA, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp().clamp(min=self.eps, max=1.0)
        one_hot = F.one_hot(targets.to(torch.int64), num_classes=logits.size(1)).float()
        p_t = (probs * one_hot).sum(dim=1)
        log_p_t = (log_probs * one_hot).sum(dim=1)
        alpha_t = self.alpha[targets.to(torch.int64)]
        loss = -alpha_t * ((1.0 - p_t) ** self.gamma) * log_p_t
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

# ─── ПАРАМЕТРЫ ОБУЧЕНИЯ ───────────────────────────────────────────────
TRAIN_JSON = Path("candles_10d.json")
MODEL_PATH = Path("lstm_jump.pt")
PNL_MODEL_PATH = Path("lstm_jump_pnl.pt")
MODEL_META_PATH = MODEL_PATH.with_suffix(".meta.json")
VAL_SPLIT, EPOCHS = 0.2, 250
BATCH_SIZE, LR = 512, 4.5e-04
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Загружаем", TRAIN_JSON)
df = load_dataframe(TRAIN_JSON)
print(f"Загружено {len(df)} свечей")

# датасет/сплиты
ds = CandleDataset(df)
print(f"Создано {len(ds)} сэмплов")
pos_cnt = sum(1 for _, y in ds.samples if y == 1)
neg_cnt = len(ds) - pos_cnt
print(f"Меток 1: {pos_cnt}")
print(f"Меток 0: {neg_cnt}")

val = int(len(ds) * VAL_SPLIT)
train_ds, val_ds = random_split(ds, [len(ds) - val, val])
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, BATCH_SIZE)

# Precompute per-trade returns on validation subset for fixed-threshold PnL (@0.565)
val_indices = np.asarray(val_ds.indices, dtype=np.int64)
entry_idx = val_indices + SEQ_LEN
entry_opens = ds.opens[entry_idx]
exit_closes = ds.closes[entry_idx + PRED_WINDOW]
ret_per_trade_val_fixed = exit_closes / np.maximum(entry_opens, 1e-12) - 1.0

# модель/опт/шедулер/лосс
model = LSTMClassifier().to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), LR)
current_patience = 6
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='max', patience=current_patience, factor=0.5, min_lr=1e-6
)
lossf = FocalLoss(alpha=(ALPHA_NEG, ALPHA_POS), gamma=FOCAL_GAMMA)

best_pr_auc = -1.0
best_pnl_sum = -float('inf')
epochs_no_improve = 0

for e in range(1, EPOCHS + 1):
    # train
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        logits = model(xb)
        loss = lossf(logits, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)

    # val
    model.eval()
    corr = tot_s = 0
    val_targets, val_probs, val_preds = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb.to(DEVICE))
            prob1  = torch.softmax(logits, dim=1)[:, 1].cpu()
            pred   = (prob1 >= 0.5).to(torch.long)
            y_cpu  = yb.to(torch.long)

            corr  += (pred.cpu() == y_cpu).sum().item()
            tot_s += y_cpu.size(0)
            val_targets.extend(y_cpu.tolist())
            val_probs.extend(prob1.tolist())
            val_preds.extend(pred.cpu().tolist())

    # metrics
    try:
        roc_auc = roc_auc_score(val_targets, val_probs)
    except Exception:
        roc_auc = float('nan')
    f1 = f1_score(val_targets, val_preds, zero_division=0)
    pr_auc = average_precision_score(val_targets, val_probs)
    p_rate = float(np.mean(val_targets)) if len(val_targets) else 0.0
    npr_auc = (pr_auc - p_rate) / (1.0 - p_rate + 1e-12)

    # PnL with fixed threshold 0.565 on validation
    val_probs_np = np.asarray(val_probs, dtype=np.float32)
    mask_fixed = val_probs_np >= 0.565
    trades_fixed = int(mask_fixed.sum())
    pnl_fixed = float(np.sum(ret_per_trade_val_fixed[mask_fixed])) if trades_fixed > 0 else 0.0

    # lr logging via scheduler
    try:
        curr_lr = scheduler.get_last_lr()[0]
    except Exception:
        curr_lr = opt.param_groups[0]['lr']

    print(
        f"Epoch {e}/{EPOCHS} lr {curr_lr:.2e} "
        f"loss {total_loss/len(train_ds):.4f} val_acc {corr/tot_s:.3f} "
        f"F1 {f1:.3f} ROC_AUC {roc_auc:.3f} PR_AUC {pr_auc:.3f} nPR_AUC {npr_auc:.3f} "
        f"PNL@0.565 {pnl_fixed*100:.2f}% trades={trades_fixed}"
    )

    # step scheduler and dynamically expand patience on LR reduction
    old_lr = opt.param_groups[0]['lr']
    scheduler.step(pr_auc)
    new_lr = opt.param_groups[0]['lr']
    if new_lr < old_lr - 1e-12:
        current_patience = int(math.ceil(current_patience * 1.5))
        scheduler.patience = current_patience
        print(f"LR reduced to {new_lr:.2e}. Next patience set to {current_patience} epochs.")

    # best-save by PR AUC
    if pr_auc > best_pr_auc + 1e-6:
        best_pr_auc = pr_auc
        epochs_no_improve = 0
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "scaler": ds.scaler,
            "meta": {"seq_len": SEQ_LEN, "pred_window": PRED_WINDOW}
        }, MODEL_PATH)
        print(f"✓ Сохранена новая лучшая модель (PR_AUC={best_pr_auc:.3f}) в {MODEL_PATH.resolve()}")

    # best-save by PnL@0.565 (sum returns)
    if pnl_fixed > best_pnl_sum + 1e-12:
        best_pnl_sum = pnl_fixed
        PNL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "scaler": ds.scaler,
            "meta": {"seq_len": SEQ_LEN, "pred_window": PRED_WINDOW}
        }, PNL_MODEL_PATH)
        print(f"✓ Сохранена лучшая по PnL модель (PNL@0.565={best_pnl_sum*100:.2f}%) в {PNL_MODEL_PATH.resolve()}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= 40:
            print(f"⏹ Ранний стоп: PR AUC не улучшается {epochs_no_improve} эпох подряд")
            break

print(f"Лучшая модель с PR_AUC={best_pr_auc:.3f} сохранена в {MODEL_PATH.resolve()}")

# ─── Подбор порога по PnL на валидации ─────────────────────────────
print("Подбираем порог по PnL на валидационном наборе…")
_ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(_ckpt["model_state"])  # перезагружаем лучшие веса
model.to(DEVICE).eval()

val_probs_all = np.zeros(len(val_ds), dtype=np.float32)
with torch.no_grad():
    ptr = 0
    for xb, _yb in DataLoader(val_ds, batch_size=BATCH_SIZE):
        logits = model(xb.to(DEVICE))
        prob1  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        val_probs_all[ptr:ptr+len(prob1)] = prob1
        ptr += len(prob1)

val_indices = np.asarray(val_ds.indices, dtype=np.int64)
entry_idx = val_indices + SEQ_LEN
entry_opens = ds.opens[entry_idx]
exit_closes = ds.closes[entry_idx + PRED_WINDOW]
ret_per_trade_val = exit_closes / np.maximum(entry_opens, 1e-12) - 1.0

thr_min, thr_max, thr_step = 0.43, 0.70, 0.0025
print(f"Перебор порога по PnL (валидация): min={thr_min:.3f}, max={thr_max:.3f}, step={thr_step:.4f}")
thresholds = np.arange(thr_min, thr_max + 1e-12, thr_step)

best_comp_ret = -np.inf
best_threshold_pnl = float(thresholds[0])
best_trades = 0

def _safe_sharpe(r: np.ndarray) -> float:
    if r.size < 2:
        return 0.0
    std = float(np.std(r))
    return float(np.mean(r) / (std + 1e-12))

for t in thresholds:
    mask = (val_probs_all >= t)
    n_trades = int(mask.sum())
    if n_trades == 0:
        comp_ret = -np.inf
        sharpe = 0.0
        sum_ret = 0.0
    else:
        r = ret_per_trade_val[mask]
        if np.any(r <= -0.999999):
            comp_ret = -1.0
        else:
            comp_ret = float(np.exp(np.sum(np.log1p(r))) - 1.0)
        sharpe = _safe_sharpe(r)
        sum_ret = float(np.sum(r))
    print(f"thr={t:.3f} trades={n_trades} pnl={sum_ret*100:.2f}% comp_ret={comp_ret*100 if np.isfinite(comp_ret) else float('nan'):.2f}% sharpe={sharpe:.2f}")
    if comp_ret > best_comp_ret:
        best_comp_ret = comp_ret
        best_threshold_pnl = float(t)
        best_trades = n_trades

print(f"Выбран порог по PnL (валидация): {best_threshold_pnl:.4f}, comp_ret={best_comp_ret*100 if np.isfinite(best_comp_ret) else float('nan'):.2f}% trades={best_trades}")

_final_ckpt = {
    "model_state": model.state_dict(),
    "scaler": ds.scaler,
    "meta": {"seq_len": SEQ_LEN, "pred_window": PRED_WINDOW, "threshold": best_threshold_pnl},
}
torch.save(_final_ckpt, MODEL_PATH)

try:
    with open(MODEL_META_PATH, "w", encoding="utf-8") as mf:
        json.dump({"seq_len": int(SEQ_LEN), "pred_window": int(PRED_WINDOW), "threshold": float(best_threshold_pnl)}, mf)
except Exception as ex:
    print(f"! Не удалось записать meta-файл {MODEL_META_PATH}: {ex}")