# price_jump_train_colab.py
# Last modified (MSK): 2025-08-13 14:53
"""Обучает LSTM, метка = 1 если
   • максимум Close за следующие 5 мин ≥ Open + 0.35%
Сохраняет модель и StandardScaler в lstm_jump.pt
"""
from pathlib import Path
import json, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader, random_split

SEQ_LEN, PRED_WINDOW, JUMP_THRESHOLD = 30, 5, 0.0035  # 30-мин история, окно 5 мин

def load_dataframe(path: Path) -> pd.DataFrame:
    with open(path) as f: raw = json.load(f)
    df = pd.DataFrame(list(raw.values()))
    df["datetime"] = pd.to_datetime(df["x"], unit="s")
    return df.set_index("datetime").sort_index()

class CandleDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.closes = df["c"].astype(np.float32).values
        self.opens  = df["o"].astype(np.float32).values
        # price features and volume as separate arrays
        price_feats = df[["o", "h", "l", "c"]].astype(np.float32).values
        volumes     = df["v"].astype(np.float32).values.reshape(-1, 1)

        # Сохраняем необработанные относительные окна, чтобы потом подогнать StandardScaler
        raw_windows = []      # список (seq_len, 5)
        labels      = []

        for i in range(SEQ_LEN, len(df) - PRED_WINDOW):
            current_open = self.opens[i]
            max_close    = self.closes[i + 1 : i + PRED_WINDOW + 1].max()
            jump         = (max_close / current_open - 1) >= JUMP_THRESHOLD
            label        = 1 if jump else 0

            # относительные признаки по ценам — к Open первой свечи окна
            window_raw_prices = price_feats[i - SEQ_LEN + 1 : i + 1].copy()
            ref_open          = window_raw_prices[0, 0]
            window_rel_prices = window_raw_prices / ref_open - 1.0

            # относительные признаки по объёму — к объёму первой свечи окна
            window_raw_vol = volumes[i - SEQ_LEN + 1 : i + 1].copy()   # (seq_len, 1)
            ref_vol        = max(float(window_raw_vol[0, 0]), 1e-8)
            window_rel_vol = window_raw_vol / ref_vol - 1.0

            # объединяем 4 ценовых + 1 объёмной канал
            window_rel = np.concatenate([window_rel_prices, window_rel_vol], axis=1)

            raw_windows.append(window_rel)
            labels.append(label)

        # Фитируем scaler на всех относительных значениях
        all_rows = np.vstack(raw_windows)                 # shape: (n_samples*seq_len, 5)
        self.scaler = StandardScaler().fit(all_rows)

        # Трансформируем и сохраняем финальные выборки
        self.samples = [(self.scaler.transform(w), lbl) for w, lbl in zip(raw_windows, labels)]

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x,y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

class LSTMClassifier(nn.Module):
    def __init__(self, nfeat: int = 5, hidden: int = 64, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        # dropout активен только если layers > 1
        self.lstm = nn.LSTM(nfeat, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 2)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# ─── параметры обучения ───────────────────────────────────────────
TRAIN_JSON = Path("candles_10d.json")
MODEL_PATH = Path("lstm_jump.pt")
MODEL_META_PATH = MODEL_PATH.with_suffix(".meta.json")
VAL_SPLIT, EPOCHS = 0.2, 250
BATCH_SIZE, LR = 512, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Загружаем", TRAIN_JSON)

df = load_dataframe(TRAIN_JSON)
print(f"Загружено {len(df)} свечей")

ds = CandleDataset(df)
print(f"Создано {len(ds)} сэмплов")
pos_cnt = sum(1 for _, y in ds.samples if y == 1)
neg_cnt = len(ds) - pos_cnt
print(f"Меток 1: {pos_cnt}")
print(f"Меток 0: {neg_cnt}")

# Взвешивание классов для компенсации дисбаланса
pos_weight = neg_cnt / max(pos_cnt, 1)
class_weights = torch.tensor([1.0, pos_weight], device=DEVICE)

val = int(len(ds)*VAL_SPLIT)
train_ds,val_ds = random_split(ds,[len(ds)-val,val])
tl = DataLoader(train_ds,BATCH_SIZE,shuffle=True)
vl = DataLoader(val_ds,BATCH_SIZE)

# Precompute per-trade returns on validation subset for fixed-threshold PnL (@0.565)
val_indices = np.asarray(val_ds.indices, dtype=np.int64)
entry_idx = val_indices + SEQ_LEN
entry_opens = ds.opens[entry_idx]
exit_closes = ds.closes[entry_idx + PRED_WINDOW]
ret_per_trade_val_fixed = exit_closes / np.maximum(entry_opens, 1e-12) - 1.0

model = LSTMClassifier().to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='max', patience=10, factor=0.1, min_lr=1e-6
)
lossf = nn.CrossEntropyLoss(weight=class_weights)

best_pr_auc = -1.0
epochs_no_improve = 0
for e in range(1, EPOCHS+1):
    model.train(); tot=0
    for x,y in tl:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); loss=lossf(model(x),y); loss.backward(); opt.step()
        tot += loss.item()*x.size(0)
    
    # validation: collect preds, probs for metrics
    model.eval(); corr=tot_s=0
    val_targets = []
    val_probs   = []
    val_preds   = []
    with torch.no_grad():
        for x,y in vl:
            logits = model(x.to(DEVICE))
            prob1  = torch.softmax(logits, dim=1)[:,1].cpu()
            pred   = (prob1 >= 0.5).to(torch.long)
            y_cpu  = y.to(torch.long)
            
            corr  += (pred.cpu() == y_cpu).sum().item(); tot_s += y_cpu.size(0)
            val_targets.extend(y_cpu.tolist())
            val_probs.extend(prob1.tolist())
            val_preds.extend(pred.cpu().tolist())
    
    # compute metrics
    try:
        roc_auc = roc_auc_score(val_targets, val_probs)
    except Exception:
        roc_auc = float('nan')
    f1 = f1_score(val_targets, val_preds, zero_division=0)
    pr_auc = average_precision_score(val_targets, val_probs)
    
    # PnL with fixed threshold 0.565 on validation
    val_probs_np = np.asarray(val_probs, dtype=np.float32)
    mask_fixed = val_probs_np >= 0.565
    trades_fixed = int(mask_fixed.sum())
    pnl_fixed = float(np.sum(ret_per_trade_val_fixed[mask_fixed])) if trades_fixed > 0 else 0.0
    
    # prefer scheduler.get_last_lr when available
    try:
        curr_lr = scheduler.get_last_lr()[0]
    except Exception:
        curr_lr = opt.param_groups[0]['lr']
    print(f'Epoch {e}/{EPOCHS} lr {curr_lr:.2e} loss {tot/len(train_ds):.4f} '
          f'val_acc {corr/tot_s:.3f} F1 {f1:.3f} ROC_AUC {roc_auc:.3f} PR_AUC {pr_auc:.3f} '
          f'PNL@0.565 {pnl_fixed*100:.2f}% trades={trades_fixed}')

    # save best model by PR AUC
    if pr_auc > best_pr_auc + 1e-6:
        best_pr_auc = pr_auc
        epochs_no_improve = 0
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "scaler": ds.scaler,
            "meta": {"seq_len": SEQ_LEN, "pred_window": PRED_WINDOW}
        }, MODEL_PATH)
        # write side meta file as well
        try:
            with open(MODEL_META_PATH, "w", encoding="utf-8") as mf:
                json.dump({"seq_len": int(SEQ_LEN), "pred_window": int(PRED_WINDOW)}, mf)
        except Exception as ex:
            print(f"! Не удалось записать meta-файл {MODEL_META_PATH}: {ex}")
        print(f"✓ Сохранена новая лучшая модель (PR_AUC={best_pr_auc:.3f}) в {MODEL_PATH.resolve()}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= 40:
            print(f"⏹ Ранний стоп: PR AUC не улучшается {epochs_no_improve} эпох подряд")
            break
    
    scheduler.step(pr_auc)

print(f"Лучшая модель с PR_AUC={best_pr_auc:.3f} сохранена в {MODEL_PATH.resolve()}")

# ─── Подбор порога по PnL на валидации ─────────────────────────────
print("Подбираем порог по PnL на валидационном наборе…")
# загрузим лучшую модель с диска на всякий случай
_ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(_ckpt["model_state"])  # перезагружаем лучшие веса
model.to(DEVICE).eval()

# собираем вероятности на валидации
val_probs_all = np.zeros(len(val_ds), dtype=np.float32)
with torch.no_grad():
    ptr = 0
    for xb, _yb in DataLoader(val_ds, batch_size=BATCH_SIZE):
        logits = model(xb.to(DEVICE))
        prob1  = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        val_probs_all[ptr:ptr+len(prob1)] = prob1
        ptr += len(prob1)

# считаем доходности сделок для индексов валидации
val_indices = np.asarray(val_ds.indices, dtype=np.int64)
entry_idx = val_indices + SEQ_LEN
entry_opens = ds.opens[entry_idx]
exit_closes = ds.closes[entry_idx + PRED_WINDOW]
ret_per_trade_val = exit_closes / np.maximum(entry_opens, 1e-12) - 1.0

thr_min, thr_max, thr_step = 0.45, 0.70, 0.0025
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

# дописываем threshold в чекпойнт и meta JSON
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