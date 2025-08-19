# price_jump_train_colab_FOCAL_LOSS.py
# Last modified (MSK): 2025-08-19 15:05
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

# Hoisted constants
REDUCE_ON_PLATEAU_START_LR = 5e-4
REDUCE_ON_PLATEAU_START_PATIENCE = 9
REDUCE_ON_PLATEAU_FACTOR = 1/3
REDUCE_ON_PLATEAU_MIN_LR = 1e-5
PNL_FIXED_THRESHOLD = 0.565

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
HYPER_PATH = MODEL_PATH.with_suffix(".hyper.json")
VAL_SPLIT, EPOCHS = 0.2, 250
BATCH_SIZE, LR = 512, 6e-4
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

val = int(len(ds)*VAL_SPLIT)
train_ds,val_ds = random_split(ds,[len(ds)-val,val])
tl = DataLoader(train_ds,BATCH_SIZE,shuffle=True)
vl = DataLoader(val_ds,BATCH_SIZE)

# Optional overrides from meta
DROPOUT_P = 0.3
_got_dropout = False
_got_base_lr = False
_src_dropout = "default"
_src_base_lr = "default"
try:
	if HYPER_PATH.exists():
		with open(HYPER_PATH, 'r', encoding='utf-8') as hf:
			hyper = json.load(hf)
		if isinstance(hyper, dict):
			if 'dropout' in hyper:
				DROPOUT_P = float(hyper['dropout']); _got_dropout = True; _src_dropout = f"{HYPER_PATH}"
			if 'base_lr' in hyper:
				LR = float(hyper['base_lr']); _got_base_lr = True; _src_base_lr = f"{HYPER_PATH}"
	elif MODEL_META_PATH.exists():
		with open(MODEL_META_PATH, 'r', encoding='utf-8') as mf:
			meta0 = json.load(mf)
		if isinstance(meta0, dict):
			if 'dropout' in meta0:
				DROPOUT_P = float(meta0['dropout']); _got_dropout = True; _src_dropout = f"{MODEL_META_PATH}"
			if 'base_lr' in meta0:
				LR = float(meta0['base_lr']); _got_base_lr = True; _src_base_lr = f"{MODEL_META_PATH}"
except Exception as ex:
	print(f"! Не удалось прочитать hyper/meta для dropout/base_lr: {ex}")

if _got_dropout:
	print(f"dropout прочитан из {_src_dropout}: {DROPOUT_P:.3f}")
else:
	print(f"dropout взят по умолчанию: {DROPOUT_P:.3f}")
if _got_base_lr:
	print(f"base_lr прочитан из {_src_base_lr}: {LR:.2e}")
else:
	print(f"base_lr взят по умолчанию: {LR:.2e}")

# Precompute per-trade returns on validation subset for fixed-threshold PnL (@0.565)
val_indices = np.asarray(val_ds.indices, dtype=np.int64)
entry_idx = val_indices + SEQ_LEN
entry_opens = ds.opens[entry_idx]
exit_closes = ds.closes[entry_idx + PRED_WINDOW]
ret_per_trade_val_fixed = exit_closes / np.maximum(entry_opens, 1e-12) - 1.0

# модель/опт/шедулер/лосс
model = LSTMClassifier(dropout=DROPOUT_P).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), LR)
current_patience = REDUCE_ON_PLATEAU_START_PATIENCE
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
	opt, mode='max', patience=current_patience, factor=REDUCE_ON_PLATEAU_FACTOR, min_lr=REDUCE_ON_PLATEAU_MIN_LR
)
lossf = FocalLoss(alpha=(ALPHA_NEG, ALPHA_POS), gamma=FOCAL_GAMMA)

best_pr_auc = -1.0
best_pnl_sum = -float('inf')
best_pnl_thr = 0.565
epochs_no_improve = 0
# Collect per-epoch curves for post-training plot
lr_curve = []
pr_auc_curve = []
pnl_curve_pct = []
val_acc_curve = []

for e in range(1, EPOCHS + 1):
    # train
    model.train()
    total_loss = 0.0
    for xb, yb in tl:
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
        for xb, yb in vl:
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
    mask_fixed = val_probs_np >= PNL_FIXED_THRESHOLD
    trades_fixed = int(mask_fixed.sum())
    pnl_fixed = float(np.sum(ret_per_trade_val_fixed[mask_fixed])) if trades_fixed > 0 else 0.0

    # lr logging via scheduler
    try:
        curr_lr = scheduler.get_last_lr()[0]
    except Exception:
        curr_lr = opt.param_groups[0]['lr']

    val_acc = (corr/tot_s) if tot_s > 0 else 0.0
    print(
        f"Epoch {e}/{EPOCHS} lr {curr_lr:.2e} "
        f"loss {total_loss/len(train_ds):.4f} val_acc {val_acc:.3f} "
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

    # collect curves (use curr_lr of this epoch)
    lr_curve.append(float(curr_lr))
    pr_auc_curve.append(float(pr_auc))
    pnl_curve_pct.append(float(pnl_fixed*100.0))
    val_acc_curve.append(float(val_acc))

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
        best_pnl_thr = 0.565
        PNL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "scaler": ds.scaler,
            "meta": {"seq_len": SEQ_LEN, "pred_window": PRED_WINDOW}
        }, PNL_MODEL_PATH)
        print(f"✓ Сохранена новая лучшая модель (PNL@{best_pnl_thr:.4f}={best_pnl_sum*100:.2f}%) в {PNL_MODEL_PATH.resolve()}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= 25:
            print(f"⏹ Ранний стоп: PR AUC не улучшается {epochs_no_improve} эпох подряд")
            break

print(f"Лучшая модель с PR_AUC={best_pr_auc:.3f} сохранена в {MODEL_PATH.resolve()}")
print(f"Лучшая модель с pnl@{best_pnl_thr:.4f}={best_pnl_sum*100:.2f}% сохранена в {PNL_MODEL_PATH.resolve()}")

# Post-training curves (normalized)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    curves = {
        'LR': np.asarray(lr_curve, dtype=np.float64),
        'PR_AUC': np.asarray(pr_auc_curve, dtype=np.float64),
        'PnL%': np.asarray(pnl_curve_pct, dtype=np.float64),
        'ValAcc': np.asarray(val_acc_curve, dtype=np.float64),
    }
    eps = 1e-12
    plt.figure(figsize=(8,5))
    x = np.arange(1, len(lr_curve)+1)
    colors = {}
    for name, arr in curves.items():
        if arr.size == 0:
            continue
        arr_norm = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + eps)
        line, = plt.plot(x[:len(arr_norm)], arr_norm, label=name)
        colors[name] = line.get_color()
    # annotate max PR_AUC and max PnL%
    if len(pr_auc_curve) > 0:
        i_best_pr = int(np.nanargmax(pr_auc_curve))
        y_best_pr = (pr_auc_curve[i_best_pr] - np.nanmin(pr_auc_curve)) / (np.nanmax(pr_auc_curve) - np.nanmin(pr_auc_curve) + eps)
        plt.scatter([i_best_pr+1], [y_best_pr], color=colors.get('PR_AUC', '#2ca02c'), s=40)
        plt.annotate(f"max PR_AUC={pr_auc_curve[i_best_pr]:.3f}\n(ep={i_best_pr+1})",
                     xy=(i_best_pr+1, y_best_pr), xytext=(5, 12), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))
    if len(pnl_curve_pct) > 0:
        i_best_pnl = int(np.nanargmax(pnl_curve_pct))
        y_best_pnl = (pnl_curve_pct[i_best_pnl] - np.nanmin(pnl_curve_pct)) / (np.nanmax(pnl_curve_pct) - np.nanmin(pnl_curve_pct) + eps)
        plt.scatter([i_best_pnl+1], [y_best_pnl], color=colors.get('PnL%', '#d62728'), s=40)
        plt.annotate(f"max PnL={pnl_curve_pct[i_best_pnl]:.2f}%\n(ep={i_best_pnl+1})",
                     xy=(i_best_pnl+1, y_best_pnl), xytext=(5, -28), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))
    const_text = (
        f"VAL_SPLIT={VAL_SPLIT}\nEPOCHS={EPOCHS}\nBATCH={BATCH_SIZE}\nLR0={REDUCE_ON_PLATEAU_START_LR:.2e}\n"
        f"patience0={REDUCE_ON_PLATEAU_START_PATIENCE}\nfactor={REDUCE_ON_PLATEAU_FACTOR}\nmin_lr={REDUCE_ON_PLATEAU_MIN_LR:.1e}\n"
        f"PNL_thr={PNL_FIXED_THRESHOLD}\nLOSS=Focal(gamma={FOCAL_GAMMA}, alpha_neg={ALPHA_NEG}, alpha_pos={ALPHA_POS})"
    )
    plt.gca().text(0.98, 0.02, const_text, transform=plt.gca().transAxes,
                   ha='right', va='bottom', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    plt.xlabel('Epoch'); plt.ylabel('Normalized scale [0,1]')
    plt.title('Training curves (normalized)')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    from datetime import datetime
    import pytz
    msk = pytz.timezone('Europe/Moscow')
    ts = datetime.now(msk).strftime('%Y%m%d_%H%M')
    out_name = f'training_curves_reduce_on_plateau_{ts}.png'
    plt.savefig(out_name, dpi=120)
    print(f"Saved post-training curves to {Path(out_name).resolve()}")
    try:
        from IPython.display import Image, display
        display(Image(out_name))
    except Exception:
        pass
    plt.close()
except Exception as ex:
    print(f"! Не удалось построить/сохранить пост-обучающие кривые: {ex}")

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

thr_list=[]; pnl_list=[]; comp_list=[]; sharpe_list=[]

def _safe_sharpe_arr(r: np.ndarray) -> float:
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
        sharpe = _safe_sharpe_arr(r)
        sum_ret = float(np.sum(r))
    thr_list.append(float(t)); pnl_list.append(sum_ret*100.0); comp_list.append(comp_ret*100.0 if np.isfinite(comp_ret) else np.nan); sharpe_list.append(sharpe)
    if comp_ret > best_comp_ret:
        best_comp_ret = comp_ret
        best_threshold_pnl = float(t)
        best_trades = n_trades

print(f"Выбран порог по PnL (валидация): {best_threshold_pnl:.4f}, comp_ret={best_comp_ret*100 if np.isfinite(best_comp_ret) else float('nan'):.2f}% trades={best_trades}")

# график метрик vs threshold
try:
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax2 = ax1.twinx()
    thr_arr = np.asarray(thr_list)
    pnl_arr = np.asarray(pnl_list)
    comp_arr = np.asarray(comp_list)
    shp_arr = np.asarray(sharpe_list)
    l1, = ax1.plot(thr_arr, comp_arr, label='comp_ret %', color='#1f77b4')
    l2, = ax1.plot(thr_arr, pnl_arr, label='pnl_sum %', color='#ff7f0e')
    l3, = ax2.plot(thr_arr, shp_arr, label='sharpe', color='#2ca02c', alpha=0.8)
    if np.isfinite(best_comp_ret):
        idx = int(np.nanargmax(comp_arr))
        ax1.axvline(thr_arr[idx], color=l1.get_color(), linestyle='--', alpha=0.6)
        ax1.scatter([thr_arr[idx]],[comp_arr[idx]], color=l1.get_color(), s=35)
        ax1.annotate(f"best comp={comp_arr[idx]:.2f}%\nthr={thr_arr[idx]:.4f}",
                     xy=(thr_arr[idx], comp_arr[idx]), xytext=(10, 12), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('% metrics (comp_ret, pnl_sum)')
    ax2.set_ylabel('Sharpe')
    lines = [l1,l2,l3]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc='best')
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    from datetime import datetime
    import pytz
    msk = pytz.timezone('Europe/Moscow')
    ts = datetime.now(msk).strftime('%Y%m%d_%H%M')
    out_name = f'threshold_sweep_{ts}.png'
    fig.savefig(out_name, dpi=130)
    print(f"Saved threshold sweep plot to {Path(out_name).resolve()}")
    try:
        from IPython.display import Image, display
        display(Image(out_name))
    except Exception:
        pass
    plt.close(fig)
except Exception as ex:
    print(f"! Не удалось построить график перебора порога: {ex}")

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