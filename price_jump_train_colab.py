# price_jump_train_colab.py
# Last modified (MSK): 2025-08-25 15:40
"""Обучает LSTM, метка = 1 если
   • максимум Close за следующие 5 мин ≥ Open + 0.35%
 Сохраняет модель и StandardScaler в lstm_jump.pt
"""
from pathlib import Path
import json, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader, random_split
import math
import matplotlib.pyplot as plt
import time

SEQ_LEN, PRED_WINDOW, JUMP_THRESHOLD = 30, 5, 0.0035  # 30-мин история, окно 5 мин

# Scheduler and PnL constants (hoisted)
REDUCE_ON_PLATEAU_START_LR = 4e-4
REDUCE_ON_PLATEAU_START_PATIENCE = 7
REDUCE_ON_PLATEAU_FACTOR = 1/1.7
REDUCE_ON_PLATEAU_MIN_LR = 1e-5
PNL_FIXED_THRESHOLD = 0.565
EARLY_STOP_EPOCHS = 40
# Model/training constants
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
DEFAULT_DROPOUT = 0.35
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Data preprocessing epsilons
REF_VOL_EPS = 1e-8
MIN_DENOM_EPS = 1e-12
PLOT_NORM_EPS = 1e-12           # eps при нормализации кривых на графике
LR_CHANGE_EPS = 1e-12           # eps для сравнения изменения LR
BLACK_SWAN_LIMIT = -0.999999    # защита от краха при комп. доходности
NPR_EPS = 1e-12                 # eps для нормализации PR AUC
# Threshold sweep defaults (post-training)
THR_SWEEP_MIN = 0.43
THR_SWEEP_MAX = 0.70
THR_SWEEP_STEP = 0.0025
# Training session hyperparams
VAL_SPLIT = 0.2
EPOCHS = 450
BATCH_SIZE = 128
BASE_LR_DEFAULT = REDUCE_ON_PLATEAU_START_LR
PRED_THRESHOLD = 0.5
PATIENCE_GROWTH = 1.5
IMPROVE_EPS = 1e-6
COMP_EPS = 1e-12
SHARPE_MIN_SAMPLES = 2


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
            ref_vol        = max(float(window_raw_vol[0, 0]), REF_VOL_EPS)
            window_rel_vol = window_raw_vol / ref_vol - 1.0

            # объединяем 4 ценовых + 1 объёмной канал
            window_rel = np.concatenate([window_rel_prices, window_rel_vol], axis=1)

            raw_windows.append(window_rel)
            labels.append(label)

        all_rows = np.vstack(raw_windows)                 # shape: (n_samples*seq_len, 5)
        self.scaler = StandardScaler().fit(all_rows)

        self.samples = [(self.scaler.transform(w), lbl) for w, lbl in zip(raw_windows, labels)]

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x,y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


class LSTMClassifier(nn.Module):
    def __init__(self, nfeat: int = 5, hidden: int = LSTM_HIDDEN, layers: int = LSTM_LAYERS, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(nfeat, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 2)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


# ─── параметры обучения ───────────────────────────────────────────
TRAIN_JSON = Path("candles_10d.json")
MODEL_PATH = Path("lstm_jump.pt")
PNL_MODEL_PATH = Path("lstm_jump_pnl.pt")
MODEL_META_PATH = MODEL_PATH.with_suffix(".meta.json")
HYPER_PATH = MODEL_PATH.with_suffix(".hyper.json")

print(f"Device: {DEVICE}")
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
ret_per_trade_val_fixed = exit_closes / np.maximum(entry_opens, MIN_DENOM_EPS) - 1.0

# Optional overrides from meta/hyper
DROPOUT_P = DEFAULT_DROPOUT
LR = BASE_LR_DEFAULT
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

model = LSTMClassifier(hidden=LSTM_HIDDEN, layers=LSTM_LAYERS, dropout=DROPOUT_P).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), LR)
current_patience = REDUCE_ON_PLATEAU_START_PATIENCE
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='max', patience=current_patience, factor=REDUCE_ON_PLATEAU_FACTOR, min_lr=REDUCE_ON_PLATEAU_MIN_LR
)
lossf = nn.CrossEntropyLoss(weight=class_weights)

best_pr_auc = -1.0
best_pnl_sum = -float('inf')
best_pnl_thr = PNL_FIXED_THRESHOLD
epochs_no_improve = 0
# Collect per-epoch curves for post-training plot
lr_curve = []
pr_auc_curve = []
pnl_curve_pct = []
val_acc_curve = []
for e in range(1, EPOCHS+1):
	_t0 = time.time()
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
			pred   = (prob1 >= PRED_THRESHOLD).to(torch.long)
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
	# normalized PR AUC relative to positive rate p
	p_pos = (sum(val_targets)/max(len(val_targets),1)) if len(val_targets)>0 else 0.0
	npr_auc = (pr_auc - p_pos) / (1.0 - p_pos + COMP_EPS)
	# fixed threshold PnL
	val_probs_np = np.asarray(val_probs, dtype=np.float32)
	mask_fixed = val_probs_np >= PNL_FIXED_THRESHOLD
	trades_fixed = int(mask_fixed.sum())
	pnl_fixed = float(np.sum(ret_per_trade_val_fixed[mask_fixed])) if trades_fixed > 0 else 0.0
	# prefer scheduler.get_last_lr when available
	try:
		curr_lr = scheduler.get_last_lr()[0]
	except Exception:
		curr_lr = opt.param_groups[0]['lr']
	val_acc = (corr/tot_s) if tot_s > 0 else 0.0
	_dt = time.time() - _t0
	print(f'Epoch {e}/{EPOCHS} lr {curr_lr:.2e} loss {tot/len(train_ds):.4f} '
	      f'val_acc {val_acc:.3f} F1 {f1:.3f} ROC_AUC {roc_auc:.3f} PR_AUC {pr_auc:.3f} nPR_AUC {npr_auc:.3f} '
	      f'PNL@{PNL_FIXED_THRESHOLD} {pnl_fixed*100:.2f}% trades={trades_fixed} time {(_dt):.1f}s')
	# step scheduler and dynamically expand patience on LR reduction
	old_lr = opt.param_groups[0]['lr']
	scheduler.step(pr_auc)
	new_lr = opt.param_groups[0]['lr']
	if new_lr < old_lr - LR_CHANGE_EPS:
		current_patience = int(math.ceil(current_patience * PATIENCE_GROWTH))
		scheduler.patience = current_patience
		print(f"LR reduced to {new_lr:.2e}. Next patience set to {current_patience} epochs.")

	# collect curves (use curr_lr used in this epoch for plotting)
	lr_curve.append(float(curr_lr))
	pr_auc_curve.append(float(pr_auc))
	pnl_curve_pct.append(float(pnl_fixed*100.0))
	val_acc_curve.append(float(val_acc))

	# save best model by PR AUC
	if pr_auc > best_pr_auc + IMPROVE_EPS:
		best_pr_auc = pr_auc
		epochs_no_improve = 0
		MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
		torch.save({"model_state": model.state_dict(), "scaler": ds.scaler, "meta": {"seq_len": SEQ_LEN, "pred_window": PRED_WINDOW}}, MODEL_PATH)
		print(f"✓ Сохранена новая лучшая модель (PR_AUC={best_pr_auc:.3f}) в {MODEL_PATH.resolve()}")
	
	# save best-by-PnL model (using PNL@0.565 sum of returns)
	if pnl_fixed > best_pnl_sum + COMP_EPS:
		best_pnl_sum = pnl_fixed
		best_pnl_thr = PNL_FIXED_THRESHOLD
		PNL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
		torch.save({"model_state": model.state_dict(), "scaler": ds.scaler, "meta": {"seq_len": SEQ_LEN, "pred_window": PRED_WINDOW}}, PNL_MODEL_PATH)
		print(f"✓ Сохранена новая лучшая модель (PNL@{best_pnl_thr:.4f}={best_pnl_sum*100:.2f}%) в {PNL_MODEL_PATH.resolve()}")
	else:
		epochs_no_improve += 1
		if epochs_no_improve >= EARLY_STOP_EPOCHS:
			print(f"⏹ Ранний стоп: PR AUC не улучшается {epochs_no_improve} эпох подряд")
			break

print(f"Лучшая модель с PR_AUC={best_pr_auc:.3f} сохранена в {MODEL_PATH.resolve()}")
print(f"Лучшая модель с pnl@{best_pnl_thr:.4f}={best_pnl_sum*100:.2f}% сохранена в {PNL_MODEL_PATH.resolve()}")

# Post-training curves (normalized): LR, PR_AUC, PnL%@thr, ValAcc
try:
    curves = {
        'LR': np.asarray(lr_curve, dtype=np.float64),
        'PR_AUC': np.asarray(pr_auc_curve, dtype=np.float64),
        'PnL%': np.asarray(pnl_curve_pct, dtype=np.float64),
        'ValAcc': np.asarray(val_acc_curve, dtype=np.float64),
    }
    plt.figure(figsize=(8,5))
    x = np.arange(1, len(lr_curve)+1)
    colors = {}
    for name, arr in curves.items():
        if arr.size == 0:
            continue
        arr_norm = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + PLOT_NORM_EPS)
        line, = plt.plot(x[:len(arr_norm)], arr_norm, label=name)
        colors[name] = line.get_color()
    # annotate max PR_AUC and max PnL%
    i_best_pr = None
    if len(pr_auc_curve) > 0:
        i_best_pr = int(np.nanargmax(pr_auc_curve))
        y_best_pr = (pr_auc_curve[i_best_pr] - np.nanmin(pr_auc_curve)) / (np.nanmax(pr_auc_curve) - np.nanmin(pr_auc_curve) + PLOT_NORM_EPS)
        plt.scatter([i_best_pr+1], [y_best_pr], color=colors.get('PR_AUC', '#2ca02c'), s=40)
        plt.annotate(f"max PR_AUC={pr_auc_curve[i_best_pr]:.3f}\n(ep={i_best_pr+1})",
                     xy=(i_best_pr+1, y_best_pr), xytext=(8, 18), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))
    if len(pnl_curve_pct) > 0:
        i_best_pnl = int(np.nanargmax(pnl_curve_pct))
        y_best_pnl = (pnl_curve_pct[i_best_pnl] - np.nanmin(pnl_curve_pct)) / (np.nanmax(pnl_curve_pct) - np.nanmin(pnl_curve_pct) + PLOT_NORM_EPS)
        plt.scatter([i_best_pnl+1], [y_best_pnl], color=colors.get('PnL%', '#d62728'), s=40)
        dy = 36 if (i_best_pr is not None and i_best_pnl == i_best_pr) else 18
        plt.annotate(f"max PnL={pnl_curve_pct[i_best_pnl]:.2f}%\n(ep={i_best_pnl+1})",
                     xy=(i_best_pnl+1, y_best_pnl), xytext=(12, dy), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))
    const_text = (
        f"VAL_SPLIT={VAL_SPLIT}\nEPOCHS={EPOCHS}\nBATCH={BATCH_SIZE}\nLR0={REDUCE_ON_PLATEAU_START_LR:.2e}\n"
        f"patience0={REDUCE_ON_PLATEAU_START_PATIENCE}\nfactor={REDUCE_ON_PLATEAU_FACTOR}\nmin_lr={REDUCE_ON_PLATEAU_MIN_LR:.1e}\n"
        f"PNL_thr={PNL_FIXED_THRESHOLD}\nDROPOUT={DROPOUT_P:.3f}"
    )
    plt.gca().text(0.98, 0.02, const_text, transform=plt.gca().transAxes,
                   ha='right', va='bottom', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    # script filename at bottom-left
    try:
        _script_name = Path(__file__).name
    except Exception:
        _script_name = "price_jump_train_colab.py"
    plt.gca().text(0.02, 0.02, _script_name, transform=plt.gca().transAxes,
                   ha='left', va='bottom', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5))
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
model.load_state_dict(_ckpt["model_state"])
model.to(DEVICE).eval()

val_probs_all = np.zeros(len(val_ds), dtype=np.float32)
with torch.no_grad():
    ptr = 0
    for xb, _yb in DataLoader(val_ds, batch_size=BATCH_SIZE):
        logits = model(xb.to(DEVICE))
        prob1  = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        val_probs_all[ptr:ptr+len(prob1)] = prob1
        ptr += len(prob1)

val_indices = np.asarray(val_ds.indices, dtype=np.int64)
entry_idx = val_indices + SEQ_LEN
entry_opens = ds.opens[entry_idx]
exit_closes = ds.closes[entry_idx + PRED_WINDOW]
ret_per_trade_val = exit_closes / np.maximum(entry_opens, MIN_DENOM_EPS) - 1.0

print(f"Перебор порога по PnL (валидация): min={THR_SWEEP_MIN:.3f}, max={THR_SWEEP_MAX:.3f}, step={THR_SWEEP_STEP:.4f}")
thresholds = np.arange(THR_SWEEP_MIN, THR_SWEEP_MAX + 1e-12, THR_SWEEP_STEP)

best_comp_ret = -np.inf
best_threshold_pnl = float(thresholds[0])
best_trades = 0

thr_list=[]; pnl_list=[]; comp_list=[]; sharpe_list=[]
mean_ret_list=[]; median_ret_list=[]; mdd_list=[]

for t in thresholds:
    mask = (val_probs_all >= t)
    n_trades = int(mask.sum())
    if n_trades == 0:
        comp_ret = -np.inf
        sharpe = 0.0
        sum_ret = 0.0
        mean_ret = 0.0
        median_ret = 0.0
        mdd_pct = 0.0
    else:
        r = ret_per_trade_val[mask]
        if np.any(r <= BLACK_SWAN_LIMIT):
            comp_ret = -1.0
        else:
            comp_ret = float(np.exp(np.sum(np.log1p(r))) - 1.0)
        sharpe = float(np.mean(r) / (np.std(r) + PLOT_NORM_EPS)) if r.size >= SHARPE_MIN_SAMPLES else 0.0
        sum_ret = float(np.sum(r))
        # mean/median returns in %
        mean_ret = float(np.mean(r) * 100.0)
        median_ret = float(np.median(r) * 100.0)
        # max drawdown (absolute positive %) computed on chronological equity curve
        ent = entry_idx[mask]
        order = np.argsort(ent)
        r_sorted = r[order]
        equity = np.cumprod(1.0 + r_sorted.astype(np.float64))
        run_max = np.maximum.accumulate(equity)
        dd = np.min(equity / (run_max + COMP_EPS) - 1.0) if equity.size > 0 else 0.0
        mdd_pct = float(abs(dd) * 100.0)
    thr_list.append(float(t)); pnl_list.append(sum_ret*100.0); comp_list.append(comp_ret*100.0 if np.isfinite(comp_ret) else np.nan); sharpe_list.append(sharpe); mean_ret_list.append(mean_ret); median_ret_list.append(median_ret); mdd_list.append(mdd_pct)
    if comp_ret > best_comp_ret:
        best_comp_ret = comp_ret
        best_threshold_pnl = float(t)
        best_trades = n_trades

print(f"Выбран порог по PnL (валидация): {best_threshold_pnl:.4f}, comp_ret={best_comp_ret*100 if np.isfinite(best_comp_ret) else float('nan'):.2f}% trades={best_trades}")

try:
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax2 = ax1.twinx()
    thr_arr = np.asarray(thr_list)
    pnl_arr = np.asarray(pnl_list)
    comp_arr = np.asarray(comp_list)
    shp_arr = np.asarray(sharpe_list)
    mean_arr = np.asarray(mean_ret_list)
    med_arr = np.asarray(median_ret_list)
    mdd_arr = np.asarray(mdd_list)
    l1, = ax1.plot(thr_arr, comp_arr, label='comp_ret %', color='#1f77b4')
    l2, = ax1.plot(thr_arr, pnl_arr, label='pnl_sum %', color='#ff7f0e')
    l3, = ax2.plot(thr_arr, shp_arr, label='sharpe', color='#2ca02c', alpha=0.8)
    l4, = ax1.plot(thr_arr, mean_arr, label='mean_ret %', color='#9467bd', alpha=0.9)
    l5, = ax1.plot(thr_arr, med_arr, label='median_ret %', color='#8c564b', alpha=0.9)
    l6, = ax1.plot(thr_arr, mdd_arr, label='max_drawdown %', color='#2ca02c', linestyle='--', alpha=0.9)
    if np.isfinite(best_comp_ret):
        idx = int(np.nanargmax(comp_arr))
        ax1.axvline(thr_arr[idx], color=l1.get_color(), linestyle='--', alpha=0.6)
        ax1.scatter([thr_arr[idx]],[comp_arr[idx]], color=l1.get_color(), s=35)
        ax1.annotate(
            f"best comp={comp_arr[idx]:.2f}%\nthr={thr_arr[idx]:.4f}\ntrades={best_trades}\n"
            f"pnl_sum={pnl_arr[idx]:.2f}%\nsharpe={shp_arr[idx]:.3f}\n"
            f"mean={mean_arr[idx]:.2f}%\nmedian={med_arr[idx]:.2f}%\nmax_dd={mdd_arr[idx]:.2f}%",
            xy=(thr_arr[idx], comp_arr[idx]), xytext=(10, 12), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
        )
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('% metrics (comp_ret, pnl_sum)')
    ax2.set_ylabel('Sharpe')
    lines = [l1,l2,l3,l4,l5,l6]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc='best')
    ax1.grid(True, alpha=0.3)
    # constants box in bottom-right similar to curves plot
    const_text = (
        f"VAL_SPLIT={VAL_SPLIT}\nEPOCHS={EPOCHS}\nBATCH={BATCH_SIZE}\nLR0={REDUCE_ON_PLATEAU_START_LR:.2e}\n"
        f"patience0={REDUCE_ON_PLATEAU_START_PATIENCE}\nfactor={REDUCE_ON_PLATEAU_FACTOR}\nmin_lr={REDUCE_ON_PLATEAU_MIN_LR:.1e}\n"
        f"PNL_thr={PNL_FIXED_THRESHOLD}\nDROPOUT={DROPOUT_P:.3f}"
    )
    ax1.text(0.98, 0.02, const_text, transform=ax1.transAxes,
             ha='right', va='bottom', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    # script filename at bottom-left
    try:
        _script_name = Path(__file__).name
    except Exception:
        _script_name = "price_jump_train_colab.py"
    ax1.text(0.02, 0.02, _script_name, transform=ax1.transAxes,
             ha='left', va='bottom', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5))
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