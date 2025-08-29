# price_jump_train_colab.py
# Last modified (MSK): 2025-08-29 14:43 — правка номер 1
# Changes:
# - Add Max IntraTrade DD (price, %) and PnL (seq, %) metrics on threshold
# - Extend max CompRet annotation with new metrics (real values)
# - Legend placed below with increased figure height; restore plot proportions
# - Implement lateral anti-overlap for value rectangles at same x; base anchoring kept
# - Add maker/taker commission constants and switch; compute net returns across curves & threshold
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
from matplotlib.ticker import FuncFormatter
from matplotlib.offsetbox import AnnotationBbox, TextArea
import time

# Commissions (maker/taker) — net PnL calculations
MAKER_FEE = 0.0002
TAKER_FEE = 0.0005
USE_MAKER_FEES = False
ENTRY_FEE = MAKER_FEE if USE_MAKER_FEES else TAKER_FEE
EXIT_FEE  = MAKER_FEE if USE_MAKER_FEES else TAKER_FEE

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
GRADCLIP_MAXNORM_1_APPLY = True
GRADCLIP_MAXNORM = 1.0


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
ret_per_trade_val_fixed = (exit_closes * (1.0 - EXIT_FEE)) / (np.maximum(entry_opens, MIN_DENOM_EPS) * (1.0 + ENTRY_FEE)) - 1.0

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

print(f"Grad clipping: {'ON' if GRADCLIP_MAXNORM_1_APPLY else 'OFF'} (max_norm={GRADCLIP_MAXNORM})")

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
		opt.zero_grad(); loss=lossf(model(x),y); loss.backward();
		if GRADCLIP_MAXNORM_1_APPLY:
			torch.nn.utils.clip_grad_norm_(model.parameters(), GRADCLIP_MAXNORM)
		opt.step()
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
    tab10 = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    colors = {}
    for idx, (name, arr) in enumerate(curves.items()):
        if arr.size == 0:
            continue
        arr_norm = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + PLOT_NORM_EPS)
        line, = plt.plot(
            x[:len(arr_norm)], arr_norm, label=name,
            color=tab10[idx % len(tab10)], linewidth=1.8, alpha=0.95
        )
        colors[name] = line.get_color()
    ax = plt.gca()
    # place constants bottom-right inside; legend left of it
    const_text = (
        f"VAL_SPLIT={VAL_SPLIT}\nEPOCHS={EPOCHS}\nBATCH={BATCH_SIZE}\nLR0={REDUCE_ON_PLATEAU_START_LR:.2e}\n"
        f"patience0={REDUCE_ON_PLATEAU_START_PATIENCE}\nfactor={REDUCE_ON_PLATEAU_FACTOR}\nmin_lr={REDUCE_ON_PLATEAU_MIN_LR:.1e}\n"
        f"PNL_thr={PNL_FIXED_THRESHOLD}\nDROPOUT={DROPOUT_P:.3f}\nGRADCLIP={GRADCLIP_MAXNORM_1_APPLY}\nGRADCLIP_MAXNORM={GRADCLIP_MAXNORM}\nbest_lr_default={REDUCE_ON_PLATEAU_START_LR:.2e}"
    )
    ax.text(0.98, 0.02, const_text, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    # separated max annotations above axes with simple collision avoidance
    try:
        xlen = max(1, len(lr_curve))
        pr_ann = None; pnl_ann = None
        if len(pr_auc_curve) > 0:
            i_best_pr = int(np.nanargmax(pr_auc_curve))
            y_best_pr = (pr_auc_curve[i_best_pr] - np.nanmin(pr_auc_curve)) / (np.nanmax(pr_auc_curve) - np.nanmin(pr_auc_curve) + PLOT_NORM_EPS)
            x_frac_pr = (i_best_pr + 1) / xlen
            pr_ann = ax.annotate(
                f"max PR_AUC={pr_auc_curve[i_best_pr]:.3f} (ep={i_best_pr+1})",
                xy=(i_best_pr+1, y_best_pr), xycoords='data',
                xytext=(x_frac_pr, 1.06), textcoords='axes fraction',
                ha='center', va='bottom', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.8))
        if len(pnl_curve_pct) > 0:
            i_best_pnl = int(np.nanargmax(pnl_curve_pct))
            y_best_pnl = (pnl_curve_pct[i_best_pnl] - np.nanmin(pnl_curve_pct)) / (np.nanmax(pnl_curve_pct) - np.nanmin(pnl_curve_pct) + PLOT_NORM_EPS)
            x_frac_pnl = (i_best_pnl + 1) / xlen
            pnl_ann = ax.annotate(
                f"max PnL={pnl_curve_pct[i_best_pnl]:.2f}% (ep={i_best_pnl+1})",
                xy=(i_best_pnl+1, y_best_pnl), xycoords='data',
                xytext=(x_frac_pnl, 1.12), textcoords='axes fraction',
                ha='center', va='bottom', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.8))
        if pr_ann is not None and pnl_ann is not None:
            (xpr, ypr) = pr_ann.get_position(); (xpn, ypn) = pnl_ann.get_position()
            if abs(xpr - xpn) < 0.08:
                pr_ann.set_position((xpr - 0.06, ypr))
                pnl_ann.set_position((xpn + 0.06, ypn))
    except Exception:
        pass
    leg = plt.legend(loc='lower right', bbox_to_anchor=(0.80, 0.02))
    try:
        fig = plt.gcf(); fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        const_bb = ax.texts[-1].get_window_extent(renderer=renderer)
        const_left_axes = ax.transAxes.inverted().transform((const_bb.x0, const_bb.y0))[0]
        margin = 0.01
        new_x = max(0.02, const_left_axes - margin)
        leg.set_bbox_to_anchor((new_x, 0.02), transform=ax.transAxes)
    except Exception:
        pass
    plt.xlabel('Epoch'); plt.ylabel('Normalized scale [0,1]')
    plt.title('Training curves (normalized)')
    plt.grid(True, alpha=0.3); plt.tight_layout()
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
ret_per_trade_val = (exit_closes * (1.0 - EXIT_FEE)) / (np.maximum(entry_opens, MIN_DENOM_EPS) * (1.0 + ENTRY_FEE)) - 1.0
# lows for intra-trade drawdown computations
lows_all = df["l"].astype(np.float32).values

print(f"Перебор порога по PnL (валидация): min={THR_SWEEP_MIN:.3f}, max={THR_SWEEP_MAX:.3f}, step={THR_SWEEP_STEP:.4f}")
thresholds = np.arange(THR_SWEEP_MIN, THR_SWEEP_MAX + 1e-12, THR_SWEEP_STEP)

best_comp_ret = -np.inf
best_threshold_pnl = float(thresholds[0])
best_trades = 0

thr_list=[]; pnl_list=[]; comp_list=[]; sharpe_list=[]
mean_ret_list=[]; median_ret_list=[]; mdd_list=[]
trades_list = [] # Added for plotting

def _max_intratrade_dd_pct_for_mask(mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    ent = entry_idx[mask]
    dd_min = 0.0
    has_any = False
    for k in ent:
        end = int(k + PRED_WINDOW)
        if end >= len(lows_all):
            continue
        min_low = float(np.min(lows_all[k:end+1]))
        entry_open = float(ds.opens[int(k)]) if int(k) < len(ds.opens) else float('nan')
        if not np.isfinite(entry_open) or entry_open <= 0:
            continue
        dd_i = (min_low / max(entry_open, MIN_DENOM_EPS)) - 1.0
        if not has_any:
            dd_min = dd_i; has_any = True
        else:
            dd_min = min(dd_min, dd_i)
    return float(abs(dd_min) * 100.0) if has_any else 0.0

def _pnl_seq_pct_for_mask(mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    ent = entry_idx[mask]
    order = np.argsort(ent)
    ent_sorted = ent[order]
    r_sorted = ret_per_trade_val[mask][order]
    equity = 1.0
    last_exit = -10**9
    for e_i, r_i in zip(ent_sorted, r_sorted):
        if e_i >= last_exit:
            equity *= (1.0 + float(r_i))
            last_exit = int(e_i + PRED_WINDOW)
    return float((equity - 1.0) * 100.0)

max_intra_dd_list = []
pnl_seq_list = []

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
        max_intra_dd_pct = 0.0
        pnl_seq_pct = 0.0
        trades_list.append(0) # Append 0 for plotting
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
        # new metrics
        max_intra_dd_pct = _max_intratrade_dd_pct_for_mask(mask)
        pnl_seq_pct = _pnl_seq_pct_for_mask(mask)
        trades_list.append(n_trades) # Append n_trades for plotting
    thr_list.append(float(t)); pnl_list.append(sum_ret*100.0); comp_list.append(comp_ret*100.0 if np.isfinite(comp_ret) else np.nan); sharpe_list.append(sharpe); mean_ret_list.append(mean_ret); median_ret_list.append(median_ret); mdd_list.append(mdd_pct); max_intra_dd_list.append(max_intra_dd_pct); pnl_seq_list.append(pnl_seq_pct)
    if comp_ret > best_comp_ret:
        best_comp_ret = comp_ret
        best_threshold_pnl = float(t)
        best_trades = n_trades

print(f"Выбран порог по PnL (валидация): {best_threshold_pnl:.4f}, comp_ret={best_comp_ret*100 if np.isfinite(best_comp_ret) else float('nan'):.2f}% trades={best_trades}")

try:
    fig, ax1 = plt.subplots(figsize=(9.2,6.5))
    ax2 = ax1.twinx()
    thr_arr = np.asarray(thr_list)
    pnl_arr = np.asarray(pnl_list)
    comp_arr = np.asarray(comp_list)
    shp_arr = np.asarray(sharpe_list)
    mean_arr = np.asarray(mean_ret_list)
    med_arr = np.asarray(median_ret_list)
    mdd_arr = np.asarray(mdd_list)
    intradd_arr = np.asarray(max_intra_dd_list)
    pnlseq_arr = np.asarray(pnl_seq_list)
    # normalize left-axis metrics
    def _norm(a):
        a = np.asarray(a, dtype=np.float64)
        return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a) + PLOT_NORM_EPS) if a.size>0 else a
    comp_n = _norm(comp_arr); pnl_n = _norm(pnl_arr); mean_n = _norm(mean_arr); med_n = _norm(med_arr); mdd_n = _norm(mdd_arr); intradd_n = _norm(intradd_arr); pnlseq_n = _norm(pnlseq_arr)
    # styles
    l1, = ax1.plot(thr_arr, comp_n, label='comp_ret (norm)', color='#1f77b4', linewidth=1.8)
    l2, = ax1.plot(thr_arr, pnl_n,  label='pnl_sum (norm)',  color='#ff7f0e', linewidth=1.8)
    l3, = ax1.plot(thr_arr, mean_n, label='mean_ret (norm)', color='#000000', linestyle='--', linewidth=1.6)
    l4, = ax1.plot(thr_arr, med_n,  label='median_ret (norm)', color='#7f7f7f', linestyle='--', linewidth=1.6)
    l5, = ax1.plot(thr_arr, mdd_n,  label='max_drawdown (norm)', color='#2ca02c', linestyle='-', linewidth=1.6)
    l6, = ax2.plot(thr_arr, shp_arr, label='Sharpe', color='#9467bd', alpha=0.9)
    # new metrics on left axis
    l8, = ax1.plot(thr_arr, intradd_n, label='Max IntraTrade DD (price, %)', color='#98df8a', linewidth=1.6)
    l9, = ax1.plot(thr_arr, pnlseq_n, label='PnL (seq, %)', color='#d62728', linewidth=1.6)
    # add Trades on separate invisible y-axis
    ax3 = ax1.twinx(); ax3.get_yaxis().set_visible(False)
    l7, = ax3.plot(thr_arr, np.asarray(trades_list), label='Trades', color='#8c564b')
    # constants box OUTSIDE axes on the right; legend BELOW axes
    const_text = (
        f"VAL_SPLIT={VAL_SPLIT}\nEPOCHS={EPOCHS}\nBATCH={BATCH_SIZE}\nLR0={REDUCE_ON_PLATEAU_START_LR:.2e}\n"
        f"patience0={REDUCE_ON_PLATEAU_START_PATIENCE}\nfactor={REDUCE_ON_PLATEAU_FACTOR}\nmin_lr={REDUCE_ON_PLATEAU_MIN_LR:.1e}\n"
        f"PNL_thr={PNL_FIXED_THRESHOLD}\nDROPOUT={DROPOUT_P:.3f}\nGRADCLIP={GRADCLIP_MAXNORM_1_APPLY}\nGRADCLIP_MAXNORM={GRADCLIP_MAXNORM}\nbest_lr_default={REDUCE_ON_PLATEAU_START_LR:.2e}"
    )
    fig.text(0.985, 0.02, const_text, ha='right', va='bottom', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    handles, labels = [], []
    for ln in (l1, l2, l3, l4, l5, l6, l7, l8, l9):
        handles.append(ln); labels.append(ln.get_label())
    leg2 = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=5)
    ax1.grid(True, alpha=0.3)
    # fixed-point annotations at thr_min, thirds, thr_max: real values, hard-anchored with lateral anti-overlap when colliding at same x
    try:
        thr_min_v = float(THR_SWEEP_MIN); thr_max_v = float(THR_SWEEP_MAX)
        delta = thr_max_v - thr_min_v
        t_points = [thr_min_v, thr_min_v + delta/3.0, thr_min_v + 2.0*delta/3.0, thr_max_v]
        series = [
            (comp_n, comp_arr, l1.get_color()),
            (pnl_n,  pnl_arr,  l2.get_color()),
            (mean_n, mean_arr, l3.get_color()),
            (med_n,  med_arr,  l4.get_color()),
            (mdd_n,  mdd_arr,  l5.get_color()),
            (intradd_n, intradd_arr, l8.get_color()),
            (pnlseq_n, pnlseq_arr, l9.get_color()),
        ]
        y_tol = 0.04
        for t in t_points:
            idx = int(np.argmin(np.abs(thr_arr - t)))
            items = []
            for (yn, yr, col) in series:
                items.append((float(yn[idx]), float(yr[idx]), col))
            # group by approximate y to detect collisions
            buckets = {}
            for j, (yv, rv, col) in enumerate(items):
                b = int(round(yv / max(y_tol, 1e-6)))
                buckets.setdefault(b, []).append((j, yv, rv, col))
            # place annotations
            for b, group in buckets.items():
                if len(group) == 1:
                    _, yv, rv, col = group[0]
                    ax1.scatter([thr_arr[idx]],[yv], color=col, s=14)
                    ab = AnnotationBbox(TextArea(f"{rv:.2f}", textprops=dict(color=col, fontsize=7)),
                                         (thr_arr[idx], yv),
                                         box_alignment=(0.5, 1.0),
                                         bboxprops=dict(boxstyle='round,pad=0.15', fc='white', ec=col, alpha=0.7))
                    ax1.add_artist(ab)
                else:
                    # two or more -> take first as left, second as right; others keep center
                    for k, (_j, yv, rv, col) in enumerate(group):
                        ax1.scatter([thr_arr[idx]],[yv], color=col, s=14)
                        if k == 0:
                            align = (1.0, 0.5)
                        elif k == 1:
                            align = (0.0, 0.5)
                        else:
                            align = (0.5, 1.0)
                        ab = AnnotationBbox(TextArea(f"{rv:.2f}", textprops=dict(color=col, fontsize=7)),
                                             (thr_arr[idx], yv),
                                             box_alignment=align,
                                             bboxprops=dict(boxstyle='round,pad=0.15', fc='white', ec=col, alpha=0.7))
                        ax1.add_artist(ab)
                

        # detailed annotation at max comp_ret with new metrics
        if np.any(np.isfinite(comp_arr)):
            i_best = int(np.nanargmax(comp_arr))
            best_thr = float(thr_arr[i_best])
            ax1.axvline(best_thr, color=l1.get_color(), linestyle='--', linewidth=1.0, alpha=0.7)
            ax1.scatter([best_thr],[comp_n[i_best]], color=l1.get_color(), s=18)
            # compute real metrics at best
            mask_best = (val_probs_all >= best_thr)
            n_best = int(mask_best.sum())
            r_best = ret_per_trade_val[mask_best] if n_best>0 else np.array([], dtype=np.float64)
            sharpe_best = float(np.mean(r_best) / (np.std(r_best) + PLOT_NORM_EPS)) if r_best.size>=SHARPE_MIN_SAMPLES else 0.0
            sum_best = float(np.sum(r_best)) * 100.0
            mean_best = float(np.mean(r_best) * 100.0) if r_best.size>0 else 0.0
            med_best  = float(np.median(r_best) * 100.0) if r_best.size>0 else 0.0
            # equity dd and avg_dd
            ent_best = entry_idx[mask_best]
            ord_best = np.argsort(ent_best)
            r_sorted_best = r_best[ord_best] if r_best.size>0 else np.array([], dtype=np.float64)
            if r_sorted_best.size>0:
                equity_best = np.cumprod(1.0 + r_sorted_best.astype(np.float64))
                run_max_b = np.maximum.accumulate(equity_best)
                dd_series = equity_best / (run_max_b + COMP_EPS) - 1.0
                max_dd_best = float(abs(np.min(dd_series)) * 100.0)
                avg_dd_best = float(abs(np.mean(np.clip(dd_series, -1.0, 0.0))) * 100.0)
            else:
                max_dd_best = 0.0; avg_dd_best = 0.0
            max_intra_best = _max_intratrade_dd_pct_for_mask(mask_best)
            pnl_seq_best = _pnl_seq_pct_for_mask(mask_best)
            text = (
                f"comp_ret: {float(comp_arr[i_best]):.2f}%\n"
                f"thr: {best_thr:.3f}\n"
                f"trades: {n_best}\n"
                f"pnl_sum: {sum_best:.2f}%\n"
                f"sharpe: {sharpe_best:.2f}\n"
                f"mean: {mean_best:.2f}%\n"
                f"median: {med_best:.2f}%\n"
                f"max_dd: {max_dd_best:.2f}%\n"
                f"avg_dd: {avg_dd_best:.2f}%\n"
                f"max_intratrade_dd: {max_intra_best:.2f}%\n"
                f"pnl_seq: {pnl_seq_best:.2f}%"
            )
            ax1.annotate(text, xy=(best_thr, comp_n[i_best]), xycoords='data',
                         xytext=(0.5, 1.04), textcoords='axes fraction',
                         ha='center', va='bottom', fontsize=8,
                         bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    except Exception:
        pass
    plt.tight_layout(rect=[0.0, 0.22, 0.86, 1])
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