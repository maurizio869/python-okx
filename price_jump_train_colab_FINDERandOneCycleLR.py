# price_jump_train_colab_FINDERandOneCycleLR.py
# Last modified (MSK): 2025-08-29 14:43 — правка номер 1
# Changes:
# - Add Max IntraTrade DD (price, %) and PnL (seq, %) metrics on threshold
# - Extend max CompRet annotation with new metrics (real values)
# - Add Trades to legend; place legend below; restore plot proportions
# - Implement lateral anti-overlap for value rectangles at same x
# - Add maker/taker commission constants and switch; compute net returns across curves & threshold
"""Тренировка LSTM: LR Finder + OneCycleLR вместо ReduceLROnPlateau.
- 1-я стадия: короткий LR finder на подмножестве данных/эпохах
- 2-я стадия: основное обучение с OneCycleLR
Остальное: как в базовом тренинге (v-канал, SEQ_LEN=30, ранний стоп по PR AUC, PnL@best, подбор порога по PnL).
"""
from pathlib import Path
import json, math, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.offsetbox import AnnotationBbox, TextArea
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader, random_split
import time

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
try:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass

SEQ_LEN, PRED_WINDOW, JUMP_THRESHOLD = 30, 5, 0.0035
# Commissions (maker/taker) — net PnL calculations
MAKER_FEE = 0.0002
TAKER_FEE = 0.0005
USE_MAKER_FEES = False
ENTRY_FEE = MAKER_FEE if USE_MAKER_FEES else TAKER_FEE
EXIT_FEE  = MAKER_FEE if USE_MAKER_FEES else TAKER_FEE

TRAIN_JSON = Path("candles_10d.json")
MODEL_PATH = Path("lstm_jump_PRAUC.pt")
PNL_MODEL_PATH = Path("lstm_jump_pnl.pt")
VALACC_MODEL_PATH = Path("lstm_jump_valacc.pt")
MODEL_META_PATH = MODEL_PATH.with_suffix(".meta.json")
HYPER_PATH = MODEL_PATH.with_suffix(".hyper.json")
VAL_SPLIT, EPOCHS = 0.2, 360
BATCH_SIZE, BASE_LR = 512, 3e-4
best_lr_default = 2.17e-03
# Tunable LR Finder params
LR_FINDER_MIN_FACTOR = 1.0/20.0  # min_lr = BASE_LR * LR_FINDER_MIN_FACTOR
LR_FINDER_MAX_FACTOR = 8.0       # max_lr = BASE_LR * LR_FINDER_MAX_FACTOR
# How to pick OneCycle max_lr from best_lr and clip range around BASE_LR
BEST_LR_MULTIPLIER = 0.7         # max_lr ~ BEST_LR_MULTИПLIER * best_lr
CLIP_MIN_FACTOR = 0.8            # clip lower bound = BASE_LR * CLIP_MIN_FACTOR
CLIP_MAX_FACTOR = 8.0            # clip upper bound = BASE_LR * CLIP_MAX_FACTOR
# OneCycleLR shape parameters
ONECYCLE_PCT_START = 0.12
ONECYCLE_DIV_FACTOR = 2.0
ONECYCLE_FINAL_DIV_FACTOR = 3.5
WEIGHT_DECAY = 3.5e-5
# Default dropout if no hyper/meta provided
DEFAULT_DROPOUT = 0.35
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EARLY_STOP_EPOCHS = 80
NPR_EPS = 1e-12
SAVE_MIN_PR_AUC = 0.58
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
        self.opens = df["o"].astype(np.float32).values
        self.highs = df["h"].astype(np.float32).values
        self.lows = df["l"].astype(np.float32).values
        self.volumes = df["v"].astype(np.float32).values
        self.scaler = StandardScaler()
        # build samples (seq_len -> future label by max close in window)
        self.samples = []
        for i in range(SEQ_LEN, len(self.closes) - PRED_WINDOW):
            current_open = float(self.opens[i])
            max_close = float(np.max(self.closes[i+1:i+PRED_WINDOW+1]))
            label = 1 if (max_close / max(current_open, 1e-12) - 1.0) >= JUMP_THRESHOLD else 0
            self.samples.append((i, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        i, y = self.samples[idx]
        x_seq = np.stack([
            self.closes[i-SEQ_LEN:i],
            self.opens[i-SEQ_LEN:i],
            self.highs[i-SEQ_LEN:i],
            self.lows[i-SEQ_LEN:i],
            self.volumes[i-SEQ_LEN:i],
        ], axis=0).astype(np.float32)
        x_seq = torch.from_numpy(x_seq)
        return x_seq, int(y)

class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 2, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x); return self.fc(h[-1])

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
POS_FRAC = float(pos_cnt) / max(1, (pos_cnt + neg_cnt))

val = int(len(ds)*VAL_SPLIT)
# fixed split
gen = torch.Generator().manual_seed(SEED)
train_ds, val_ds = random_split(ds,[len(ds)-val,val], generator=gen)
train_loader = DataLoader(train_ds,BATCH_SIZE,shuffle=True)
val_loader   = DataLoader(val_ds,BATCH_SIZE)

# Read optional overrides for dropout and base_lr from meta
DROPOUT_P = DEFAULT_DROPOUT
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
				BASE_LR = float(hyper['base_lr']); _got_base_lr = True; _src_base_lr = f"{HYPER_PATH}"
	elif MODEL_META_PATH.exists():
		with open(MODEL_META_PATH, 'r', encoding='utf-8') as mf:
			meta0 = json.load(mf)
		if isinstance(meta0, dict):
			if 'dropout' in meta0:
				DROPOUT_P = float(meta0['dropout']); _got_dropout = True; _src_dropout = f"{MODEL_META_PATH}"
			if 'base_lr' in meta0:
				BASE_LR = float(meta0['base_lr']); _got_base_lr = True; _src_base_lr = f"{MODEL_META_PATH}"
except Exception as ex:
	print(f"! Не удалось прочитать hyper/meta для dropout/base_lr: {ex}")

if _got_dropout:
	print(f"dropout прочитан из {_src_dropout}: {DROPOUT_P:.3f}")
else:
	print(f"dropout взят по умолчанию: {DROPOUT_P:.3f}")
if _got_base_lr:
	print(f"base_lr прочитан из {_src_base_lr}: {BASE_LR:.2e}")
else:
	print(f"base_lr взят по умолчанию: {BASE_LR:.2e}")

model = LSTMClassifier(dropout=DROPOUT_P).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), BASE_LR)
pos_weight = neg_cnt / max(pos_cnt, 1)
class_weights = torch.tensor([1.0, pos_weight], device=DEVICE)
lossf = nn.CrossEntropyLoss(weight=class_weights)

# LR Finder: 1 эпоха по train_loader, lr от BASE_LR/20 до BASE_LR*8
print("LR Finder: старт…")
min_lr, max_lr = BASE_LR*LR_FINDER_MIN_FACTOR, BASE_LR*LR_FINDER_MAX_FACTOR
finder_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=False)
num_steps = max(1, len(finder_loader))
lr_mult = (max_lr/min_lr) ** (1/num_steps)
print(f"LR Finder params: BASE_LR={BASE_LR:.2e}, min_lr={min_lr:.2e}, max_lr={max_lr:.2e}, steps={num_steps}, lr_mult≈{lr_mult:.6f}")
for pg in opt.param_groups: pg['lr'] = min_lr
best_loss = float('inf'); best_lr = BASE_LR
model.eval(); step_id=0
for xb, yb in finder_loader:
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    opt.zero_grad(); logits = model(xb); loss = lossf(logits, yb)
    loss.backward(); opt.step()
    if loss.item() < best_loss:
        best_loss = loss.item(); best_lr = opt.param_groups[0]['lr']
    for pg in opt.param_groups: pg['lr'] *= lr_mult
    step_id += 1
print(f"LR Finder: best_lr≈{best_lr:.2e}, best_loss={best_loss:.4f}")
# fallback if unstable
best_lr = best_lr_default
print("lr finder is unstable, best_lr=", best_lr)
# OneCycleLR на весь ран: max_lr = BEST_LR_MULTИПLIER×best_lr (clipped)
max_lr_use = float(np.clip(BEST_LR_MULTIPLIER*best_lr, BASE_LR*CLIP_MIN_FACTOR, BASE_LR*CLIP_MAX_FACTOR))
opt = torch.optim.Adam(model.parameters(), BASE_LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=max_lr_use, epochs=EPOCHS, steps_per_epoch=max(1, len(train_loader)),
    pct_start=ONECYCLE_PCT_START, div_factor=ONECYCLE_DIV_FACTOR, final_div_factor=ONECYCLE_FINAL_DIV_FACTOR
)
# Плановая кривая LR по эпохам
try:
    tmp_opt = torch.optim.Adam(model.parameters(), BASE_LR, weight_decay=WEIGHT_DECAY)
    tmp_sched = torch.optim.lr_scheduler.OneCycleLR(
        tmp_opt, max_lr=max_lr_use, epochs=EPOCHS, steps_per_epoch=max(1, len(train_loader)),
        pct_start=ONECYCLE_PCT_START, div_factor=ONECYCLE_DIV_FACTOR, final_div_factor=ONECYCLE_FINAL_DIV_FACTOR
    )
    planned_lr = []
    for _ep in range(EPOCHS):
        for _ in range(max(1, len(train_loader))):
            tmp_sched.step()
        planned_lr.append(tmp_opt.param_groups[0]['lr'])
    plt.figure(figsize=(6,3))
    # unique style
    plt.plot(range(1, len(planned_lr)+1), planned_lr, label='Planned LR', color='#1f77b4', linewidth=1.8)
    plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.title('Planned OneCycle LR by epoch')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    def _sci_fmt(y, pos):
        s = f"{y:.0e}"; return s.replace('e-0','e-').replace('e+0','e+')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(_sci_fmt))
    ts = None
    try:
        from datetime import datetime
        import pytz
        msk = pytz.timezone('Europe/Moscow')
        ts = datetime.now(msk).strftime('%Y%m%d_%H%M')
    except Exception:
        pass
    out_name = f"onecycle_lr_curve_{ts}.png" if ts else "onecycle_lr_curve.png"
    plt.savefig(out_name, dpi=120)
    print(f"Saved LR curve to {Path(out_name).resolve()}")
    try:
        from IPython.display import Image, display
        display(Image(out_name))
    except Exception:
        pass
    plt.close()
except Exception as ex:
    print(f"! Не удалось построить/сохранить план LR: {ex}")

# PnL@0.565 support
val_indices = np.asarray(val_ds.indices, dtype=np.int64)
entry_idx = val_indices + SEQ_LEN
entry_opens = ds.opens[entry_idx]
exit_closes = ds.closes[entry_idx + PRED_WINDOW]
ret_val_fixed = (exit_closes * (1.0 - EXIT_FEE)) / (np.maximum(entry_opens, 1e-12) * (1.0 + ENTRY_FEE)) - 1.0

# Threshold sweep defaults for in-epoch PnL selection
thr_min, thr_max, thr_step = 0.15, 0.85, 0.0025
last_best_thr = 0.565
best_pnl_thr = last_best_thr
pnl_best_sum = 0.0
trades_best = 0

best_pr_auc = -1.0
best_pnl_sum = -float('inf')
best_val_acc = -1.0
epochs_no_improve = 0
# Lists to collect per-epoch metrics for post-training plots
lr_curve = []
pr_auc_curve = []
npr_auc_curve = []
pnl_curve_pct = []
val_acc_curve = []
for e in range(1, EPOCHS+1):
    _t0 = time.time()
    model.train(); total_loss=0.0
    for xb,yb in train_loader:
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(); logits=model(xb); loss=lossf(logits,yb)
        loss.backward();
        if GRADCLIP_MAXNORM_1_APPLY:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADCLIP_MAXNORM)
        opt.step(); sched.step()
        total_loss += loss.item()*xb.size(0)

    model.eval(); corr=tot_s=0
    val_targets=[]; val_probs=[]; val_preds=[]
    with torch.no_grad():
        for xb,yb in val_loader:
            logits=model(xb.to(DEVICE))
            prob1=torch.softmax(logits,dim=1)[:,1].cpu()
            pred=(prob1>=0.5).to(torch.long); y_cpu=yb.to(torch.long)
            corr+=(pred.cpu()==y_cpu).sum().item(); tot_s+=y_cpu.size(0)
            val_targets.extend(y_cpu.tolist()); val_probs.extend(prob1.tolist()); val_preds.extend(pred.cpu().tolist())
    try: roc_auc=roc_auc_score(val_targets,val_probs)
    except Exception: roc_auc=float('nan')
    f1=f1_score(val_targets,val_preds,zero_division=0)
    pr_auc=average_precision_score(val_targets,val_probs)
    npr_auc=(pr_auc - POS_FRAC) / (1.0 - POS_FRAC + NPR_EPS)

    # compute best-threshold PnL on validation every 5 epochs
    val_probs_np=np.asarray(val_probs,dtype=np.float32)
    y_true_np = np.asarray(val_targets, dtype=np.int32)
    if e % 10 == 0 or e == 1:
        best_comp=-np.inf; best_thr=last_best_thr; best_trades=0; best_sum=0.0
        for t in np.arange(thr_min,thr_max+1e-12,thr_step):
            m=(val_probs_np>=t); n=int(m.sum())
            if n==0:
                comp=-np.inf; sret=0.0; prec=0.0
            else:
                # precision constraint removed
                r=ret_val_fixed[m]
                comp=-1.0 if np.any(r<=-0.999999) else float(np.exp(np.sum(np.log1p(r)))-1.0)
                sret=float(np.sum(r))
            if comp>best_comp:
                best_comp=comp; best_thr=float(t); best_trades=n; best_sum=sret
        last_best_thr = best_thr
        pnl_best_sum = best_sum
        trades_best = best_trades
    else:
        m=(val_probs_np>=last_best_thr); trades_best=int(m.sum())
        pnl_best_sum = float(np.sum(ret_val_fixed[m])) if trades_best>0 else 0.0

    curr_lr = opt.param_groups[0]['lr']
    val_acc = (corr/tot_s) if tot_s>0 else 0.0
    _dt = time.time() - _t0
    print(f'Epoch {e}/{EPOCHS} lr {curr_lr:.2e} loss {total_loss/len(train_ds):.4f} '
          f'val_acc {val_acc:.3f} F1 {f1:.3f} ROC_AUC {roc_auc:.3f} PR_AUC {pr_auc:.3f} nPR_AUC {npr_auc:.3f} '
          f'PNL@best(thr={last_best_thr:.4f}) {pnl_best_sum*100:.2f}% trades={trades_best} time {(_dt):.1f}s')

    # collect curves
    lr_curve.append(curr_lr)
    pr_auc_curve.append(float(pr_auc))
    npr_auc_curve.append(float(npr_auc))
    pnl_curve_pct.append(float(pnl_best_sum*100.0))
    val_acc_curve.append(float(val_acc))

    if pr_auc > best_pr_auc + 1e-6:
        best_pr_auc = pr_auc; epochs_no_improve = 0
        if pr_auc > SAVE_MIN_PR_AUC:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state":model.state_dict(),"scaler":ds.scaler,
                        "meta":{"seq_len":SEQ_LEN,"pred_window":PRED_WINDOW}}, MODEL_PATH)
            try:
                with open(MODEL_META_PATH,'w',encoding='utf-8') as mf:
                    json.dump({"seq_len":int(SEQ_LEN),"pred_window":int(PRED_WINDOW)}, mf)
            except Exception as ex:
                print(f"! Не удалось записать meta-файл {MODEL_META_PATH}: {ex}")
            print(f"✓ Сохранена новая лучшая модель по PR_AUC (PR_AUC={best_pr_auc:.3f}) в {MODEL_PATH.resolve()}")
    
    # save best-by- ValAcc
    if val_acc > best_val_acc + 1e-9:
        best_val_acc = val_acc
        if pr_auc > SAVE_MIN_PR_AUC:
            VALACC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state":model.state_dict(),"scaler":ds.scaler,
                        "meta":{"seq_len":SEQ_LEN,"pred_window":PRED_WINDOW}}, VALACC_MODEL_PATH)
            print(f"✓ Сохранена новая лучшая модель по ValAcc (ValAcc={best_val_acc:.3f}) в {VALACC_MODEL_PATH.resolve()}")

    # save best-by-PnL (using best-threshold PnL sum)
    if pnl_best_sum > best_pnl_sum + 1e-12:
        best_pnl_sum = pnl_best_sum
        best_pnl_thr = last_best_thr
        if pr_auc > SAVE_MIN_PR_AUC:
            PNL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state":model.state_dict(),"scaler":ds.scaler,
                        "meta":{"seq_len":SEQ_LEN,"pred_window":PRED_WINDOW,"threshold":best_pnl_thr}}, PNL_MODEL_PATH)
            print(f"✓ Сохранена новая лучшая модель по PnL (pnl@{best_pnl_thr:.4f}={best_pnl_sum*100:.2f}%) в {PNL_MODEL_PATH.resolve()}")

# Итоговые сообщения о лучших моделях
if best_pr_auc > -1.0:
    print(f"Лучшая модель (PR_AUC={best_pr_auc:.3f}) сохранена в {MODEL_PATH.resolve()}")
if best_pnl_sum > -float('inf'):
    print(f"Лучшая модель с pnl@{best_pnl_thr:.4f}={best_pnl_sum*100:.2f}% сохранена в {PNL_MODEL_PATH.resolve()}")

# PnL threshold sweep (0.15..0.60 step 0.0025)
print("Подбираем порог по PnL на валидационном наборе…")
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"]); model.to(DEVICE).eval()

val_targets_all=[]; val_probs_all=[]; val_preds_all=[]
with torch.no_grad():
    for xb,yb in val_loader:
        logits=model(xb.to(DEVICE)); prob1=torch.softmax(logits,dim=1)[:,1].cpu()
        pred=(prob1>=0.5).to(torch.long); y_cpu=yb.to(torch.long)
        val_targets_all.extend(y_cpu.tolist()); val_probs_all.extend(prob1.tolist()); val_preds_all.extend(pred.cpu().tolist())

ret_val = (exit_closes * (1.0 - EXIT_FEE)) / (np.maximum(entry_opens,1e-12) * (1.0 + ENTRY_FEE)) - 1.0
thr_min,thr_max,thr_step=0.15,0.85,0.0025
print(f"Перебор порога по PnL (валидация): min={thr_min:.3f}, max={thr_max:.3f}, step={thr_step:.4f}")
thresholds=np.arange(thr_min,thr_max+1e-12,thr_step)
# collect metrics for plotting
thr_list=[]; pnl_list=[]; comp_list=[]; sharpe_list=[]; trades_list=[]; mean_ret_list=[]; median_ret_list=[]; mdd_list=[]
def _safe_sharpe_arr(r: np.ndarray) -> float:
    if r.size < 2: return 0.0
    std = float(np.std(r))
    return float(np.mean(r) / (std + 1e-12))
best_comp=-np.inf; best_thr=float(thresholds[0]); best_trades=0
for t in thresholds:
    m=(val_probs_all>=t); n=int(m.sum())
    if n==0:
        comp=-np.inf; shp=0.0; sret=0.0; meanp=0.0; medp=0.0; mddp=0.0
    else:
        r=ret_val[m]
        comp=-1.0 if np.any(r<=-0.999999) else float(np.exp(np.sum(np.log1p(r)))-1.0)
        shp=_safe_sharpe_arr(r)
        sret=float(np.sum(r))
        meanp = float(np.mean(r)*100.0)
        medp  = float(np.median(r)*100.0)
        # max drawdown on equity in chronological order
        ent = entry_idx[m]
        order = np.argsort(ent)
        r_sorted = r[order]
        equity = np.cumprod(1.0 + r_sorted.astype(np.float64))
        run_max = np.maximum.accumulate(equity)
        dd = np.min(equity / (run_max + 1e-12) - 1.0) if equity.size>0 else 0.0
        mddp = float(abs(dd) * 100.0)
    thr_list.append(float(t)); pnl_list.append(sret*100.0); comp_list.append(comp*100.0 if np.isfinite(comp) else np.nan); sharpe_list.append(shp); trades_list.append(n); mean_ret_list.append(meanp); median_ret_list.append(medp); mdd_list.append(mddp)
    if comp>best_comp: best_comp=comp; best_thr=float(t); best_trades=n
print(f"Выбран порог по PnL (валидация): {best_thr:.4f}, comp_ret={best_comp*100 if np.isfinite(best_comp) else float('nan'):.2f}% trades={best_trades}")
# plot metrics vs threshold with max comp_ret annotated
try:
    fig, ax1 = plt.subplots(figsize=(9.2,6.5))
    ax2 = ax1.twinx()
    # normalize left-axis metrics
    def _norm(a):
        a = np.asarray(a, dtype=np.float64)
        return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a) + 1e-12) if a.size>0 else a
    comp_arr = np.asarray(comp_list)
    pnl_arr = np.asarray(pnl_list)
    mean_arr = np.asarray(mean_ret_list)
    med_arr  = np.asarray(median_ret_list)
    mdd_arr  = np.asarray(mdd_list)
    thr_arr  = np.asarray(thr_list)
    # additional arrays for new metrics
    # compute per-threshold max intra-trade drawdown (price) and sequential PnL
    lows_all = df["l"].astype(np.float32).values
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
            dd_i = (min_low / max(entry_open, 1e-12)) - 1.0
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
        r_sorted = ret_val[mask][order]
        equity = 1.0
        last_exit = -10**9
        for e_i, r_i in zip(ent_sorted, r_sorted):
            if e_i >= last_exit:
                equity *= (1.0 + float(r_i))
                last_exit = int(e_i + PRED_WINDOW)
        return float((equity - 1.0) * 100.0)
    max_intra_dd_list = []
    pnl_seq_list = []
    for t in thr_arr:
        mask = (val_probs_all >= t)
        max_intra_dd_list.append(_max_intratrade_dd_pct_for_mask(mask))
        pnl_seq_list.append(_pnl_seq_pct_for_mask(mask))
    intradd_arr = np.asarray(max_intra_dd_list)
    pnlseq_arr = np.asarray(pnl_seq_list)
    comp_n = _norm(comp_arr); pnl_n = _norm(pnl_arr); mean_n = _norm(mean_arr); med_n = _norm(med_arr); mdd_n = _norm(mdd_arr); intradd_n = _norm(intradd_arr); pnlseq_n = _norm(pnlseq_arr)
    # styles
    l1, = ax1.plot(thr_arr, comp_n, label='comp_ret (norm)', color='#1f77b4', linewidth=1.8)
    l2, = ax1.plot(thr_arr, pnl_n,  label='pnl_sum (norm)',  color='#ff7f0e', linewidth=1.8)
    l3, = ax1.plot(thr_arr, mean_n, label='mean_ret (norm)', color='#000000', linestyle='--', linewidth=1.6)
    l4, = ax1.plot(thr_arr, med_n,  label='median_ret (norm)', color='#7f7f7f', linestyle='--', linewidth=1.6)
    l5, = ax1.plot(thr_arr, mdd_n,  label='max_drawdown (norm)', color='#2ca02c', linestyle='-', linewidth=1.6)
    l6, = ax2.plot(thr_arr, np.asarray(sharpe_list), label='Sharpe', color='#9467bd', alpha=0.9)
    # add Trades on separate invisible y-axis
    ax3 = ax1.twinx(); ax3.get_yaxis().set_visible(False)
    l7, = ax3.plot(thr_arr, np.asarray(trades_list), label='Trades', color='#8c564b')
    # new metrics on left axis
    l8, = ax1.plot(thr_arr, intradd_n, label='Max IntraTrade DD (price, %)', color='#98df8a', linewidth=1.6)
    l9, = ax1.plot(thr_arr, pnlseq_n, label='PnL (seq, %)', color='#d62728', linewidth=1.6)
    # constants box OUTSIDE axes on the right; legend BELOW axes
    const_text = (
        f"SEQ_LEN={SEQ_LEN}\nPRED_WINDOW={PRED_WINDOW}\nVAL_SPLIT={VAL_SPLIT}\n"
        f"EPOCHS={EPOCHS}\nBATCH={BATCH_SIZE}\nBASE_LR={BASE_LR:.2e}\n"
        f"pct_start={ONECYCLE_PCT_START}\ndiv_factor={ONECYCLE_DIV_FACTOR}\nfinal_div={ONECYCLE_FINAL_DIV_FACTOR}\n"
        f"WD={WEIGHT_DECAY}\nDROPOUT={DROPOUT_P:.3f}\nBEST_LR_MULT={BEST_LR_MULTIPLIER}\nGRADCLIP={GRADCLIP_MAXNORM_1_APPLY}\nGRADCLIP_MAXNORM={GRADCLIP_MAXNORM}\nUSE_STANDARD_SCALER=False\nbest_lr_default={best_lr_default:.2e}"
    )
    fig.text(0.985, 0.02, const_text, ha='right', va='bottom', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    handles, labels = [], []
    for ln in (l1, l2, l3, l4, l5, l6, l7, l8, l9):
        handles.append(ln); labels.append(ln.get_label())
    leg2 = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=5)
    ax1.grid(True, alpha=0.3)
    # fixed-point annotations at thr_min, thirds, thr_max: real values, hard-anchored; lateral anti-overlap at same x
    try:
        thr_min_v = float(thr_min); thr_max_v = float(thr_max)
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
            side_by_bucket = {}
            for (yn, yr, col) in series:
                yv = float(yn[idx]); rv = float(yr[idx])
                ax1.scatter([thr_arr[idx]],[yv], color=col, s=14)
                align = (0.5, 1.0)
                bucket = int(round(yv / max(y_tol, 1e-6)))
                used = side_by_bucket.get(bucket, None)
                if used is None:
                    align = (1.0, 0.5); side_by_bucket[bucket] = 'left'
                elif used == 'left':
                    align = (0.0, 0.5); side_by_bucket[bucket] = 'both'
                ab = AnnotationBbox(TextArea(f"{rv:.2f}", textprops=dict(color=col, fontsize=7)),
                                     (thr_arr[idx], yv),
                                     box_alignment=align,
                                     bboxprops=dict(boxstyle='round,pad=0.15', fc='white', ec=col, alpha=0.7))
                ax1.add_artist(ab)
        # detailed annotation at max comp_ret with new metrics
        if np.any(np.isfinite(comp_arr)):
            i_best = int(np.nanargmax(comp_arr))
            best_thr_local = float(thr_arr[i_best])
            ax1.axvline(best_thr_local, color=l1.get_color(), linestyle='--', linewidth=1.0, alpha=0.7)
            ax1.scatter([best_thr_local],[comp_n[i_best]], color=l1.get_color(), s=18)
            mask_best = (val_probs_all >= best_thr_local)
            n_best = int(mask_best.sum())
            r_best = ret_val[mask_best] if n_best>0 else np.array([], dtype=np.float64)
            sharpe_best = float(np.mean(r_best) / (np.std(r_best) + 1e-12)) if r_best.size>=2 else 0.0
            sum_best = float(np.sum(r_best)) * 100.0
            mean_best = float(np.mean(r_best) * 100.0) if r_best.size>0 else 0.0
            med_best  = float(np.median(r_best) * 100.0) if r_best.size>0 else 0.0
            ent_best = entry_idx[mask_best]
            ord_best = np.argsort(ent_best)
            r_sorted_best = r_best[ord_best] if r_best.size>0 else np.array([], dtype=np.float64)
            if r_sorted_best.size>0:
                equity_best = np.cumprod(1.0 + r_sorted_best.astype(np.float64))
                run_max_b = np.maximum.accumulate(equity_best)
                dd_series = equity_best / (run_max_b + 1e-12) - 1.0
                max_dd_best = float(abs(np.min(dd_series)) * 100.0)
                avg_dd_best = float(abs(np.mean(np.clip(dd_series, -1.0, 0.0))) * 100.0)
            else:
                max_dd_best = 0.0; avg_dd_best = 0.0
            max_intra_best = _max_intratrade_dd_pct_for_mask(mask_best)
            pnl_seq_best = _pnl_seq_pct_for_mask(mask_best)
            text = (
                f"comp_ret: {float(comp_arr[i_best]):.2f}%\n"
                f"thr: {best_thr_local:.3f}\n"
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
            ax1.annotate(text, xy=(best_thr_local, comp_n[i_best]), xycoords='data',
                         xytext=(0.5, 1.04), textcoords='axes fraction',
                         ha='center', va='bottom', fontsize=8,
                         bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    except Exception:
        pass
    plt.tight_layout(rect=[0.0, 0.22, 0.86, 1])
except Exception as ex:
    print(f"! Не удалось построить график перебора порога: {ex}")
# finalize meta
final={"model_state":model.state_dict(),"scaler":ds.scaler,
        "meta":{"seq_len":SEQ_LEN,"pred_window":PRED_WINDOW,"threshold":best_thr}}
torch.save(final, MODEL_PATH)
try:
    with open(MODEL_META_PATH,'w',encoding='utf-8') as mf:
        json.dump({"seq_len":int(SEQ_LEN),"pred_window":int(PRED_WINDOW),"threshold":float(best_thr)}, mf)
except Exception as ex:
    print(f"! Не удалось записать meta-файл {MODEL_META_PATH}: {ex}")

# Пост-обучающий график: нормализованные кривые LR, PR_AUC, PnL%(@thr), ValAcc
try:
    curves = {
        'LR': np.asarray(lr_curve, dtype=np.float64),
        'PR_AUC': np.asarray(pr_auc_curve, dtype=np.float64),
        'PnL%': np.asarray(pnl_curve_pct, dtype=np.float64),
        'ValAcc': np.asarray(val_acc_curve, dtype=np.float64),
    }
    eps = 1e-12
    plt.figure(figsize=(8,5))
    x = np.arange(1, len(lr_curve)+1)
    tab10 = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    colors = {}
    for idx, (name, arr) in enumerate(curves.items()):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            continue
        arr_norm = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + eps)
        line, = plt.plot(
            x[:len(arr_norm)], arr_norm, label=name,
            color=tab10[idx % len(tab10)], linewidth=1.8, alpha=0.95
        )
        colors[name] = line.get_color()
    ax = plt.gca()
    const_text = (
        f"SEQ_LEN={SEQ_LEN}\nPRED_WINDOW={PRED_WINDOW}\nVAL_SPLIT={VAL_SPLIT}\n"
        f"EPOCHS={EPOCHS}\nBATCH={BATCH_SIZE}\nBASE_LR={BASE_LR:.2e}\n"
        f"pct_start={ONECYCLE_PCT_START}\ndiv_factor={ONECYCLE_DIV_FACTOR}\nfinal_div={ONECYCLE_FINAL_DIV_FACTOR}\n"
        f"WD={WEIGHT_DECAY}\nDROPOUT={DROPOUT_P:.3f}\nBEST_LR_MULT={BEST_LR_MULTIPLIER}\nbest_lr={best_lr:.2e}\nGRADCLIP={GRADCLIP_MAXNORM_1_APPLY}\nGRADCLIP_MAXNORM={GRADCLIP_MAXNORM}\nUSE_STANDARD_SCALER=False\nbest_lr_default={best_lr_default:.2e}"
    )
    ax.text(0.98, 0.02, const_text, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
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
    try:
        _script_name = Path(__file__).name
    except Exception:
        _script_name = "price_jump_train_colab_FINDERandOneCycleLR.py"
    plt.gca().text(0.02, 0.02, _script_name, transform=plt.gca().transAxes,
                   ha='left', va='bottom', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5))
    plt.xlabel('Epoch'); plt.ylabel('Normalized scale [0,1]')
    plt.legend(loc='best'); plt.grid(True, alpha=0.3)
    plt.tight_layout();
    from datetime import datetime
    import pytz
    msk = pytz.timezone('Europe/Moscow')
    ts = datetime.now(msk).strftime('%Y%m%d_%H%M')
    out_name = f'training_curves_{ts}.png'
    plt.savefig(out_name, dpi=120)
    print(f"Saved post-training curves to {Path(out_name).resolve()}")
    plt.show()
    plt.close()
except Exception as ex:
    print(f"! Не удалось построить график кривых обучения: {ex}")