# price_jump_train_colab_FINDERandOneCycleLR.py
# Last modified (MSK): 2025-08-20 10:52
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader, random_split

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
TRAIN_JSON = Path("candles_10d.json")
MODEL_PATH = Path("lstm_jump.pt")
PNL_MODEL_PATH = Path("lstm_jump_pnl.pt")
MODEL_META_PATH = MODEL_PATH.with_suffix(".meta.json")
HYPER_PATH = MODEL_PATH.with_suffix(".hyper.json")
VAL_SPLIT, EPOCHS = 0.2, 130
BATCH_SIZE, BASE_LR = 512, 3e-4
best_lr_default = 3.0e-03
# Tunable LR Finder params
LR_FINDER_MIN_FACTOR = 1.0/20.0  # min_lr = BASE_LR * LR_FINDER_MIN_FACTOR
LR_FINDER_MAX_FACTOR = 8.0       # max_lr = BASE_LR * LR_FINDER_MAX_FACTOR
# How to pick OneCycle max_lr from best_lr and clip range around BASE_LR
BEST_LR_MULTIPLIER = 0.9         # max_lr ~ BEST_LR_MULTIPLIER * best_lr
CLIP_MIN_FACTOR = 0.8            # clip lower bound = BASE_LR * CLIP_MIN_FACTOR
CLIP_MAX_FACTOR = 8.0            # clip upper bound = BASE_LR * CLIP_MAX_FACTOR
# OneCycleLR shape parameters
ONECYCLE_PCT_START = 0.38
ONECYCLE_DIV_FACTOR = 30.0
ONECYCLE_FINAL_DIV_FACTOR = 30
WEIGHT_DECAY = 3e-5
# Default dropout if no hyper/meta provided
DEFAULT_DROPOUT = 0.24
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EARLY_STOP_EPOCHS = 25

class CandleDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.closes = df["c"].astype(np.float32).values
        self.opens  = df["o"].astype(np.float32).values
        price_feats = df[["o","h","l","c"]].astype(np.float32).values
        volumes     = df["v"].astype(np.float32).values.reshape(-1, 1)
        raw_windows, labels = [], []
        for i in range(SEQ_LEN, len(df) - PRED_WINDOW):
            current_open = self.opens[i]
            max_close    = self.closes[i+1:i+PRED_WINDOW+1].max()
            labels.append(1 if (max_close/current_open - 1) >= JUMP_THRESHOLD else 0)
            sl = slice(i-SEQ_LEN+1, i+1)
            pr = price_feats[sl].copy(); ref_open = pr[0,0]
            prices_rel = pr/ref_open - 1.0
            vw = volumes[sl].copy(); ref_vol = max(float(vw[0,0]), 1e-8)
            vol_rel = vw/ref_vol - 1.0
            raw_windows.append(np.concatenate([prices_rel, vol_rel], axis=1))
        all_rows = np.vstack(raw_windows)
        self.scaler = StandardScaler().fit(all_rows)
        self.samples = [(self.scaler.transform(w), y) for w,y in zip(raw_windows, labels)]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x,y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

def load_dataframe(path: Path) -> pd.DataFrame:
    with open(path) as f: raw = json.load(f)
    df = pd.DataFrame(list(raw.values()))
    df["datetime"] = pd.to_datetime(df["x"], unit="s")
    return df.set_index("datetime").sort_index()

class LSTMClassifier(nn.Module):
    def __init__(self, nfeat: int = 5, hidden: int = 64, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(nfeat, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 2)
    def forward(self, x):
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
# LR Finder uses a deterministic, no-shuffle loader and disables dropout
finder_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=False)
num_steps = max(1, len(finder_loader))
lr_mult = (max_lr/min_lr) ** (1/num_steps)
print(f"LR Finder params: BASE_LR={BASE_LR:.2e}, min_lr={min_lr:.2e}, max_lr={max_lr:.2e}, num_steps={num_steps}, lr_mult≈{lr_mult:.6f}")
for pg in opt.param_groups: pg['lr'] = min_lr
best_loss = float('inf'); best_lr = BASE_LR
model.eval(); step_id=0  # disable dropout for LR Finder
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

# switch back to train mode for main loop
model.train()

# OneCycleLR на весь ран: max_lr = BEST_LR_MULTIPLIER×best_lr (clipped)
max_lr = float(np.clip(BEST_LR_MULTIPLIER*best_lr, BASE_LR*CLIP_MIN_FACTOR, BASE_LR*CLIP_MAX_FACTOR))
opt = torch.optim.Adam(model.parameters(), BASE_LR, weight_decay=WEIGHT_DECAY)
pct_start=ONECYCLE_PCT_START; div_factor=ONECYCLE_DIV_FACTOR; final_div_factor=ONECYCLE_FINAL_DIV_FACTOR; weight_decay=WEIGHT_DECAY
print(f"OneCycleLR params: epochs={EPOCHS}, steps_per_epoch={len(train_loader)}, BASE_LR={BASE_LR:.2e}, max_lr={max_lr:.2e}, pct_start={pct_start}, div_factor={div_factor}, final_div_factor={final_div_factor}, weight_decay={weight_decay}")
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=max_lr, epochs=EPOCHS, steps_per_epoch=len(train_loader),
    pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div_factor
)

# До начала обучения: построим и выведем планируемую кривую LR по эпохам (OneCycle)
try:
    tmp_opt = torch.optim.Adam(model.parameters(), BASE_LR, weight_decay=WEIGHT_DECAY)
    tmp_sched = torch.optim.lr_scheduler.OneCycleLR(
        tmp_opt, max_lr=max_lr, epochs=EPOCHS, steps_per_epoch=len(train_loader),
        pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div_factor
    )
    planned_lr = []
    for _ep in range(EPOCHS):
        for _ in range(len(train_loader)):
            tmp_sched.step()
        planned_lr.append(tmp_opt.param_groups[0]['lr'])
    plt.figure(figsize=(6,3))
    plt.plot(range(1, len(planned_lr)+1), planned_lr, label='Planned LR')
    plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.title('Planned OneCycle LR by epoch')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    # format y-axis as scientific like 5e-4 instead of 0.0005
    def _sci_fmt(y, pos):
        s = f"{y:.0e}"
        return s.replace('e-0','e-').replace('e+0','e+')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(_sci_fmt))
    plt.savefig('onecycle_lr_curve.png', dpi=120)
    print("Saved LR curve to onecycle_lr_curve.png (planned)")
    try:
        from IPython.display import Image, display
        display(Image('onecycle_lr_curve.png'))
    except Exception:
        pass
    plt.close()
except Exception as ex:
    print(f"! Не удалось построить/сохранить дообучающий график LR: {ex}")

# PnL@best threshold support
thr_min,thr_max,thr_step=0.15,0.60,0.0025
last_best_thr = 0.565
best_pnl_thr = last_best_thr

# PnL@0.565 support
val_indices = np.asarray(val_ds.indices, dtype=np.int64)
entry_idx = val_indices + SEQ_LEN
entry_opens = ds.opens[entry_idx]
exit_closes = ds.closes[entry_idx + PRED_WINDOW]
ret_val_fixed = exit_closes / np.maximum(entry_opens, 1e-12) - 1.0

best_pr_auc = -1.0
best_pnl_sum = -float('inf')
epochs_no_improve = 0
# Lists to collect per-epoch metrics for post-training plots
lr_curve = []
pr_auc_curve = []
pnl_curve_pct = []
val_acc_curve = []
for e in range(1, EPOCHS+1):
    model.train(); total_loss=0.0
    for xb,yb in train_loader:
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(); logits=model(xb); loss=lossf(logits,yb)
        loss.backward(); opt.step(); sched.step()
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

    # compute best-threshold PnL on validation every 5 epochs
    val_probs_np=np.asarray(val_probs,dtype=np.float32)
    y_true_np = np.asarray(val_targets, dtype=np.int32)
    if e % 5 == 0 or e == 1:
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
    print(f'Epoch {e}/{EPOCHS} lr {curr_lr:.2e} loss {total_loss/len(train_ds):.4f} '
          f'val_acc {val_acc:.3f} F1 {f1:.3f} ROC_AUC {roc_auc:.3f} PR_AUC {pr_auc:.3f} '
          f'PNL@best(thr={last_best_thr:.4f}) {pnl_best_sum*100:.2f}% trades={trades_best}')

    # collect curves
    lr_curve.append(curr_lr)
    pr_auc_curve.append(float(pr_auc))
    pnl_curve_pct.append(float(pnl_best_sum*100.0))
    val_acc_curve.append(float(val_acc))

    if pr_auc > best_pr_auc + 1e-6:
        best_pr_auc = pr_auc; epochs_no_improve = 0
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state":model.state_dict(),"scaler":ds.scaler,
                    "meta":{"seq_len":SEQ_LEN,"pred_window":PRED_WINDOW}}, MODEL_PATH)
        try:
            with open(MODEL_META_PATH,'w',encoding='utf-8') as mf:
                json.dump({"seq_len":int(SEQ_LEN),"pred_window":int(PRED_WINDOW)}, mf)
        except Exception as ex:
            print(f"! Не удалось записать meta-файл {MODEL_META_PATH}: {ex}")
        print(f"✓ Сохранена новая лучшая модель (PR_AUC={best_pr_auc:.3f}) в {MODEL_PATH.resolve()}")
    
    # save best-by-PnL (using best-threshold PnL sum)
    if pnl_best_sum > best_pnl_sum + 1e-12:
        best_pnl_sum = pnl_best_sum
        best_pnl_thr = last_best_thr
        PNL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state":model.state_dict(),"scaler":ds.scaler,
                    "meta":{"seq_len":SEQ_LEN,"pred_window":PRED_WINDOW}}, PNL_MODEL_PATH)
        print(f"✓ Сохранена новая лучшая модель (PNL@{last_best_thr:.4f}={best_pnl_sum*100:.2f}%) в {PNL_MODEL_PATH.resolve()}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_EPOCHS:
            print(f"⏹ Ранний стоп: PR AUC не улучшается {epochs_no_improve} эпох подряд"); break

print(f"Лучшая модель с PR_AUC={best_pr_auc:.3f} сохранена в {MODEL_PATH.resolve()}")
# Печать финального статуса лучшей по PnL модели
if best_pnl_sum > -float('inf'):
    print(f"Лучшая модель с pnl@{best_pnl_thr:.4f}={best_pnl_sum*100:.2f}% сохранена в {PNL_MODEL_PATH.resolve()}")

# PnL threshold sweep (0.15..0.60 step 0.0025)
print("Подбираем порог по PnL на валидационном наборе…")
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"]); model.to(DEVICE).eval()
val_probs_all = np.zeros(len(val_ds), dtype=np.float32)
y_true_all = np.zeros(len(val_ds), dtype=np.int32)
with torch.no_grad():
    ptr=0
    for xb,_yb in DataLoader(val_ds,batch_size=BATCH_SIZE):
        logits=model(xb.to(DEVICE)); prob1=torch.softmax(logits,dim=1)[:,1].cpu().numpy()
        n = len(prob1)
        val_probs_all[ptr:ptr+n] = prob1
        y_true_all[ptr:ptr+n] = _yb.cpu().numpy()
        ptr += n
entry_opens = ds.opens[entry_idx]; exit_closes = ds.closes[entry_idx+PRED_WINDOW]
ret_val = exit_closes/np.maximum(entry_opens,1e-12)-1.0
thr_min,thr_max,thr_step=0.15,0.60,0.0025
print(f"Перебор порога по PnL (валидация): min={thr_min:.3f}, max={thr_max:.3f}, step={thr_step:.4f}")
thresholds=np.arange(thr_min,thr_max+1e-12,thr_step)
# collect metrics for plotting
thr_list=[]; pnl_list=[]; comp_list=[]; sharpe_list=[]; trades_list=[]
def _safe_sharpe_arr(r: np.ndarray) -> float:
    if r.size < 2: return 0.0
    std = float(np.std(r))
    return float(np.mean(r) / (std + 1e-12))
best_comp=-np.inf; best_thr=float(thresholds[0]); best_trades=0
for t in thresholds:
    m=(val_probs_all>=t); n=int(m.sum())
    if n==0:
        comp=-np.inf; shp=0.0; sret=0.0
    else:
        r=ret_val[m]
        comp=-1.0 if np.any(r<=-0.999999) else float(np.exp(np.sum(np.log1p(r)))-1.0)
        shp=_safe_sharpe_arr(r)
        sret=float(np.sum(r))
    thr_list.append(float(t)); pnl_list.append(sret*100.0); comp_list.append(comp*100.0 if np.isfinite(comp) else np.nan); sharpe_list.append(shp); trades_list.append(n)
    if comp>best_comp: best_comp=comp; best_thr=float(t); best_trades=n
print(f"Выбран порог по PnL (валидация): {best_thr:.4f}, comp_ret={best_comp*100 if np.isfinite(best_comp) else float('nan'):.2f}% trades={best_trades}")
# plot metrics vs threshold with max comp_ret annotated
try:
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax2 = ax1.twinx()
    thr_arr = np.asarray(thr_list)
    pnl_arr = np.asarray(pnl_list)
    comp_arr = np.asarray(comp_list)
    shp_arr = np.asarray(sharpe_list)
    l1, = ax1.plot(thr_arr, comp_arr, label='comp_ret %', color='#1f77b4')
    l2, = ax1.plot(thr_arr, pnl_arr, label='pnl_sum %', color='#ff7f0e')
    l3, = ax2.plot(thr_arr, shp_arr, label='sharpe', color='#2ca02c', alpha=0.8)
    # annotate best comp_ret
    if np.isfinite(best_comp):
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
    # filename with MSK datetime
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
    # plot normalized curves
    colors = {}
    for name, arr in curves.items():
        if arr.size == 0:
            continue
        arr_norm = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + eps)
        line, = plt.plot(x, arr_norm, label=name)
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
    # textbox with top constants in bottom-right
    const_text = (
        f"SEQ_LEN={SEQ_LEN}\nPRED_WINDOW={PRED_WINDOW}\nVAL_SPLIT={VAL_SPLIT}\n"
        f"EPOCHS={EPOCHS}\nBATCH={BATCH_SIZE}\nBASE_LR={BASE_LR:.2e}\n"
        f"pct_start={ONECYCLE_PCT_START}\ndiv_factor={ONECYCLE_DIV_FACTOR}\nfinal_div={ONECYCLE_FINAL_DIV_FACTOR}\n"
        f"WD={WEIGHT_DECAY}\nDROPOUT={DROPOUT_P:.3f}\nBEST_LR_MULT={BEST_LR_MULTIPLIER}\n"
        f"best_lr={best_lr:.2e}"
    )
    plt.gca().text(0.98, 0.02, const_text, transform=plt.gca().transAxes,
                   ha='right', va='bottom', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    plt.xlabel('Epoch'); plt.ylabel('Normalized scale [0,1]')
    plt.title('Training curves (normalized): LR, PR_AUC, PnL%@thr, ValAcc')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    # filename with MSK datetime
    from datetime import datetime
    import pytz
    msk = pytz.timezone('Europe/Moscow')
    ts = datetime.now(msk).strftime('%Y%m%d_%H%M')
    out_name = f'training_curves_{ts}.png'
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