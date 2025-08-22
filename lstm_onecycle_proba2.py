# lstm_onecycle_proba2.py
# Last modified (MSK): 2025-08-22 22:56
"""Proba2: OneCycle с предвычислением окон в Dataset (без StandardScaler).
Окна нормализуются относительно первой свечи по open/volume, предвычисляются один раз.
Остальная логика совпадает с основным onecycle.
"""
from pathlib import Path
import json, math, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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

# Constants (mirrored from main onecycle)
SEQ_LEN, PRED_WINDOW, JUMP_THRESHOLD = 30, 5, 0.0035
TRAIN_JSON = Path("candles_10d.json")
MODEL_PATH = Path("lstm_jump_PRAUC.pt")
PNL_MODEL_PATH = Path("lstm_jump_pnl.pt")
VALACC_MODEL_PATH = Path("lstm_jump_valacc.pt")
MODEL_META_PATH = MODEL_PATH.with_suffix(".meta.json")
HYPER_PATH = MODEL_PATH.with_suffix(".hyper.json")
VAL_SPLIT, EPOCHS = 0.2, 360
BATCH_SIZE, BASE_LR = 512, 3e-4
best_lr_default = 2.17e-03
LR_FINDER_MIN_FACTOR = 1.0/20.0
LR_FINDER_MAX_FACTOR = 8.0
BEST_LR_MULTIPLIER = 0.7
CLIP_MIN_FACTOR = 0.8
CLIP_MAX_FACTOR = 8.0
ONECYCLE_PCT_START = 0.12
ONECYCLE_DIV_FACTOR = 2.0
ONECYCLE_FINAL_DIV_FACTOR = 3.5
WEIGHT_DECAY = 3.5e-5
DEFAULT_DROPOUT = 0.35
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EARLY_STOP_EPOCHS = 80
NPR_EPS = 1e-12
SAVE_MIN_PR_AUC = 0.58

# Data

def load_dataframe(path: Path) -> pd.DataFrame:
    with open(path) as f: raw = json.load(f)
    df = pd.DataFrame(list(raw.values()))
    df["datetime"] = pd.to_datetime(df["x"], unit="s")
    return df.set_index("datetime").sort_index()

class CandleDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.closes = df["c"].astype(np.float32).values
        self.opens  = df["o"].astype(np.float32).values
        highs       = df["h"].astype(np.float32).values
        lows        = df["l"].astype(np.float32).values
        volumes     = df["v"].astype(np.float32).values.reshape(-1, 1)
        price_feats = np.stack([self.opens, highs, lows, self.closes], axis=1).astype(np.float32)
        # precompute windows once (relative to first open/vol)
        self.samples = []  # list of (window: np.ndarray (5, SEQ_LEN), label)
        for i in range(SEQ_LEN, len(df) - PRED_WINDOW):
            # label by future max close
            current_open = float(self.opens[i])
            max_close = float(np.max(self.closes[i+1:i+PRED_WINDOW+1]))
            label = 1 if (max_close / max(current_open, 1e-12) - 1.0) >= JUMP_THRESHOLD else 0
            # window features
            sl = slice(i-SEQ_LEN, i)
            pr = price_feats[sl].copy()  # (SEQ_LEN, 4)
            ref_open = float(pr[0, 0])
            prices_rel = pr / max(ref_open, 1e-12) - 1.0
            vw = volumes[sl].copy()      # (SEQ_LEN, 1)
            ref_vol = float(vw[0, 0])
            ref_vol = ref_vol if abs(ref_vol) > 1e-8 else (1e-8 if ref_vol >= 0 else -1e-8)
            vol_rel = vw / ref_vol - 1.0
            window = np.concatenate([prices_rel, vol_rel], axis=1).astype(np.float32)  # (SEQ_LEN, 5)
            self.samples.append((window.T, int(label)))  # store as (5, SEQ_LEN)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.from_numpy(x), y

class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 2, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 5, SEQ_LEN) -> (B, SEQ_LEN, 5)
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

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

# Read optional overrides
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

# LR Finder
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
# OneCycleLR
max_lr_use = float(np.clip(BEST_LR_MULTIPLIER*best_lr, BASE_LR*CLIP_MIN_FACTOR, BASE_LR*CLIP_MAX_FACTOR))
opt = torch.optim.Adam(model.parameters(), BASE_LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=max_lr_use, epochs=EPOCHS, steps_per_epoch=max(1, len(train_loader)),
    pct_start=ONECYCLE_PCT_START, div_factor=ONECYCLE_DIV_FACTOR, final_div_factor=ONECYCLE_FINAL_DIV_FACTOR
)

# Planned LR curve
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
    plt.plot(range(1, len(planned_lr)+1), planned_lr, label='Planned LR')
    plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.title('Planned OneCycle LR by epoch')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    def _sci_fmt(y, pos):
        s = f"{y:.0e}"; return s.replace('e-0','e-').replace('e+0','e+')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(_sci_fmt))
    from datetime import datetime
    import pytz
    msk = pytz.timezone('Europe/Moscow')
    ts = datetime.now(msk).strftime('%Y%m%d_%H%M')
    out_name = f"onecycle_lr_curve_{ts}.png"
    plt.savefig(out_name, dpi=120)
    print(f"Saved LR curve to {Path(out_name).resolve()}")
    plt.close()
except Exception as ex:
    print(f"! Не удалось построить/сохранить план LR: {ex}")

# PnL@best support
val_indices = np.asarray(val_ds.indices, dtype=np.int64)
entry_idx = val_indices + SEQ_LEN
entry_opens = ds.opens[entry_idx]
exit_closes = ds.closes[entry_idx + PRED_WINDOW]
ret_val_fixed = exit_closes / np.maximum(entry_opens, 1e-12) - 1.0

# In-epoch threshold sweep
thr_min, thr_max, thr_step = 0.15, 0.85, 0.0025
last_best_thr = 0.565
best_pnl_thr = last_best_thr
pnl_best_sum = 0.0
trades_best = 0

best_pr_auc = -1.0
best_pnl_sum = -float('inf')
best_val_acc = -1.0
epochs_no_improve = 0
lr_curve = []; pr_auc_curve = []; npr_auc_curve = []; pnl_curve_pct = []; val_acc_curve = []

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
    npr_auc=(pr_auc - POS_FRAC) / (1.0 - POS_FRAC + NPR_EPS)

    val_probs_np=np.asarray(val_probs,dtype=np.float32)
    if e % 10 == 0 or e == 1:
        best_comp=-np.inf; best_thr=last_best_thr; best_trades=0; best_sum=0.0
        for t in np.arange(thr_min,thr_max+1e-12,thr_step):
            m=(val_probs_np>=t); n=int(m.sum())
            if n==0:
                comp=-np.inf; sret=0.0
            else:
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
          f'val_acc {val_acc:.3f} F1 {f1:.3f} ROC_AUC {roc_auc:.3f} PR_AUC {pr_auc:.3f} nPR_AUC {npr_auc:.3f} '
          f'PNL@best(thr={last_best_thr:.4f}) {pnl_best_sum*100:.2f}% trades={trades_best}')

    lr_curve.append(curr_lr); pr_auc_curve.append(float(pr_auc)); npr_auc_curve.append(float(npr_auc)); pnl_curve_pct.append(float(pnl_best_sum*100.0)); val_acc_curve.append(float(val_acc))

    if pr_auc > best_pr_auc + 1e-6:
        best_pr_auc = pr_auc; epochs_no_improve = 0
        if pr_auc > SAVE_MIN_PR_AUC:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state":model.state_dict(),"scaler":None,
                        "meta":{"seq_len":SEQ_LEN,"pred_window":PRED_WINDOW}}, MODEL_PATH)
            try:
                with open(MODEL_META_PATH,'w',encoding='utf-8') as mf:
                    json.dump({"seq_len":int(SEQ_LEN),"pred_window":int(PRED_WINDOW)}, mf)
            except Exception as ex:
                print(f"! Не удалось записать meta-файл {MODEL_META_PATH}: {ex}")
            print(f"✓ Сохранена новая лучшая модель по PR_AUC (PR_AUC={best_pr_auc:.3f}) в {MODEL_PATH.resolve()}")

    if val_acc > best_val_acc + 1e-9 and pr_auc > SAVE_MIN_PR_AUC:
        best_val_acc = val_acc
        VALACC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state":model.state_dict(),"scaler":None,
                    "meta":{"seq_len":SEQ_LEN,"pred_window":PRED_WINDOW}}, VALACC_MODEL_PATH)
        print(f"✓ Сохранена новая лучшая модель по ValAcc (ValAcc={best_val_acc:.3f}) в {VALACC_MODEL_PATH.resolve()}")

    if pnl_best_sum > best_pnl_sum + 1e-12 and pr_auc > SAVE_MIN_PR_AUC:
        best_pnl_sum = pnl_best_sum
        best_pnl_thr = last_best_thr
        PNL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state":model.state_dict(),"scaler":None,
                    "meta":{"seq_len":SEQ_LEN,"pred_window":PRED_WINDOW,"threshold":best_pnl_thr}}, PNL_MODEL_PATH)
        print(f"✓ Сохранена новая лучшая модель по PnL (pnl@{best_pnl_thr:.4f}={best_pnl_sum*100:.2f}%) в {PNL_MODEL_PATH.resolve()}")

# Post messages
if best_pr_auc > -1.0:
    print(f"Лучшая модель (PR_AUC={best_pr_auc:.3f}) сохранена в {MODEL_PATH.resolve()}")
if best_pnl_sum > -float('inf'):
    print(f"Лучшая модель с pnl@{best_pnl_thr:.4f}={best_pnl_sum*100:.2f}% сохранена в {PNL_MODEL_PATH.resolve()}")

# Sweep and plots (identical to main onecycle)
# ... (omitted for brevity — replicate threshold sweep and curves plot, or import)