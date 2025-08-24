# price_jump_train_OneCFocalL.py
# Last modified (MSK): 2025-08-24 12:35
"""OneCycle LSTM training with Focal Loss.
Based on current OneCycle script; integrates Focal Loss for class imbalance.
"""
from pathlib import Path
import json, math, random, time
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
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
try:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass

# Constants copied from OneCycle
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
# LR Finder
LR_FINDER_MIN_FACTOR = 1.0/20.0
LR_FINDER_MAX_FACTOR = 8.0
# OneCycle shape
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
SAVE_MIN_PR_AUC = 0.60

# Focal Loss params
FOCAL_GAMMA = 1.5

# Autotune parameters (triggered once when PR_AUC crosses threshold)
AUTOTUNE_PRAUC_THRESHOLD = 0.601
AUTOTUNE_GAMMA = 1.4
AUTOTUNE_WD_MULT = 1.5

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
        self.samples = []
        for i in range(SEQ_LEN, len(self.closes) - PRED_WINDOW):
            current_open = float(self.opens[i])
            max_close = float(np.max(self.closes[i+1:i+PRED_WINDOW+1]))
            label = 1 if (max_close / max(current_open, 1e-12) - 1.0) >= JUMP_THRESHOLD else 0
            self.samples.append((i, label))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx: int):
        i, y = self.samples[idx]
        x_seq = np.stack([
            self.closes[i-SEQ_LEN:i],
            self.opens[i-SEQ_LEN:i],
            self.highs[i-SEQ_LEN:i],
            self.lows[i-SEQ_LEN:i],
            self.volumes[i-SEQ_LEN:i],
        ], axis=0).astype(np.float32)
        return torch.from_numpy(x_seq), int(y)

class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 2, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

class FocalLossWeightedCE(nn.Module):
    def __init__(self, gamma: float = FOCAL_GAMMA, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits, dim=1)
        # weighted CE per-sample
        ce = torch.nn.functional.nll_loss(logp, targets.to(torch.long), weight=self.class_weights, reduction='none')
        # pt = p_t
        pt = torch.exp(logp[torch.arange(logits.size(0), device=logits.device), targets.to(torch.long)].clamp_min(-50.0))
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()

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

# Optional overrides
DROPOUT_P = DEFAULT_DROPOUT; _got_dropout = False; _got_base_lr = False; _src_dropout = _src_base_lr = "default"
try:
    if HYPER_PATH.exists():
        with open(HYPER_PATH, 'r', encoding='utf-8') as hf: hyper = json.load(hf)
        if isinstance(hyper, dict):
            if 'dropout' in hyper: DROPOUT_P = float(hyper['dropout']); _got_dropout = True; _src_dropout = f"{HYPER_PATH}"
            if 'base_lr' in hyper: BASE_LR = float(hyper['base_lr']); _got_base_lr = True; _src_base_lr = f"{HYPER_PATH}"
    elif MODEL_META_PATH.exists():
        with open(MODEL_META_PATH, 'r', encoding='utf-8') as mf: meta0 = json.load(mf)
        if isinstance(meta0, dict):
            if 'dropout' in meta0: DROPOUT_P = float(meta0['dropout']); _got_dropout = True; _src_dropout = f"{MODEL_META_PATH}"
            if 'base_lr' in meta0: BASE_LR = float(meta0['base_lr']); _got_base_lr = True; _src_base_lr = f"{MODEL_META_PATH}"
except Exception as ex:
    print(f"! Не удалось прочитать hyper/meta: {ex}")
print(f"dropout: {DROPOUT_P:.3f} ({_src_dropout})")
print(f"base_lr: {BASE_LR:.2e} ({_src_base_lr})")

model = LSTMClassifier(dropout=DROPOUT_P).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), BASE_LR)
# class weights from CE (neg/pos)
pos_weight = float(neg_cnt) / max(float(pos_cnt), 1.0)
class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=DEVICE)
lossf = FocalLossWeightedCE(gamma=FOCAL_GAMMA, class_weights=class_weights)

# LR Finder
print("LR Finder: старт…")
min_lr, max_lr = BASE_LR*LR_FINDER_MIN_FACTOR, BASE_LR*LR_FINDER_MAX_FACTOR
finder_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=False)
num_steps = max(1, len(finder_loader))
lr_mult = (max_lr/min_lr) ** (1/num_steps)
print(f"LR Finder params: BASE_LR={BASE_LR:.2e}, min_lr={min_lr:.2e}, max_lr={max_lr:.2e}, steps={num_steps}, lr_mult≈{lr_mult:.6f}")
for pg in opt.param_groups: pg['lr'] = min_lr
best_loss = float('inf'); best_lr = BASE_LR
model.eval();
for xb, yb in finder_loader:
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    opt.zero_grad(); logits = model(xb); loss = lossf(logits, yb)
    loss.backward(); opt.step()
    if loss.item() < best_loss:
        best_loss = loss.item(); best_lr = opt.param_groups[0]['lr']
    for pg in opt.param_groups: pg['lr'] *= lr_mult
print(f"LR Finder: best_lr≈{best_lr:.2e}, best_loss={best_loss:.4f}")
# fallback
best_lr = best_lr_default
print("lr finder is unstable, best_lr=", best_lr)

# OneCycle scheduler
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
        for _ in range(max(1, len(train_loader))): tmp_sched.step()
        planned_lr.append(tmp_opt.param_groups[0]['lr'])
    plt.figure(figsize=(6,3))
    plt.plot(range(1, len(planned_lr)+1), planned_lr, label='Planned LR')
    plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.title('Planned OneCycle LR by epoch')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    def _sci_fmt(y, pos): s = f"{y:.0e}"; return s.replace('e-0','e-').replace('e+0','e+')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(_sci_fmt))
    from datetime import datetime; import pytz
    msk = pytz.timezone('Europe/Moscow'); ts = datetime.now(msk).strftime('%Y%m%d_%H%M')
    out_name = f"onecycle_lr_curve_{ts}.png"; plt.savefig(out_name, dpi=120)
    print(f"Saved LR curve to {Path(out_name).resolve()}"); plt.show(); plt.close()
except Exception as ex:
    print(f"! Не удалось построить/сохранить план LR: {ex}")

# PnL threshold support on validation
val_indices = np.asarray(val_ds.indices, dtype=np.int64)
entry_idx = val_indices + SEQ_LEN
entry_opens = ds.opens[entry_idx]; exit_closes = ds.closes[entry_idx + PRED_WINDOW]
ret_val_fixed = exit_closes / np.maximum(entry_opens, 1e-12) - 1.0
thr_min, thr_max, thr_step = 0.15, 0.85, 0.0025
last_best_thr = 0.565

best_pr_auc = -1.0; best_pnl_sum = -float('inf'); best_val_acc = -1.0

# Buffers for post-training curves
lr_curve = []; pr_auc_curve = []; npr_auc_curve = []; pnl_curve_pct = []; val_acc_curve = []
autotune_done = False
autotune_epoch = None

for e in range(1, EPOCHS+1):
    t0 = time.time()
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
            logits=model(xb.to(DEVICE)); prob1=torch.softmax(logits,dim=1)[:,1].cpu()
            pred=(prob1>=0.5).to(torch.long); y_cpu=yb.to(torch.long)
            corr+=(pred.cpu()==y_cpu).sum().item(); tot_s+=y_cpu.size(0)
            val_targets.extend(y_cpu.tolist()); val_probs.extend(prob1.tolist()); val_preds.extend(pred.cpu().tolist())
    try: roc_auc=roc_auc_score(val_targets,val_probs)
    except Exception: roc_auc=float('nan')
    f1=f1_score(val_targets,val_preds,zero_division=0)
    pr_auc=average_precision_score(val_targets,val_probs)
    npr_auc=(pr_auc - POS_FRAC) / (1.0 - POS_FRAC + NPR_EPS)

    # autotune on threshold hit
    if (not autotune_done) and (pr_auc >= AUTOTUNE_PRAUC_THRESHOLD):
        lossf.gamma = AUTOTUNE_GAMMA
        for pg in opt.param_groups:
            pg['weight_decay'] *= AUTOTUNE_WD_MULT
        wd_now = opt.param_groups[0]['weight_decay']
        print(f"↻ Auto-tune: PR_AUC≥{AUTOTUNE_PRAUC_THRESHOLD:.3f} → gamma={lossf.gamma:.2f}, weight_decay={wd_now:.2e}")
        autotune_done = True
        autotune_epoch = e

    # threshold sweep every 10 epochs
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
            if comp>best_comp: best_comp=comp; best_thr=float(t); best_trades=n; best_sum=sret
        last_best_thr = best_thr; pnl_best_sum = best_sum; trades_best = best_trades
    else:
        m=(val_probs_np>=last_best_thr); trades_best=int(m.sum())
        pnl_best_sum = float(np.sum(ret_val_fixed[m])) if trades_best>0 else 0.0

    curr_lr = opt.param_groups[0]['lr']
    val_acc = (corr/tot_s) if tot_s>0 else 0.0
    dt_s = time.time() - t0
    print(f'Epoch {e}/{EPOCHS} lr {curr_lr:.2e} loss {total_loss/len(train_ds):.4f} '
          f'val_acc {val_acc:.3f} F1 {f1:.3f} ROC_AUC {roc_auc:.3f} PR_AUC {pr_auc:.3f} nPR_AUC {npr_auc:.3f} '
          f'PNL@best(thr={last_best_thr:.4f}) {pnl_best_sum*100:.2f}% trades={trades_best} time {dt_s:.1f}s')

    lr_curve.append(curr_lr); pr_auc_curve.append(float(pr_auc)); npr_auc_curve.append(float(npr_auc)); pnl_curve_pct.append(float(pnl_best_sum*100.0)); val_acc_curve.append(float(val_acc))

    if pr_auc > best_pr_auc + 1e-6:
        best_pr_auc = pr_auc
        if pr_auc > SAVE_MIN_PR_AUC:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state":model.state_dict(),"scaler":ds.scaler,
                        "meta":{"seq_len":SEQ_LEN,"pred_window":PRED_WINDOW}}, MODEL_PATH)
            print(f"✓ Сохранена новая лучшая модель по PR_AUC (PR_AUC={best_pr_auc:.3f}) в {MODEL_PATH.resolve()}")
    if val_acc > best_val_acc + 1e-9 and pr_auc > SAVE_MIN_PR_AUC:
        best_val_acc = val_acc
        VALACC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state":model.state_dict(),"scaler":ds.scaler,
                    "meta":{"seq_len":SEQ_LEN,"pred_window":PRED_WINDOW}}, VALACC_MODEL_PATH)
        print(f"✓ Сохранена новая лучшая модель по ValAcc (ValAcc={best_val_acc:.3f}) в {VALACC_MODEL_PATH.resolve()}")
    if pnl_best_sum > best_pnl_sum + 1e-12 and pr_auc > SAVE_MIN_PR_AUC:
        best_pnl_sum = pnl_best_sum
        best_pnl_thr = last_best_thr
        PNL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state":model.state_dict(),"scaler":ds.scaler,
                    "meta":{"seq_len":SEQ_LEN,"pred_window":PRED_WINDOW,"threshold":best_pnl_thr}}, PNL_MODEL_PATH)
        print(f"✓ Сохранена новая лучшая модель по PnL (pnl@{best_pnl_thr:.4f}={best_pnl_sum*100:.2f}%) в {PNL_MODEL_PATH.resolve()}")

# Post messages
if best_pr_auc > -1.0:
    print(f"Лучшая модель (PR_AUC={best_pr_auc:.3f}) сохранена в {MODEL_PATH.resolve()}")
if best_pnl_sum > -float('inf'):
    print(f"Лучшая модель с pnl@{best_pnl_thr:.4f}={best_pnl_sum*100:.2f}% сохранена в {PNL_MODEL_PATH.resolve()}")

# Final training curves (normalized): LR, PR_AUC, PnL%(@thr), ValAcc
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
    colors = {}
    for name, arr in curves.items():
        arr = np.asarray(arr, dtype=np.float64)
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
    # mark autotune epoch with dashed vertical line
    if autotune_epoch is not None:
        plt.axvline(autotune_epoch, color='#999999', linestyle='--', linewidth=1.0, alpha=0.7)
    const_text = (
        f"SEQ_LEN={SEQ_LEN}\nPRED_WINDOW={PRED_WINDOW}\nVAL_SPLIT={VAL_SPLIT}\n"
        f"EPOCHS={EPOCHS}\nBATCH={BATCH_SIZE}\nBASE_LR={BASE_LR:.2e}\n"
        f"pct_start={ONECYCLE_PCT_START}\ndiv_factor={ONECYCLE_DIV_FACTOR}\nfinal_div={ONECYCLE_FINAL_DIV_FACTOR}\n"
        f"WD={WEIGHT_DECAY}\nDROPOUT={DEFAULT_DROPOUT:.3f}\nBEST_LR_MULT={BEST_LR_MULTIPLIER}"
        f"\nauto_thr={AUTOTUNE_PRAUC_THRESHOLD}\nauto_gamma={AUTOTUNE_GAMMA}\nauto_WD×{AUTOTUNE_WD_MULT}"
    )
    plt.gca().text(0.98, 0.02, const_text, transform=plt.gca().transAxes,
                   ha='right', va='bottom', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    # script filename at bottom-left
    try:
        _script_name = Path(__file__).name
    except Exception:
        _script_name = "price_jump_train_OneCFocalL.py"
    plt.gca().text(0.02, 0.02, _script_name, transform=plt.gca().transAxes,
                   ha='left', va='bottom', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5))
    plt.xlabel('Epoch'); plt.ylabel('Normalized scale [0,1]'); plt.legend(loc='best'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    from datetime import datetime; import pytz
    msk = pytz.timezone('Europe/Moscow'); ts = datetime.now(msk).strftime('%Y%m%d_%H%M')
    out_name = f'training_curves_{ts}.png'; plt.savefig(out_name, dpi=120)
    print(f"Saved post-training curves to {Path(out_name).resolve()}"); plt.show(); plt.close()
except Exception as ex:
    print(f"! Не удалось построить график кривых обучения: {ex}")

# Threshold sweep on validation (post-training)
print("Подбираем порог по PnL на валидационном наборе…")
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"]); model.to(DEVICE).eval()
val_targets_all=[]; val_probs_all=[]; val_preds_all=[]
with torch.no_grad():
    for xb,yb in val_loader:
        logits=model(xb.to(DEVICE)); prob1=torch.softmax(logits,dim=1)[:,1].cpu()
        pred=(prob1>=0.5).to(torch.long); y_cpu=yb.to(torch.long)
        val_targets_all.extend(y_cpu.tolist()); val_probs_all.extend(prob1.tolist()); val_preds_all.extend(pred.cpu().tolist())
ret_val = exit_closes/np.maximum(entry_opens,1e-12)-1.0
thr_min,thr_max,thr_step=0.15,0.85,0.0025
print(f"Перебор порога по PnL (валидация): min={thr_min:.3f}, max={thr_max:.3f}, step={thr_step:.4f}")
thresholds=np.arange(thr_min,thr_max+1e-12,thr_step)
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

try:
    fig, ax1 = plt.subplots(figsize=(8,5)); ax2 = ax1.twinx()
    ax1.plot(thr_list, pnl_list, label='PnL%', color='#d62728')
    ax1.plot(thr_list, comp_list, label='CompRet%', color='#1f77b4')
    ax2.plot(thr_list, sharpe_list, label='Sharpe', color='#9467bd')
    ax3 = ax1.twinx(); ax3.get_yaxis().set_visible(False)
    ax3.plot(thr_list, trades_list, label='Trades', color='#8c564b')
    if np.isfinite(best_comp):
        ax1.scatter([best_thr], [best_comp*100.0], color='#1f77b4', s=30)
        ax1.annotate(f"max CompRet={best_comp*100:.2f}%\n(thr={best_thr:.4f})",
                     xy=(best_thr, best_comp*100.0), xytext=(6, 12), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))
    lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels(); lines3, labels3 = ax3.get_legend_handles_labels()
    lines = lines1 + lines2 + lines3; labels = labels1 + labels2 + labels3
    ax1.legend(lines, labels, loc='best'); ax1.grid(True, alpha=0.3)
    const_text = (f"SEQ_LEN={SEQ_LEN}\nPRED_WINDOW={PRED_WINDOW}\nVAL_SPLIT={VAL_SPLIT}\n"
                  f"EPOCHS={EPOCHS}\nBATCH={BATCH_SIZE}\nBASE_LR={BASE_LR:.2e}\n"
                  f"pct_start={ONECYCLE_PCT_START}\ndiv_factor={ONECYCLE_DIV_FACTOR}\nfinal_div={ONECYCLE_FINAL_DIV_FACTOR}\n"
                  f"WD={WEIGHT_DECAY}\nDROPOUT={DEFAULT_DROPOUT:.3f}\nBEST_LR_MULT={BEST_LR_MULTIPLIER}")
    ax1.text(0.98, 0.02, const_text, transform=ax1.transAxes, ha='right', va='bottom', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    fig.tight_layout(); from datetime import datetime; import pytz
    msk = pytz.timezone('Europe/Moscow'); ts = datetime.now(msk).strftime('%Y%m%d_%H%M')
    out_name = f'threshold_sweep_{ts}.png'; fig.savefig(out_name, dpi=130)
    print(f"Saved threshold sweep plot to {Path(out_name).resolve()}"); plt.show()
except Exception as ex:
    print(f"! Не удалось построить график перебора порога: {ex}")