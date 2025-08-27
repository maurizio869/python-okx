# price_jump_train_OneCFocalL.py
# Last modified (MSK): 2025-08-27 20:46
# Changes:
# - Add Max IntraTrade DD (price, %) and PnL (seq, %) metrics on threshold
# - Extend max CompRet annotation with new metrics (real values)
# - Move threshold constants block outside axes on the right; legend stays bottom
# - Increase threshold figure height and bottom padding to preserve plot proportions
# - Fix threshold bug: use NumPy array for val_probs_all comparisons (masks, avg_dd, mask_best)
# - Params: BEST_LR_MULTIPLIER=2.0; ONECYCLE_FINAL_DIV_FACTOR=7.5; WEIGHT_DECAY=4.5e-5; EPOCHS=400; add 2 new features (upper_wick/body, lower_wick/body) and set input_size=7
# - Add avg_dd (seq, price) line + rectangles; add pnl_ddd metric; double figure size; improve rectangle anti-overlap; constants bottom aligned with x-axis; refine pnl_ddd exit (close-open current < close-open prev, prev green, +0.25%)
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
from matplotlib.offsetbox import AnnotationBbox, TextArea
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
VAL_SPLIT, EPOCHS = 0.2, 400
BATCH_SIZE, BASE_LR = 512, 3e-4
best_lr_default = 6.17e-03
# LR Finder
LR_FINDER_MIN_FACTOR = 1.0/20.0
LR_FINDER_MAX_FACTOR = 8.0
# OneCycle shape
BEST_LR_MULTIPLIER = 2.0
CLIP_MIN_FACTOR = 0.8
CLIP_MAX_FACTOR = 8.0
ONECYCLE_PCT_START = 0.12
ONECYCLE_DIV_FACTOR = 2.0
ONECYCLE_FINAL_DIV_FACTOR = 7.5
WEIGHT_DECAY = 4.5e-5
DEFAULT_DROPOUT = 0.35
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EARLY_STOP_EPOCHS = 80
NPR_EPS = 1e-12
SAVE_MIN_PR_AUC = 0.60
GRADCLIP_MAXNORM_1_APPLY = True
GRADCLIP_MAXNORM = 0.8
USE_STANDARD_SCALER = False
USE_EARLY_STOP = False

# PnL_DDD parameters (sequential, dynamic exit)
PNL_DDD_THRESH_PCT = 0.0025   # +0.25% above entry open
PNL_DDD_STOP_LOSS_PCT = -0.002  # -0.20% stop vs entry open
PNL_DDD_MAX_HOLD_MIN = 10      # fallback hold minutes if no early exit

# Focal Loss params
FOCAL_GAMMA = 1.5

# Autotune parameters (triggered once when PR_AUC crosses threshold)
AUTOTUNE_PRAUC_THRESHOLD = 0.601
AUTOTUNE_GAMMA = 1.4
AUTOTUNE_WD_MULT = 1.5
AUTOTUNE_BETA1 = 0.8
AUTOTUNE_APPLY_BETA = True

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
        self.scaler = None
        self.use_scaler = False
        self.samples = []
        for i in range(SEQ_LEN, len(self.closes) - PRED_WINDOW):
            current_open = float(self.opens[i])
            max_close = float(np.max(self.closes[i+1:i+PRED_WINDOW+1]))
            label = 1 if (max_close / max(current_open, 1e-12) - 1.0) >= JUMP_THRESHOLD else 0
            self.samples.append((i, label))

    def fit_scaler_on_indices(self, sample_indices: list[int]) -> None:
        try:
            windows = []
            for sample_idx in sample_indices:
                i, _ = self.samples[sample_idx]
                closes_w = self.closes[i-SEQ_LEN:i]
                opens_w  = self.opens[i-SEQ_LEN:i]
                highs_w  = self.highs[i-SEQ_LEN:i]
                lows_w   = self.lows[i-SEQ_LEN:i]
                vols_w   = self.volumes[i-SEQ_LEN:i]
                body_w   = np.abs(closes_w - opens_w) + 1e-12
                upper_w  = np.clip(highs_w - np.maximum(opens_w, closes_w), 0.0, None)
                lower_w  = np.clip(np.minimum(opens_w, closes_w) - lows_w, 0.0, None)
                ratio_up = upper_w / body_w
                ratio_dn = lower_w / body_w
                x_seq_t = np.stack([
                    closes_w, opens_w, highs_w, lows_w, vols_w, ratio_up, ratio_dn
                ], axis=1)  # shape (SEQ_LEN, 7)
                windows.append(x_seq_t)
            if len(windows) > 0:
                feats = np.concatenate(windows, axis=0)  # (N*SEQ_LEN, 7)
                self.scaler = StandardScaler().fit(feats)
                self.use_scaler = True
        except Exception:
            self.scaler = None
            self.use_scaler = False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        i, y = self.samples[idx]
        closes_w = self.closes[i-SEQ_LEN:i]
        opens_w  = self.opens[i-SEQ_LEN:i]
        highs_w  = self.highs[i-SEQ_LEN:i]
        lows_w   = self.lows[i-SEQ_LEN:i]
        vols_w   = self.volumes[i-SEQ_LEN:i]
        body_w   = np.abs(closes_w - opens_w) + 1e-12
        upper_w  = np.clip(highs_w - np.maximum(opens_w, closes_w), 0.0, None)
        lower_w  = np.clip(np.minimum(opens_w, closes_w) - lows_w, 0.0, None)
        ratio_up = upper_w / body_w
        ratio_dn = lower_w / body_w
        x_seq = np.stack([
            closes_w,
            opens_w,
            highs_w,
            lows_w,
            vols_w,
            ratio_up,
            ratio_dn,
        ], axis=0).astype(np.float32)  # (7, SEQ_LEN)
        if self.use_scaler and self.scaler is not None:
            x_seq = self.scaler.transform(x_seq.T).T.astype(np.float32)
        return torch.from_numpy(x_seq), int(y)

class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 2, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=hidden_size, num_layers=num_layers,
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
if USE_STANDARD_SCALER:
    try:
        ds.fit_scaler_on_indices(val_ds.indices if False else train_ds.indices)
        print("StandardScaler: fitted on train windows")
    except Exception as ex:
        print(f"! StandardScaler fit failed: {ex}")
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
thr_min, thr_max, thr_step = 0.15, 0.99, 0.0025
last_best_thr = 0.565

best_pr_auc = -1.0; best_pnl_sum = -float('inf'); best_val_acc = -1.0

# Buffers for post-training curves
lr_curve = []; pr_auc_curve = []; npr_auc_curve = []; pnl_curve_pct = []; val_acc_curve = []
autotune_done = False
autotune_epoch = None
no_improve_epochs = 0

for e in range(1, EPOCHS+1):
    t0 = time.time()
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
        if AUTOTUNE_APPLY_BETA:
            try:
                beta2 = pg.get('betas', (0.9, 0.999))[1]
                pg['betas'] = (AUTOTUNE_BETA1, beta2)
            except Exception:
                pass
        wd_now = opt.param_groups[0]['weight_decay']
        betas_now = None
        try:
            betas_now = opt.param_groups[0]['betas']
        except Exception:
            pass
        print(f"↻ Auto-tune: PR_AUC≥{AUTOTUNE_PRAUC_THRESHOLD:.3f} → gamma={lossf.gamma:.2f}, weight_decay={wd_now:.2e}, betas={betas_now}")
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
            # update best inside the loop
            if comp>best_comp:
                best_comp=comp; best_thr=float(t); best_trades=n; best_sum=sret
        # after loop, set best metrics
        last_best_thr = best_thr
        trades_best = best_trades
        pnl_best_sum = best_sum
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

    # metrics computed: roc_auc, f1, pr_auc, npr_auc

    improved = False
    if pr_auc > best_pr_auc + 1e-6:
        best_pr_auc = pr_auc
        improved = True
        no_improve_epochs = 0
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

    if not improved:
        no_improve_epochs += 1
    if USE_EARLY_STOP and no_improve_epochs >= EARLY_STOP_EPOCHS:
        print(f"Early stop: no PR_AUC improvement for {EARLY_STOP_EPOCHS} epochs")
        break

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
    # unique colors, no markers per requirement
    tab10 = [
        '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
        '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'
    ]
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
    # annotate max PR_AUC and max PnL above axes, avoid overlap
    pr_ann = None; pnl_ann = None
    xlen = max(1, len(lr_curve))
    # vertical dashed line for autotune epoch
    try:
        if autotune_epoch is not None:
            ax.axvline(autotune_epoch, color='#999999', linestyle='--', linewidth=1.0, alpha=0.7)
    except Exception:
        pass
    if len(pr_auc_curve) > 0:
        i_best_pr = int(np.nanargmax(pr_auc_curve))
        y_best_pr = (pr_auc_curve[i_best_pr] - np.nanmin(pr_auc_curve)) / (np.nanmax(pr_auc_curve) - np.nanmin(pr_auc_curve) + eps)
        x_frac_pr = (i_best_pr + 1) / xlen
        # show point on curve
        plt.scatter([i_best_pr+1], [y_best_pr], color=colors.get('PR_AUC', '#2ca02c'), s=32)
        pr_ann = ax.annotate(
            f"max PR_AUC={pr_auc_curve[i_best_pr]:.3f} (ep={i_best_pr+1})",
            xy=(i_best_pr+1, y_best_pr), xycoords='data',
            xytext=(x_frac_pr, 1.06), textcoords='axes fraction',
            ha='center', va='bottom', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.8))
    if len(pnl_curve_pct) > 0:
        i_best_pnl = int(np.nanargmax(pnl_curve_pct))
        y_best_pnl = (pnl_curve_pct[i_best_pnl] - np.nanmin(pnl_curve_pct)) / (np.nanmax(pnl_curve_pct) - np.nanmin(pnl_curve_pct) + eps)
        x_frac_pnl = (i_best_pnl + 1) / xlen
        # show point on curve
        plt.scatter([i_best_pnl+1], [y_best_pnl], color=colors.get('PnL%', '#d62728'), s=32)
        pnl_ann = ax.annotate(
            f"max PnL={pnl_curve_pct[i_best_pnl]:.2f}% (ep={i_best_pnl+1})",
            xy=(i_best_pnl+1, y_best_pnl), xycoords='data',
            xytext=(x_frac_pnl, 1.12), textcoords='axes fraction',
            ha='center', va='bottom', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.8))
    # simple collision avoidance: if texts too close in x, shift left/right
    try:
        if pr_ann is not None and pnl_ann is not None:
            (xpr, ypr) = pr_ann.get_position()
            (xpn, ypn) = pnl_ann.get_position()
            if abs(xpr - xpn) < 0.08:
                pr_ann.set_position((xpr - 0.06, ypr))
                pnl_ann.set_position((xpn + 0.06, ypn))
    except Exception:
        pass
    # constants box (inside axes, bottom-right)
    const_text = (
        f"SEQ_LEN={SEQ_LEN}\nPRED_WINDOW={PRED_WINDOW}\nVAL_SPLIT={VAL_SPLIT}\n"
        f"EPOCHS={EPOCHS}\nBATCH={BATCH_SIZE}\nBASE_LR={BASE_LR:.2e}\n"
        f"pct_start={ONECYCLE_PCT_START}\ndiv_factor={ONECYCLE_DIV_FACTOR}\nfinal_div={ONECYCLE_FINAL_DIV_FACTOR}\n"
        f"WD={WEIGHT_DECAY}\nDROPOUT={DEFAULT_DROPOUT:.3f}\nBEST_LR_MULT={BEST_LR_MULTIPLIER}"
        f"\nauto_thr={AUTOTUNE_PRAUC_THRESHOLD}\nauto_gamma={AUTOTUNE_GAMMA}\nauto_WD×{AUTOTUNE_WD_MULT}"
        f"\nauto_beta1={AUTOTUNE_BETA1}\nAPPLY_BETA={AUTOTUNE_APPLY_BETA}"
        f"\nUSE_STANDARD_SCALER={USE_STANDARD_SCALER}\nGRADCLIP={GRADCLIP_MAXNORM_1_APPLY}\nGRADCLIP_MAXNORM={GRADCLIP_MAXNORM}\nbest_lr_default={best_lr_default:.2e}"
    )
    ax.text(0.98, 0.02, const_text, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    # legend strictly to the left of constants box (bottom-right area)
    leg = plt.legend(loc='lower right', bbox_to_anchor=(0.80, 0.02))
    # post-draw realignment: right edge of legend ≈ left edge of constants − margin
    try:
        fig = plt.gcf(); fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        const_bb = ax.texts[-1].get_window_extent(renderer=renderer)
        leg_bb = leg.get_window_extent(renderer=renderer)
        # convert const left to axes coords
        const_left_axes = ax.transAxes.inverted().transform((const_bb.x0, const_bb.y0))[0]
        # set legend anchor so that legend right aligns to const_left_axes - margin
        margin = 0.01
        new_x = max(0.02, const_left_axes - margin)
        leg.set_bbox_to_anchor((new_x, 0.02), transform=ax.transAxes)
    except Exception:
        pass
    try:
        _script_name = Path(__file__).name
    except Exception:
        _script_name = "price_jump_train_OneCFocalL.py"
    ax.text(0.02, 0.02, _script_name, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5))
    plt.xlabel('Epoch'); plt.ylabel('Normalized scale [0,1]'); plt.grid(True, alpha=0.3)
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
# ensure numpy array for threshold masking
val_probs_all_np = np.asarray(val_probs_all, dtype=np.float32)
thr_min,thr_max,thr_step=0.15,0.99,0.0025
print(f"Перебор порога по PnL (валидация): min={thr_min:.3f}, max={thr_max:.3f}, step={thr_step:.4f}")

thresholds=np.arange(thr_min,thr_max+1e-12,thr_step)
thr_list=[]; pnl_list=[]; comp_list=[]; sharpe_list=[]; trades_list=[]; mean_ret_list=[]; median_ret_list=[]; mdd_list=[]

max_intra_dd_list=[]; pnl_seq_list=[]

def _safe_sharpe_arr(r: np.ndarray) -> float:
    if r.size < 2: return 0.0
    std = float(np.std(r))
    return float(np.mean(r) / (std + 1e-12))

best_comp=-np.inf; best_thr=float(thresholds[0]); best_trades=0
for t in thresholds:
    m=(val_probs_all_np>=t); n=int(m.sum())
    if n==0:
        comp=-np.inf; shp=0.0; sret=0.0
    else:
        r=ret_val[m]
        comp=-1.0 if np.any(r<=-0.999999) else float(np.exp(np.sum(np.log1p(r)))-1.0)

        shp=_safe_sharpe_arr(r)
        sret=float(np.sum(r))
        meanp = float(np.mean(r)*100.0)
        medp  = float(np.median(r)*100.0)
        ent = entry_idx[m]
        order = np.argsort(ent)
        r_sorted = r[order]
        equity = np.cumprod(1.0 + r_sorted.astype(np.float64))
        run_max = np.maximum.accumulate(equity)
        dd = np.min(equity / (run_max + 1e-12) - 1.0) if equity.size>0 else 0.0
        mddp = float(abs(dd) * 100.0)
    # new metrics per threshold
    def _max_intratrade_dd_pct_for_mask(mask: np.ndarray) -> float:
        if not np.any(mask): return 0.0
        ent_ = entry_idx[mask]
        dd_min = 0.0; has_any=False
        for k in ent_:
            end = int(k + PRED_WINDOW)
            if end >= len(ds.lows):
                continue
            min_low = float(np.min(ds.lows[k:end+1]))
            entry_open = float(ds.opens[int(k)]) if int(k) < len(ds.opens) else float('nan')
            if not np.isfinite(entry_open) or entry_open <= 0: continue
            dd_i = (min_low / max(entry_open, 1e-12)) - 1.0
            if not has_any: dd_min = dd_i; has_any=True
            else: dd_min = min(dd_min, dd_i)
        return float(abs(dd_min) * 100.0) if has_any else 0.0
    def _pnl_seq_pct_for_mask(mask: np.ndarray) -> float:
        if not np.any(mask): return 0.0
        ent_ = entry_idx[mask]
        order = np.argsort(ent_)
        ent_sorted = ent_[order]
        r_sorted2 = ret_val[mask][order]
        equity = 1.0; last_exit = -10**9
        for e_i, r_i in zip(ent_sorted, r_sorted2):
            if e_i >= last_exit:
                equity *= (1.0 + float(r_i))
                last_exit = int(e_i + PRED_WINDOW)
        return float((equity - 1.0) * 100.0)
    max_intra_dd_list.append(_max_intratrade_dd_pct_for_mask(m))
    pnl_seq_list.append(_pnl_seq_pct_for_mask(m))
    thr_list.append(float(t)); pnl_list.append(sret*100.0); comp_list.append(comp*100.0 if np.isfinite(comp) else np.nan); sharpe_list.append(shp); trades_list.append(n); mean_ret_list.append(meanp); median_ret_list.append(medp); mdd_list.append(mddp)
print(f"Выбран порог по PnL (валидация): {best_thr:.4f}, comp_ret={best_comp*100 if np.isfinite(best_comp) else float('nan'):.2f}% trades={best_trades}")

try:
    fig, ax1 = plt.subplots(figsize=(18.4,13.0)); ax2 = ax1.twinx()
    # left metrics normalized to [0,1]
    thr_arr = np.asarray(thr_list)
    pnl_arr = np.asarray(pnl_list)
    comp_arr = np.asarray(comp_list)
    shp_arr = np.asarray(sharpe_list)
    mean_arr = np.asarray(mean_ret_list)
    med_arr  = np.asarray(median_ret_list)
    mdd_arr  = np.asarray(mdd_list)
    intradd_arr = np.asarray(max_intra_dd_list)
    pnlseq_arr = np.asarray(pnl_seq_list)
    # avg_dd (seq, price) per threshold using sequential non-overlapping trades
    def _avg_price_dd_seq_pct_for_mask(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        ent = entry_idx[mask]
        order = np.argsort(ent)
        ent_sorted = ent[order]
        dd_vals = []
        last_exit = -10**9
        for e_i in ent_sorted:
            if e_i >= last_exit:
                end = int(e_i + PRED_WINDOW)
                if end < len(ds.lows):
                    min_low = float(np.min(ds.lows[e_i:end+1]))
                    entry_open = float(ds.opens[int(e_i)]) if int(e_i) < len(ds.opens) else float('nan')
                    if np.isfinite(entry_open) and entry_open > 0:
                        dd_i = (min_low / max(entry_open, 1e-12)) - 1.0
                        dd_vals.append(abs(dd_i))
                last_exit = int(e_i + PRED_WINDOW)
        return float(np.mean(dd_vals) * 100.0) if len(dd_vals) > 0 else 0.0
    avgdd_seq_list = [ _avg_price_dd_seq_pct_for_mask(val_probs_all_np >= t) for t in thr_arr ]
    # pnl_ddd (seq) per threshold with dynamic exit rules
    def _pnl_ddd_pct_for_mask(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        ent = entry_idx[mask]
        order = np.argsort(ent)
        ent_sorted = ent[order]
        equity = 1.0
        last_exit = -10**9
        for e_i in ent_sorted:
            if e_i < last_exit:
                continue
            entry_open = float(ds.opens[int(e_i)]) if int(e_i) < len(ds.opens) else float('nan')
            if not np.isfinite(entry_open) or entry_open <= 0:
                continue
            exit_idx = None
            # scan up to max( PNL_DDD_MAX_HOLD_MIN, PRED_WINDOW )
            max_h = int(max(PNL_DDD_MAX_HOLD_MIN, PRED_WINDOW))
            for k in range(1, max_h+1):
                j = int(e_i + k)
                if j >= len(ds.opens):
                    break
                # stop-loss check (intra-minute via lows)
                low_j = float(ds.lows[j])
                if (low_j / entry_open - 1.0) <= PNL_DDD_STOP_LOSS_PCT:
                    exit_idx = j
                    break
                # dynamic exit condition
                close_j = float(ds.closes[j]); open_j = float(ds.opens[j])
                close_prev = float(ds.closes[j-1]) if j-1 >= 0 else close_j
                open_prev = float(ds.opens[j-1]) if j-1 >= 0 else open_j
                prev_green = (close_prev > open_prev)
                body_current = (close_j - open_j)
                body_prev = (close_prev - open_prev)
                body_smaller = (body_current < body_prev)
                price_up_enough = ((close_j / entry_open - 1.0) >= PNL_DDD_THRESH_PCT)
                if body_smaller and prev_green and price_up_enough:
                    exit_idx = j
                    break
            if exit_idx is None:
                exit_idx = int(e_i + max(PNL_DDD_MAX_HOLD_MIN, PRED_WINDOW))
                if exit_idx >= len(ds.closes):
                    exit_idx = len(ds.closes) - 1
            r_i = float(ds.closes[exit_idx] / entry_open - 1.0)
            equity *= (1.0 + r_i)
            last_exit = exit_idx
        return float((equity - 1.0) * 100.0)
    pnl_ddd_list = [ _pnl_ddd_pct_for_mask(val_probs_all_np >= t) for t in thr_arr ]
    pnl_ddd_arr = np.asarray(pnl_ddd_list)
    avgdd_arr = np.asarray(avgdd_seq_list)
    def _norm(a):
        a = np.asarray(a, dtype=np.float64)
        return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a) + 1e-12) if a.size>0 else a
    comp_n = _norm(comp_arr); pnl_n = _norm(pnl_arr); mean_n = _norm(mean_arr); med_n = _norm(med_arr); mdd_n = _norm(mdd_arr); intradd_n = _norm(intradd_arr); pnlseq_n = _norm(pnlseq_arr); avgdd_n = _norm(avgdd_arr); pnlddd_n = _norm(pnl_ddd_arr)

    # styles: mean black dashed, median gray dashed; others distinct
    l1, = ax1.plot(thr_arr, comp_n, label='comp_ret (norm)', color='#1f77b4', linewidth=1.8)
    l2, = ax1.plot(thr_arr, pnl_n,  label='pnl_sum (norm)',  color='#ff7f0e', linewidth=1.8)
    l3, = ax1.plot(thr_arr, mean_n, label='mean_ret (norm)', color='#000000', linestyle='--', linewidth=1.6)
    l4, = ax1.plot(thr_arr, med_n,  label='median_ret (norm)', color='#7f7f7f', linestyle='--', linewidth=1.6)
    l5, = ax1.plot(thr_arr, mdd_n,  label='max_drawdown (norm)', color='#2ca02c', linestyle='-', linewidth=1.6)
    l6, = ax2.plot(thr_arr, shp_arr, label='Sharpe', color='#9467bd', alpha=0.9)
    # add Trades on separate invisible y-axis
    ax3 = ax1.twinx(); ax3.get_yaxis().set_visible(False)
    l7, = ax3.plot(thr_arr, np.asarray(trades_list), label='Trades', color='#8c564b')
    # new metrics on left axis
    l8, = ax1.plot(thr_arr, intradd_n, label='Max IntraTrade DD (price, %)', color='#98df8a', linewidth=1.6)
    l9, = ax1.plot(thr_arr, pnlseq_n, label='PnL (seq, %)', color='#d62728', linewidth=1.6)
    l10, = ax1.plot(thr_arr, avgdd_n, label='avg_dd (%)', color='#17becf', linewidth=1.6)
    l11, = ax1.plot(thr_arr, pnlddd_n, label='PnL (ddd, %)', color='#bcbd22', linewidth=1.6)
    # placeholder for pnl_ddd; will compute below

    # constants box outside on the right; legend below
    const_text = (f"SEQ_LEN={SEQ_LEN}\nPRED_WINDOW={PRED_WINDOW}\nVAL_SPLIT={VAL_SPLIT}\n"
                  f"EPOCHS={EPOCHS}\nBATCH={BATCH_SIZE}\nBASE_LR={BASE_LR:.2e}\n"
                  f"pct_start={ONECYCLE_PCT_START}\ndiv_factor={ONECYCLE_DIV_FACTOR}\nfinal_div={ONECYCLE_FINAL_DIV_FACTOR}\n"
                  f"WD={WEIGHT_DECAY}\nDROPOUT={DEFAULT_DROPOUT:.3f}\nBEST_LR_MULT={BEST_LR_MULTIPLIER}"
                  f"\nauto_thr={AUTOTUNE_PRAUC_THRESHOLD}\nauto_gamma={AUTOTUNE_GAMMA}\nauto_WD×{AUTOTUNE_WD_MULT}\nauto_beta1={AUTOTUNE_BETA1}\nAPPLY_BETA={AUTOTUNE_APPLY_BETA}\nUSE_STANDARD_SCALER={USE_STANDARD_SCALER}\nGRADCLIP={GRADCLIP_MAXNORM_1_APPLY}\nGRADCLIP_MAXNORM={GRADCLIP_MAXNORM}\nbest_lr_default={best_lr_default:.2e}")
    try:
        fig.canvas.draw()
        ax_pos = ax1.get_position()
        panel_left = min(0.82, ax_pos.x1 + 0.01)
        panel_width = 1.0 - panel_left - 0.02
        if panel_width < 0.12:
            panel_width = 0.12
        # Detach from x-axis bottom: align top edge with y=1 of normalized curves (axes top)
        panel_height = max(0.18, 0.35 * ax_pos.height)
        panel_bottom = ax_pos.y1 - panel_height
        const_ax = fig.add_axes([panel_left, panel_bottom, panel_width, panel_height])
        const_ax.axis('off')
        const_ax.text(0.5, 1.0, const_text, ha='center', va='top', fontsize=8,
                      bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    except Exception:
        fig.text(0.985, 0.02, const_text, ha='right', va='bottom', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    handles, labels = [], []
    for ln in (l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11):
        handles.append(ln); labels.append(ln.get_label())
    leg2 = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=5)

    ax1.grid(True, alpha=0.3)
    # fixed-point annotations: real values; base top-center; lateral split on collision at same x
    try:
        thr_min_v = float(thr_min); thr_max_v = float(thr_max)
        delta = thr_max_v - thr_min_v
        t_points = [thr_min_v, thr_min_v + delta/3.0, thr_min_v + 2.0*delta/3.0, thr_max_v]
        def _avg_dd_for_mask(mask):
            ent = entry_idx[mask]
            order = np.argsort(ent)
            r_sorted = ret_val[mask][order] if np.any(mask) else np.array([], dtype=np.float64)
            if r_sorted.size == 0:
                return 0.0
            equity = np.cumprod(1.0 + r_sorted.astype(np.float64))
            run_max = np.maximum.accumulate(equity)
            dd = equity / (run_max + 1e-12) - 1.0
            dd = np.clip(dd, -1.0, 0.0)
            return float(abs(np.mean(dd)) * 100.0)
        series = [
            (comp_n, comp_arr, l1.get_color(), False),
            (pnl_n,  pnl_arr,  l2.get_color(), False),
            (mean_n, mean_arr, l3.get_color(), False),
            (med_n,  med_arr,  l4.get_color(), False),
            (mdd_n,  mdd_arr,  l5.get_color(), False),
            (intradd_n, intradd_arr, l8.get_color(), False),
            (pnlseq_n, pnlseq_arr, l9.get_color(), False),
            (avgdd_n, avgdd_arr, l10.get_color(), False),
            (pnlddd_n, pnl_ddd_arr, l11.get_color(), False),
        ]
        y_tol = 0.02
        # shifted thirds to reduce overlaps at x: for series indices per-third
        shift_map = [-1, +1, 0, -1, +1, -1, +1, 0, +1]
        offset = 0.10 * delta
        base_points = [thr_min_v, thr_min_v + delta/3.0, thr_min_v + 2.0*delta/3.0, thr_max_v]
        for base_idx, base_t in enumerate(base_points):
            items = []
            for si, (yn, yr, col, with_avg) in enumerate(series):
                t_mod = base_t
                if base_idx in (1, 2):
                    sh = shift_map[si] if si < len(shift_map) else 0
                    t_mod = base_t + sh * offset
                    if t_mod < thr_min_v: t_mod = thr_min_v
                    if t_mod > thr_max_v: t_mod = thr_max_v
                idx = int(np.argmin(np.abs(thr_arr - t_mod)))
                yv = float(yn[idx])
                rv = float(yr[idx])
                text = f"{rv:.2f}"
                if with_avg:
                    mask_here = (val_probs_all_np >= t_mod)
                    avg_dd = _avg_dd_for_mask(mask_here)
                    text = f"{rv:.2f}\navg_dd={avg_dd:.2f}%"
                items.append((yv, text, col, idx))
            buckets = {}
            for (yv, text, col, idx) in items:
                b = int(round(yv / max(y_tol, 1e-6)))
                buckets.setdefault(b, []).append((yv, text, col, idx))
            for b, group in buckets.items():
                if len(group) == 1:
                    yv, text, col, idx = group[0]
                    ax1.scatter([thr_arr[idx]],[yv], color=col, s=14)
                    ab = AnnotationBbox(TextArea(text, textprops=dict(color=col, fontsize=7)),
                                         (thr_arr[idx], yv), box_alignment=(0.5, 1.0),
                                         bboxprops=dict(boxstyle='round,pad=0.15', fc='white', ec=col, alpha=0.32))
                    ax1.add_artist(ab)
                else:
                    for k, (yv, text, col, idx) in enumerate(group):
                        ax1.scatter([thr_arr[idx]],[yv], color=col, s=14)
                        align = (1.0, 0.5) if (k % 2 == 0) else (0.0, 0.5)
                        ab = AnnotationBbox(TextArea(text, textprops=dict(color=col, fontsize=7)),
                                             (thr_arr[idx], yv), box_alignment=align,
                                             bboxprops=dict(boxstyle='round,pad=0.15', fc='white', ec=col, alpha=0.32))
                        ax1.add_artist(ab)

        # annotate Sharpe and Trades at 1/6 and 1/2 of threshold range
        try:
            t_points_shrt = [thr_min_v + delta/6.0, thr_min_v + 0.5*delta]
            for t in t_points_shrt:
                idx = int(np.argmin(np.abs(thr_arr - t)))
                # Sharpe (right axis)
                y_shp = float(shp_arr[idx])
                ax2.scatter([thr_arr[idx]],[y_shp], color=l6.get_color(), s=14)
                ab_shp = AnnotationBbox(TextArea(f"{y_shp:.2f}", textprops=dict(color=l6.get_color(), fontsize=7)),
                                        (thr_arr[idx], y_shp),
                                        box_alignment=(0.5, 1.0),
                                        bboxprops=dict(boxstyle='round,pad=0.15', fc='white', ec=l6.get_color(), alpha=0.7))
                ax2.add_artist(ab_shp)
                # Trades (hidden right axis)
                y_tr = float(np.asarray(trades_list)[idx])
                ax3.scatter([thr_arr[idx]],[y_tr], color=l7.get_color(), s=14)
                ab_tr = AnnotationBbox(TextArea(f"{int(y_tr)}", textprops=dict(color=l7.get_color(), fontsize=7)),
                                        (thr_arr[idx], y_tr),
                                        box_alignment=(0.5, 1.0),
                                        bboxprops=dict(boxstyle='round,pad=0.15', fc='white', ec=l7.get_color(), alpha=0.7))
                ax3.add_artist(ab_tr)
        except Exception:
            pass
    except Exception:
        pass
    # extend max CompRet annotation with new metrics
    if np.any(np.isfinite(comp_arr)):
        i_best = int(np.nanargmax(comp_arr))
        best_thr_local = float(thr_arr[i_best])
        ax1.axvline(best_thr_local, color=l1.get_color(), linestyle='--', linewidth=1.0, alpha=0.7)
        ax1.scatter([best_thr_local],[comp_n[i_best]], color=l1.get_color(), s=18)
        mask_best = (val_probs_all_np >= best_thr_local)
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
            _avg_dd_equity_unused = float(abs(np.mean(np.clip(dd_series, -1.0, 0.0))) * 100.0)
        else:
            max_dd_best = 0.0; _avg_dd_equity_unused = 0.0
        max_intra_best = _max_intratrade_dd_pct_for_mask(mask_best)
        pnl_seq_best = _pnl_seq_pct_for_mask(mask_best)
        avg_dd_seq_best = _avg_price_dd_seq_pct_for_mask(mask_best)
        pnl_ddd_best = _pnl_ddd_pct_for_mask(mask_best)
        text = (
            f"comp_ret: {float(comp_arr[i_best]):.2f}%\n"
            f"thr: {best_thr_local:.3f}\n"
            f"trades: {n_best}\n"
            f"pnl_sum: {sum_best:.2f}%\n"
            f"sharpe: {sharpe_best:.2f}\n"
            f"mean: {mean_best:.2f}%\n"
            f"median: {med_best:.2f}%\n"
            f"max_dd: {max_dd_best:.2f}%\n"
            f"avg_dd: {avg_dd_seq_best:.2f}%\n"
            f"max_intratrade_dd: {max_intra_best:.2f}%\n"
            f"pnl_seq: {pnl_seq_best:.2f}%\n"
            f"pnl_ddd: {pnl_ddd_best:.2f}%"
        )
        ax1.annotate(text, xy=(best_thr_local, comp_n[i_best]), xycoords='data',
                     xytext=(0.5, 1.04), textcoords='axes fraction',
                     ha='center', va='bottom', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    # keep tight_layout inside try
    plt.tight_layout(rect=[0.0, 0.22, 0.78, 1])
    out_name = f'threshold_sweep_{ts}.png'; fig.savefig(out_name, dpi=130)
    print(f"Saved threshold sweep plot to {Path(out_name).resolve()}"); plt.show()
except Exception as ex:
    print(f"! Не удалось построить график перебора порога: {ex}")