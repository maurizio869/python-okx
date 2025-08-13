# price_jump_train_colab_FINDERandOneCycleLR.py
# Last modified (MSK): 2025-08-13 21:41
"""Тренировка LSTM: LR Finder + OneCycleLR вместо ReduceLROnPlateau.
- 1-я стадия: короткий LR finder на подмножестве данных/эпохах
- 2-я стадия: основное обучение с OneCycleLR
Остальное: как в базовом тренинге (v-канал, SEQ_LEN=30, ранний стоп по PR AUC, PnL@0.565, подбор порога по PnL).
"""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader, random_split

SEQ_LEN, PRED_WINDOW, JUMP_THRESHOLD = 30, 5, 0.0035
TRAIN_JSON = Path("candles_10d.json")
MODEL_PATH = Path("lstm_jump.pt")
PNL_MODEL_PATH = Path("lstm_jump_pnl.pt")
MODEL_META_PATH = MODEL_PATH.with_suffix(".meta.json")
VAL_SPLIT, EPOCHS = 0.2, 70
BATCH_SIZE, BASE_LR = 512, 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
train_ds, val_ds = random_split(ds,[len(ds)-val,val])
train_loader = DataLoader(train_ds,BATCH_SIZE,shuffle=True)
val_loader   = DataLoader(val_ds,BATCH_SIZE)

model = LSTMClassifier().to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), BASE_LR)
lossf = nn.CrossEntropyLoss()

# LR Finder: 1 эпоха по train_loader, lr от BASE_LR/20 до BASE_LR*8
print("LR Finder: старт…")
min_lr, max_lr = BASE_LR/20, BASE_LR*8
num_steps = max(1, len(train_loader))
lr_mult = (max_lr/min_lr) ** (1/num_steps)
for pg in opt.param_groups: pg['lr'] = min_lr
best_loss = float('inf'); best_lr = BASE_LR
model.train(); step_id=0
for xb, yb in train_loader:
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    opt.zero_grad(); logits = model(xb); loss = lossf(logits, yb)
    loss.backward(); opt.step()
    if loss.item() < best_loss:
        best_loss = loss.item(); best_lr = opt.param_groups[0]['lr']
    for pg in opt.param_groups: pg['lr'] *= lr_mult
    step_id += 1
print(f"LR Finder: best_lr≈{best_lr:.2e}, best_loss={best_loss:.4f}")

# OneCycleLR на весь ран: max_lr = 1.5×best_lr (в разумных пределах)
max_lr = float(np.clip(1.2*best_lr, BASE_LR*0.5, BASE_LR*8))
opt = torch.optim.Adam(model.parameters(), BASE_LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=max_lr, epochs=EPOCHS, steps_per_epoch=len(train_loader),
    pct_start=0.45, div_factor=50.0, final_div_factor=1e3
)

# PnL@best threshold support
thr_min,thr_max,thr_step=0.30,0.70,0.0025
last_best_thr = 0.565

# PnL@0.565 support
val_indices = np.asarray(val_ds.indices, dtype=np.int64)
entry_idx = val_indices + SEQ_LEN
entry_opens = ds.opens[entry_idx]
exit_closes = ds.closes[entry_idx + PRED_WINDOW]
ret_val_fixed = exit_closes / np.maximum(entry_opens, 1e-12) - 1.0

best_pr_auc = -1.0
best_pnl_sum = -float('inf')
epochs_no_improve = 0
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
    if e % 5 == 0 or e == 1:
        best_comp=-np.inf; best_thr=last_best_thr; best_trades=0; best_sum=0.0
        for t in np.arange(thr_min,thr_max+1e-12,thr_step):
            m=(val_probs_np>=t); n=int(m.sum())
            if n==0: comp=-np.inf; sret=0.0
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
    print(f'Epoch {e}/{EPOCHS} lr {curr_lr:.2e} loss {total_loss/len(train_ds):.4f} '
          f'val_acc {corr/tot_s:.3f} F1 {f1:.3f} ROC_AUC {roc_auc:.3f} PR_AUC {pr_auc:.3f} '
          f'PNL@best(thr={last_best_thr:.4f}) {pnl_best_sum*100:.2f}% trades={trades_best}')

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
        PNL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state":model.state_dict(),"scaler":ds.scaler,
                    "meta":{"seq_len":SEQ_LEN,"pred_window":PRED_WINDOW}}, PNL_MODEL_PATH)
        print(f"Лучшая модель с pnl@{last_best_thr:.4f}={best_pnl_sum*100:.2f}% сохранена в {PNL_MODEL_PATH.resolve()}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= 40:
            print(f"⏹ Ранний стоп: PR AUC не улучшается {epochs_no_improve} эпох подряд"); break

print(f"Лучшая модель с PR_AUC={best_pr_auc:.3f} сохранена в {MODEL_PATH.resolve()}")

# PnL threshold sweep (0.43..0.70 step 0.0025)
print("Подбираем порог по PnL на валидационном наборе…")
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"]); model.to(DEVICE).eval()
val_probs_all = np.zeros(len(val_ds), dtype=np.float32)
with torch.no_grad():
    ptr=0
    for xb,_yb in DataLoader(val_ds,batch_size=BATCH_SIZE):
        logits=model(xb.to(DEVICE)); prob1=torch.softmax(logits,dim=1)[:,1].cpu().numpy()
        val_probs_all[ptr:ptr+len(prob1)] = prob1; ptr += len(prob1)
entry_opens = ds.opens[entry_idx]; exit_closes = ds.closes[entry_idx+PRED_WINDOW]
ret_val = exit_closes/np.maximum(entry_opens,1e-12)-1.0
thr_min,thr_max,thr_step=0.30,0.70,0.0025
print(f"Перебор порога по PnL (валидация): min={thr_min:.3f}, max={thr_max:.3f}, step={thr_step:.4f}")
thresholds=np.arange(thr_min,thr_max+1e-12,thr_step)
best_comp=-np.inf; best_thr=float(thresholds[0]); best_trades=0
for t in thresholds:
    m=(val_probs_all>=t); n=int(m.sum())
    if n==0: comp=-np.inf; shp=0.0; sret=0.0
    else:
        r=ret_val[m]; comp=-1.0 if np.any(r<=-0.999999) else float(np.exp(np.sum(np.log1p(r)))-1.0)
        shp=float(np.mean(r)/(np.std(r)+1e-12)) if r.size>=2 else 0.0
        sret=float(np.sum(r))
    print(f"thr={t:.4f} trades={n} pnl={sret*100:.2f}% comp_ret={comp*100 if np.isfinite(comp) else float('nan'):.2f}% sharpe={shp:.2f}")
    if comp>best_comp: best_comp=comp; best_thr=float(t); best_trades=n
print(f"Выбран порог по PnL (валидация): {best_thr:.4f}, comp_ret={best_comp*100 if np.isfinite(best_comp) else float('nan'):.2f}% trades={best_trades}")
final={"model_state":model.state_dict(),"scaler":ds.scaler,
        "meta":{"seq_len":SEQ_LEN,"pred_window":PRED_WINDOW,"threshold":best_thr}}
torch.save(final, MODEL_PATH)
try:
    with open(MODEL_META_PATH,'w',encoding='utf-8') as mf:
        json.dump({"seq_len":int(SEQ_LEN),"pred_window":int(PRED_WINDOW),"threshold":float(best_thr)}, mf)
except Exception as ex:
    print(f"! Не удалось записать meta-файл {MODEL_META_PATH}: {ex}")