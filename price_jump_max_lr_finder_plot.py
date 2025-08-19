# price_jump_max_lr_finder_plot.py
# Last modified (MSK): 2025-08-19 15:05
"""LR Finder с графиком и авто-выбором max_lr.
- Строит loss vs lr (и EMA-сглаженную кривую)
- Отмечает lr_best (минимум EMA), lr_break (точка «обрыва») и выбранный max_lr
- Логирует параметры и выбранные точки
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split

# Данные и модель
SEQ_LEN, PRED_WINDOW, JUMP_THRESHOLD = 30, 5, 0.0035
TRAIN_JSON = Path("candles_10d.json")
MODEL_PATH = Path("lstm_jump.pt")
MODEL_META_PATH = MODEL_PATH.with_suffix(".meta.json")
VAL_SPLIT = 0.2
BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LR Finder параметры
BASE_LR_DEFAULT = 3e-4    # базовый ориентир (может быть переопределён из meta)
MIN_FACTOR = 1.0/20.0     # min_lr = base_lr * MIN_FACTOR
MAX_FACTOR = 8.0          # max_lr = base_lr * MAX_FACTOR
EMA_ALPHA = 0.3           # сглаживание loss
BREAK_RISE = 0.30         # доля роста от минимума для детекта «обрыва» (30%)
RISE_STEPS = 5            # число подряд растущих шагов для альтернативного детекта
# Ограничение выбора max_lr около base_lr (для устойчивости)
CLIP_MIN_FACTOR = 0.5
CLIP_MAX_FACTOR = 8.0

class CandleDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.closes = df["c"].astype(np.float32).values
        self.opens  = df["o"].astype(np.float32).values
        price_feats = df[["o","h","l","c"]].astype(np.float32).values
        volumes     = df["v"].astype(np.float32).values.reshape(-1, 1)
        raw_windows, labels = [], []
        for i in range(SEQ_LEN, len(df) - PRED_WINDOW):
            current_open = self.opens[i]
            max_close    = self.closes[i + 1 : i + PRED_WINDOW + 1].max()
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

class LSTMClassifier(nn.Module):
    def __init__(self, nfeat: int = 5, hidden: int = 64, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(nfeat, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 2)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

def load_dataframe(path: Path) -> pd.DataFrame:
    with open(path) as f: raw = json.load(f)
    df = pd.DataFrame(list(raw.values()))
    df["datetime"] = pd.to_datetime(df["x"], unit="s")
    return df.set_index("datetime").sort_index()

def ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    m = 0.0
    for i, v in enumerate(values):
        m = alpha * v + (1 - alpha) * (m if i > 0 else v)
        out[i] = m
    return out

def sci_fmt(y, pos):
    s = f"{y:.0e}"
    return s.replace('e-0','e-').replace('e+0','e+')

def main():
    print("Загружаем", TRAIN_JSON)
    df = load_dataframe(TRAIN_JSON)
    ds = CandleDataset(df)
    val = int(len(ds)*VAL_SPLIT)
    train_ds, _val_ds = random_split(ds, [len(ds)-val, val])
    tl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)

    # class weights на дисбаланс
    pos_cnt = sum(1 for _, y in ds.samples if y == 1)
    neg_cnt = len(ds) - pos_cnt
    pos_weight = neg_cnt / max(pos_cnt, 1)
    class_weights = torch.tensor([1.0, pos_weight], device=DEVICE)

    # dropout и base_lr из meta (по возможности)
    dropout_p = 0.3
    base_lr = BASE_LR_DEFAULT
    try:
        if MODEL_META_PATH.exists():
            with open(MODEL_META_PATH, 'r', encoding='utf-8') as mf:
                meta0 = json.load(mf)
            if isinstance(meta0, dict):
                if 'dropout' in meta0:
                    dropout_p = float(meta0['dropout'])
                if 'base_lr' in meta0:
                    base_lr = float(meta0['base_lr'])
    except Exception as ex:
        print(f"! Не удалось прочитать meta: {ex}")
    print(f"LR Finder setup: base_lr={base_lr:.2e}, min_factor={MIN_FACTOR}, max_factor={MAX_FACTOR}, ema_alpha={EMA_ALPHA}")

    model = LSTMClassifier(dropout=dropout_p).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), base_lr)
    lossf = nn.CrossEntropyLoss(weight=class_weights)

    min_lr = base_lr * MIN_FACTOR
    max_lr = base_lr * MAX_FACTOR
    num_steps = max(1, len(tl))
    lr_mult = (max_lr/min_lr) ** (1/num_steps)

    for pg in opt.param_groups: pg['lr'] = min_lr
    lrs = []
    losses = []

    model.train()
    for xb, yb in tl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(); logits = model(xb); loss = lossf(logits, yb)
        loss.backward(); opt.step()
        lrs.append(opt.param_groups[0]['lr'])
        losses.append(float(loss.item()))
        for pg in opt.param_groups: pg['lr'] *= lr_mult

    lrs = np.asarray(lrs, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    ema = ema_smooth(losses, EMA_ALPHA)

    # Поиск lr_best (минимум EMA)
    best_idx = int(np.nanargmin(ema))
    lr_best = float(lrs[best_idx]); loss_min = float(ema[best_idx])

    # Поиск lr_break (первая точка после минимума, где EMA выросла >= (1+BREAK_RISE)*min
    break_idx = None
    thresh = loss_min * (1.0 + BREAK_RISE)
    # критерий по росту EMA
    for i in range(best_idx+1, len(ema)):
        if ema[i] >= thresh:
            break_idx = i; break
    # запасной критерий: RISE_STEPS подряд положительных приростов EMA
    if break_idx is None and best_idx + RISE_STEPS < len(ema):
        diffs = np.diff(ema)
        for i in range(best_idx+1, len(ema)-RISE_STEPS):
            if np.all(diffs[i:i+RISE_STEPS] > 0):
                break_idx = i + 1
                break
    # если не найдено — возьмём 90-й перцентиль по шагам
    if break_idx is None:
        break_idx = int(min(len(lrs)-1, round(0.9 * len(lrs))))
    lr_break = float(lrs[break_idx])

    # Выбор max_lr «по первому варианту»: 0.7 * lr_break, с клиппингом вокруг base_lr
    max_lr_suggested = 0.7 * lr_break
    max_lr_suggested = float(np.clip(max_lr_suggested, base_lr*CLIP_MIN_FACTOR, base_lr*CLIP_MAX_FACTOR))

    print(f"lr_best≈{lr_best:.2e} (EMA min={loss_min:.4f}) at step {best_idx+1}/{len(lrs)}")
    print(f"lr_break≈{lr_break:.2e} at step {break_idx+1}/{len(lrs)} (rise≥{BREAK_RISE*100:.0f}% from min)")
    print(f"Chosen max_lr≈{max_lr_suggested:.2e} (0.7×lr_break, clipped to [{base_lr*CLIP_MIN_FACTOR:.2e}, {base_lr*CLIP_MAX_FACTOR:.2e}])")

    # Построение графика
    plt.figure(figsize=(7,4))
    plt.plot(lrs, losses, color='#999', alpha=0.6, label='loss (raw)')
    plt.plot(lrs, ema, color='#1f77b4', label=f'loss EMA (alpha={EMA_ALPHA})')
    plt.axvline(lr_best, color='#2ca02c', linestyle='--', label=f'lr_best {lr_best:.2e}')
    plt.axvline(lr_break, color='#d62728', linestyle='--', label=f'lr_break {lr_break:.2e}')
    plt.axvline(max_lr_suggested, color='#9467bd', linestyle='--', label=f'chosen max_lr {max_lr_suggested:.2e}')
    plt.xscale('log')
    plt.xlabel('Learning Rate (log)')
    plt.ylabel('Loss')
    plt.title('LR Finder')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(FuncFormatter(sci_fmt))
    plt.tight_layout()
    out_path = Path('lr_finder_plot.png')
    plt.savefig(out_path, dpi=130)
    print(f"Saved LR Finder plot to {out_path.resolve()}")
    try:
        from IPython.display import Image, display
        display(Image(str(out_path)))
    except Exception:
        pass
    plt.close()

if __name__ == '__main__':
    main()