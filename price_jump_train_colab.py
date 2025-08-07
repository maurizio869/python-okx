# price_jump_train_colab.py
"""Обучает LSTM, метка = 1 если
   • максимум Close за следующие 5 мин ≥ Open + 0.35%
Сохраняет модель и StandardScaler в lstm_jump.pt
"""
from pathlib import Path
import json, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split

SEQ_LEN, PRED_WINDOW, JUMP_THRESHOLD = 20, 5, 0.0035  # 20-мин история, окно 5 мин

def load_dataframe(path: Path) -> pd.DataFrame:
    with open(path) as f: raw = json.load(f)
    df = pd.DataFrame(list(raw.values()))
    df["datetime"] = pd.to_datetime(df["x"], unit="s")
    return df.set_index("datetime").sort_index()

class CandleDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.closes = df["c"].astype(np.float32).values
        self.opens  = df["o"].astype(np.float32).values
        raw_feats   = df[["o", "h", "l", "c"]].astype(np.float32).values

        # Сохраняем необработанные относительные окна, чтобы потом подогнать StandardScaler
        raw_windows = []      # список (seq_len, 4)
        labels      = []

        for i in range(SEQ_LEN, len(df) - PRED_WINDOW):
            current_open = self.opens[i]
            max_close    = self.closes[i + 1 : i + PRED_WINDOW + 1].max()
            jump         = (max_close / current_open - 1) >= JUMP_THRESHOLD
            label        = 1 if jump else 0

            window_raw = raw_feats[i - SEQ_LEN + 1 : i + 1].copy()
            ref_open   = window_raw[0, 0]                # Open первой свечи
            window_rel = window_raw / ref_open - 1.0      # относительные изменения

            raw_windows.append(window_rel)
            labels.append(label)

        # Фитируем scaler на всех относительных значениях
        all_rows = np.vstack(raw_windows)                 # shape: (n_samples*seq_len, 4)
        self.scaler = StandardScaler().fit(all_rows)

        # Трансформируем и сохраняем финальные выборки
        self.samples = [(self.scaler.transform(w), lbl) for w, lbl in zip(raw_windows, labels)]

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x,y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

class LSTMClassifier(nn.Module):
    def __init__(self, nfeat: int = 4, hidden: int = 64, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        # dropout активен только если layers > 1
        self.lstm = nn.LSTM(nfeat, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 2)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# ─── параметры обучения ───────────────────────────────────────────
TRAIN_JSON = Path("candles_10d.json")
MODEL_PATH = Path("lstm_jump.pt")
VAL_SPLIT, EPOCHS = 0.2, 30
BATCH_SIZE, LR = 512, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

model = LSTMClassifier().to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), LR)
lossf = nn.CrossEntropyLoss(weight=class_weights)

for e in range(1, EPOCHS+1):
    model.train(); tot=0
    for x,y in tl:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); loss=lossf(model(x),y); loss.backward(); opt.step()
        tot += loss.item()*x.size(0)
    model.eval(); corr=tot_s=0
    with torch.no_grad():
        for x,y in vl:
            p=model(x.to(DEVICE)).argmax(1).cpu()
            corr+=(p==y).sum().item(); tot_s+=y.size(0)
    print(f'Epoch {e}/{EPOCHS} loss {tot/len(train_ds):.4f} val_acc {corr/tot_s:.3f}')

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save({"model_state":model.state_dict(),"scaler":ds.scaler}, MODEL_PATH)
print("✓ Модель сохранена в", MODEL_PATH.resolve())