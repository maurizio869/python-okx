# price_jump_train_visible_counter.py
"""Версия с видимым индикатором для Pydroid3
"""
from pathlib import Path
import json, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split
import threading
import time

# Глобальная переменная для статуса
current_status = "Инициализация..."

def background_logger():
    global current_status
    counter = 0
    while True:
        counter += 1
        try:
            with open("/storage/emulated/0/status.txt", "w", encoding="utf-8") as f:
                f.write(f"{current_status} | Счётчик: {counter}")
        except:
            pass
        time.sleep(1)

# Запускаем фоновую задачу
print("Запускаем фоновый логгер...")
thread = threading.Thread(target=background_logger, daemon=True)
thread.start()

def update_status(status):
    global current_status
    current_status = status
    print(f"[{time.strftime('%H:%M:%S')}] {status}")  # Пытаемся вывести на экран

SEQ_LEN, PRED_WINDOW, JUMP_THRESHOLD = 20, 5, 0.0035

def load_dataframe(path: Path) -> pd.DataFrame:
    update_status(f"Загружаем файл {path}")
    with open(path) as f: raw = json.load(f)
    df = pd.DataFrame(list(raw.values()))
    df["datetime"] = pd.to_datetime(df["x"], unit="s")
    return df.set_index("datetime").sort_index()

class CandleDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        update_status("Создаём датасет...")
        self.closes = df["c"].astype(np.float32).values
        self.opens = df["o"].astype(np.float32).values
        feats = df[["o","h","l","c"]].astype(np.float32).values
        self.scaler = StandardScaler().fit(feats)
        feats = self.scaler.transform(feats)

        self.samples=[]
        for i in range(SEQ_LEN, len(df)-PRED_WINDOW):
            current_open = self.opens[i]
            max_close = self.closes[i+1:i+PRED_WINDOW+1].max()
            jump = (max_close / current_open - 1) >= JUMP_THRESHOLD
            label = 1 if jump else 0
            self.samples.append((feats[i-SEQ_LEN:i], label))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x,y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

class LSTMClassifier(nn.Module):
    def __init__(self,nfeat=4,hidden=64,layers=2):
        super().__init__(); self.lstm=nn.LSTM(nfeat,hidden,layers,batch_first=True); self.fc=nn.Linear(hidden,2)
    def forward(self,x): _,(h,_) = self.lstm(x); return self.fc(h[-1])

# Параметры
TRAIN_JSON = Path("candles_10d27.07.2025.json")
MODEL_PATH = Path("lstm_jump.pt")
VAL_SPLIT, EPOCHS = 0.2, 10
BATCH_SIZE, LR = 512, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

update_status("=== НАЧАЛО ОБУЧЕНИЯ ===")
update_status(f"Устройство: {DEVICE}")

df = load_dataframe(TRAIN_JSON)
update_status(f"Загружено {len(df)} свечей")

ds = CandleDataset(df)
update_status(f"Создано {len(ds)} сэмплов")
update_status(f"Меток 1: {sum(1 for _, y in ds.samples if y == 1)}")
update_status(f"Меток 0: {sum(1 for _, y in ds.samples if y == 0)}")

val = int(len(ds)*VAL_SPLIT)
train_ds,val_ds = random_split(ds,[len(ds)-val,val])
tl = DataLoader(train_ds,BATCH_SIZE,shuffle=True)
vl = DataLoader(val_ds,BATCH_SIZE)

update_status("Создаём модель...")
model = LSTMClassifier().to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), LR)
lossf = nn.CrossEntropyLoss()

update_status("Начинаем обучение...")
for e in range(1, EPOCHS+1):
    update_status(f"Эпоха {e}/{EPOCHS}")
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
    update_status(f'Epoch {e}/{EPOCHS} loss {tot/len(train_ds):.4f} val_acc {corr/tot_s:.3f}')

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save({"model_state":model.state_dict(),"scaler":ds.scaler}, MODEL_PATH)
update_status(f"✓ Модель сохранена в {MODEL_PATH.resolve()}")
update_status("=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")