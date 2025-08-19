# lstm_jump_dropout_p_find.py
# Last modified (MSK): 2025-08-19 15:05
"""Быстрый свип по dropout p, с кратким LR Finder и коротким обучением.
Записывает выбранные 'dropout' и 'base_lr' в hyper (MODEL_PATH.with_suffix('.hyper.json')).
"""
from pathlib import Path
import json, numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset, DataLoader, random_split

# Константы/пути
SEQ_LEN, PRED_WINDOW, JUMP_THRESHOLD = 30, 5, 0.0035
TRAIN_JSON = Path("candles_10d.json")
MODEL_PATH = Path("lstm_jump.pt")
MODEL_META_PATH = MODEL_PATH.with_suffix(".meta.json")
HYPER_PATH = MODEL_PATH.with_suffix(".hyper.json")
VAL_SPLIT = 0.2
BATCH_SIZE = 512
BASE_LR_DEFAULT = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

P_GRID = [float(f"{p:.3f}") for p in np.arange(0.18, 0.40 + 1e-12, 0.01)]  # тонкий шаг 0.01 в диапазоне [0.18, 0.40]
SHORT_EPOCHS = 12
LR_FINDER_MIN_FACTOR = 1.0/20.0
LR_FINDER_MAX_FACTOR = 8.0
BEST_LR_MULTIPLIER = 1.2
CLIP_MIN_FACTOR = 0.5
CLIP_MAX_FACTOR = 8.0
ONECYCLE_PCT_START = 0.45
ONECYCLE_DIV_FACTOR = 50.0
ONECYCLE_FINAL_DIV_FACTOR = 1e3
WEIGHT_DECAY = 1e-4

# Датасет и модель
class CandleDataset(Dataset):
	def __init__(self, df: pd.DataFrame):
		self.closes = df["c"].astype(np.float32).values
		self.opens  = df["o"].astype(np.float32).values
		price_feats = df[["o","h","l","c"]].astype(np.float32).values
		volumes     = df["v"].astype(np.float32).values.reshape(-1,1)
		raw_windows, labels = [], []
		for i in range(SEQ_LEN, len(df)-PRED_WINDOW):
			cur_open = self.opens[i]
			max_close = self.closes[i+1:i+PRED_WINDOW+1].max()
			labels.append(1 if (max_close/cur_open - 1) >= JUMP_THRESHOLD else 0)
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

def evaluate_pr_auc_and_pnl(model, vl, device, ds, val_ds) -> tuple[float,float]:
	model.eval()
	targets=[]; probs=[]
	with torch.no_grad():
		for xb,yb in vl:
			logits = model(xb.to(device))
			probs.append(torch.softmax(logits,dim=1)[:,1].cpu().numpy())
			targets.append(yb.numpy())
	y = np.concatenate(targets)
	p = np.concatenate(probs)
	try:
		pr_auc = float(average_precision_score(y, p))
	except Exception:
		pr_auc = float('nan')
	# простой PnL при фиксированном thr=0.565
	val_indices = np.asarray(val_ds.indices, dtype=np.int64)
	entry_idx = val_indices + SEQ_LEN
	entry_opens = ds.opens[entry_idx]
	exit_closes = ds.closes[entry_idx + PRED_WINDOW]
	ret = exit_closes / np.maximum(entry_opens, 1e-12) - 1.0
	m = (p >= 0.565)
	pnl_sum = float(np.sum(ret[m])) if m.any() else 0.0
	return pr_auc, pnl_sum

def main():
	print("Загружаем", TRAIN_JSON)
	df = load_dataframe(TRAIN_JSON)
	ds = CandleDataset(df)
	val = int(len(ds)*VAL_SPLIT)
	train_ds,val_ds = random_split(ds,[len(ds)-val,val])
	train_loader = DataLoader(train_ds,BATCH_SIZE,shuffle=True)
	val_loader   = DataLoader(val_ds,BATCH_SIZE)

	best_p = None; best_score = -np.inf; best_base_lr = BASE_LR_DEFAULT
	for p in P_GRID:
		print(f"\n=== Try dropout p={p} ===")
		model = LSTMClassifier(dropout=p).to(DEVICE)
		opt = torch.optim.Adam(model.parameters(), BASE_LR_DEFAULT)
		lossf = nn.CrossEntropyLoss()
		# LR Finder (1 эпоха, эксп. рост lr)
		min_lr = BASE_LR_DEFAULT*LR_FINDER_MIN_FACTOR
		max_lr = BASE_LR_DEFAULT*LR_FINDER_MAX_FACTOR
		num_steps = max(1, len(train_loader))
		lr_mult = (max_lr/min_lr) ** (1/num_steps)
		for pg in opt.param_groups: pg['lr'] = min_lr
		best_loss = float('inf'); best_lr = BASE_LR_DEFAULT
		model.train()
		for xb,yb in train_loader:
			xb,yb = xb.to(DEVICE), yb.to(DEVICE)
			opt.zero_grad(); logits=model(xb); loss=lossf(logits,yb)
			loss.backward(); opt.step()
			if loss.item() < best_loss:
				best_loss = loss.item(); best_lr = opt.param_groups[0]['lr']
			for pg in opt.param_groups: pg['lr'] *= lr_mult
		print(f"LR Finder: best_lr≈{best_lr:.2e}, best_loss={best_loss:.4f}")
		# OneCycle короткий ран
		max_lr = float(np.clip(BEST_LR_MULTIPLIER*best_lr, BASE_LR_DEFAULT*CLIP_MIN_FACTOR, BASE_LR_DEFAULT*CLIP_MAX_FACTOR))
		opt = torch.optim.Adam(model.parameters(), BASE_LR_DEFAULT, weight_decay=WEIGHT_DECAY)
		sched = torch.optim.lr_scheduler.OneCycleLR(
			opt, max_lr=max_lr, epochs=SHORT_EPOCHS, steps_per_epoch=len(train_loader),
			pct_start=ONECYCLE_PCT_START, div_factor=ONECYCLE_DIV_FACTOR, final_div_factor=ONECYCLE_FINAL_DIV_FACTOR
		)
		for e in range(SHORT_EPOCHS):
			model.train();
			for xb,yb in train_loader:
				xb,yb = xb.to(DEVICE), yb.to(DEVICE)
				opt.zero_grad(); logits=model(xb); loss=lossf(logits,yb)
				loss.backward(); opt.step(); sched.step()
		# Оценка
		pr_auc, pnl_sum = evaluate_pr_auc_and_pnl(model, val_loader, DEVICE, ds, val_ds)
		print(f"p={p} => PR_AUC={pr_auc:.3f}, PnL@0.565={pnl_sum*100:.2f}%")
		if pr_auc > best_score:
			best_score = pr_auc; best_p = p; best_base_lr = BASE_LR_DEFAULT

	print(f"\nВыбрано: dropout p={best_p}, base_lr={best_base_lr:.2e}")
	# Обновляем/дописываем hyper (отдельный файл)
	hyper = {}
	if HYPER_PATH.exists():
		try:
			with open(HYPER_PATH,'r',encoding='utf-8') as hf:
				hyper = json.load(hf)
		except Exception:
			hyper = {}
	if not isinstance(hyper, dict):
		hyper = {}
	hyper['dropout'] = float(best_p)
	hyper['base_lr'] = float(best_base_lr)
	with open(HYPER_PATH,'w',encoding='utf-8') as hf:
		json.dump(hyper, hf)
	print(f"Сохранено в {HYPER_PATH}: dropout={best_p}, base_lr={best_base_lr:.2e}")

if __name__ == '__main__':
	main()

# EOF