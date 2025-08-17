#!/usr/bin/env python3
from pathlib import Path
import json
import os
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SEQ_LEN = int(os.getenv("SEQ_LEN", "20"))
PRED_WINDOW = int(os.getenv("PRED_WINDOW", "5"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
EVAL_JSON = Path(os.getenv("EVAL_JSON", "candles_2d.json"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "lstm_jump.pt"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "512"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_candles(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        raw = json.load(f)
    rows = list(raw.values())
    rows.sort(key=lambda r: r["x"])  # sort by timestamp
    return rows


def build_windows(rows: List[Dict[str, Any]], nfeat: int) -> List[List[List[float]]]:
    o_list = [float(r["o"]) for r in rows]
    h_list = [float(r["h"]) for r in rows]
    l_list = [float(r["l"]) for r in rows]
    c_list = [float(r["c"]) for r in rows]
    v_list = [float(r.get("v", 1.0)) for r in rows]

    windows: List[List[List[float]]] = []
    for i in range(SEQ_LEN, len(rows) - PRED_WINDOW):
        start = i - SEQ_LEN + 1
        ref_open = o_list[start]
        ref_vol = max(v_list[start], 1e-8)
        sample: List[List[float]] = []
        for t in range(start, i + 1):
            rel_o = o_list[t] / ref_open - 1.0
            rel_h = h_list[t] / ref_open - 1.0
            rel_l = l_list[t] / ref_open - 1.0
            rel_c = c_list[t] / ref_open - 1.0
            rel_v = v_list[t] / ref_vol - 1.0
            if nfeat == 4:
                sample.append([rel_o, rel_h, rel_l, rel_c])
            else:
                sample.append([rel_o, rel_h, rel_l, rel_c, rel_v])
        windows.append(sample)
    return windows


class LSTMClassifier(nn.Module):
    def __init__(self, nfeat: int = 5, hidden: int = 64, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(nfeat, hidden, layers, batch_first=True, dropout=0.0 if layers <= 1 else dropout)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


def main():
    print("Загружаем модель", MODEL_PATH)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    scaler = ckpt.get("scaler", {"mean": [0,0,0,0,0], "std": [1,1,1,1,1]})
    nfeat = int(ckpt.get("nfeat", 5))

    model = LSTMClassifier(nfeat=nfeat)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    print("Читаем", EVAL_JSON)
    rows = load_candles(EVAL_JSON)
    windows = build_windows(rows, nfeat)

    # apply scaler
    mean = scaler["mean"]
    std = scaler["std"]
    for w in windows:
        for row in w:
            for j in range(nfeat):
                row[j] = (row[j] - mean[j]) / (std[j] if std[j] != 0 else 1.0)

    class EvalDS(Dataset):
        def __init__(self, ws):
            self.ws = ws
        def __len__(self):
            return len(self.ws)
        def __getitem__(self, idx):
            return torch.tensor(self.ws[idx], dtype=torch.float32)

    loader = DataLoader(EvalDS(windows), batch_size=BATCH_SIZE)
    probs = []
    preds = []

    with torch.no_grad():
        for xb in loader:
            logits = model(xb.to(DEVICE))
            p = torch.softmax(logits, dim=1)[:, 1].cpu()
            probs.extend(p.tolist())
            preds.extend((p >= THRESHOLD).to(torch.int64).tolist())

    import statistics as stats
    print(f"Сделано {len(preds)} предсказаний")
    if probs:
        print(f"Средняя уверенность: {stats.mean(probs):.3f}")
        print(f"Мин/макс уверенность: {min(probs):.3f} / {max(probs):.3f}")
    print(f"Доля класса 1 по порогу {THRESHOLD}: {sum(preds) / max(len(preds),1):.3%}")


if __name__ == "__main__":
    main()