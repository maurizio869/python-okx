#!/usr/bin/env python3
from pathlib import Path
import json
import os
from typing import List, Tuple, Dict, Any
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SEQ_LEN = int(os.getenv("SEQ_LEN", "20"))
PRED_WINDOW = int(os.getenv("PRED_WINDOW", "5"))
JUMP_THRESHOLD = float(os.getenv("JUMP_THRESHOLD", "0.0035"))

TRAIN_JSON = Path(os.getenv("TRAIN_JSON", "candles_10d.json"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "lstm_jump.pt"))
EPOCHS = int(os.getenv("EPOCHS", "5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "512"))
LR = float(os.getenv("LR", "1e-3"))
NFEAT = int(os.getenv("NFEAT", "5"))  # o,h,l,c,v
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_candles(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        raw = json.load(f)
    rows = list(raw.values())
    rows.sort(key=lambda r: r["x"])  # sort by timestamp
    return rows


def build_windows(rows: List[Dict[str, Any]]) -> Tuple[List[List[List[float]]], List[int]]:
    o_list = [float(r["o"]) for r in rows]
    h_list = [float(r["h"]) for r in rows]
    l_list = [float(r["l"]) for r in rows]
    c_list = [float(r["c"]) for r in rows]
    v_list = [float(r.get("v", 1.0)) for r in rows]

    windows: List[List[List[float]]] = []  # [sample][t][feat]
    labels: List[int] = []

    for i in range(SEQ_LEN, len(rows) - PRED_WINDOW):
        current_open = o_list[i]
        max_close = max(c_list[i + 1 : i + PRED_WINDOW + 1])
        jump = (max_close / current_open - 1.0) >= JUMP_THRESHOLD
        label = 1 if jump else 0

        # relative to first open and first volume of the window
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
            if NFEAT == 4:
                sample.append([rel_o, rel_h, rel_l, rel_c])
            else:
                sample.append([rel_o, rel_h, rel_l, rel_c, rel_v])
        windows.append(sample)
        labels.append(label)

    return windows, labels


def fit_scaler(windows: List[List[List[float]]]) -> Dict[str, List[float]]:
    # compute per-feature mean/std across all time steps and windows
    if not windows:
        return {"mean": [0.0] * NFEAT, "std": [1.0] * NFEAT}
    sums = [0.0] * NFEAT
    sumsqs = [0.0] * NFEAT
    count = 0
    for w in windows:
        for row in w:
            for j in range(NFEAT):
                x = float(row[j])
                sums[j] += x
                sumsqs[j] += x * x
            count += 1
    mean = [s / max(count, 1) for s in sums]
    std = []
    for j in range(NFEAT):
        m = mean[j]
        var = max(sumsqs[j] / max(count, 1) - m * m, 0.0)
        s = math.sqrt(var)
        if s < 1e-8:
            s = 1.0
        std.append(s)
    return {"mean": mean, "std": std}


def apply_scaler(windows: List[List[List[float]]], scaler: Dict[str, List[float]]):
    mean = scaler["mean"]
    std = scaler["std"]
    for w in windows:
        for row in w:
            for j in range(NFEAT):
                row[j] = (row[j] - mean[j]) / std[j]


class CandleDS(Dataset):
    def __init__(self, windows: List[List[List[float]]], labels: List[int]):
        self.windows = windows
        self.labels = labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class LSTMClassifier(nn.Module):
    def __init__(self, nfeat: int = NFEAT, hidden: int = 64, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=nfeat,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


def main():
    print("Загружаем", TRAIN_JSON)
    rows = load_candles(TRAIN_JSON)
    print(f"Свечей: {len(rows)}")

    windows, labels = build_windows(rows)
    print(f"Сэмплов: {len(windows)}")
    pos_cnt = sum(1 for y in labels if y == 1)
    neg_cnt = len(labels) - pos_cnt
    print(f"Класс 1: {pos_cnt}, класс 0: {neg_cnt}")

    scaler = fit_scaler(windows)
    apply_scaler(windows, scaler)

    ds = CandleDS(windows, labels)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMClassifier().to(DEVICE)
    pos_weight = (neg_cnt / max(pos_cnt, 1)) if pos_cnt > 0 else 1.0
    class_weights = torch.tensor([1.0, float(pos_weight)], device=DEVICE)
    lossf = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.Adam(model.parameters(), LR)

    for e in range(1, EPOCHS + 1):
        model.train()
        tot_loss = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = lossf(logits, yb)
            loss.backward()
            opt.step()
            bsz = xb.size(0)
            seen += bsz
            tot_loss += loss.item() * bsz
        print(f"Epoch {e}/{EPOCHS} loss {tot_loss / max(seen,1):.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "scaler": scaler,
        "seq_len": SEQ_LEN,
        "nfeat": NFEAT,
    }
    torch.save(ckpt, MODEL_PATH)
    print("✓ Модель сохранена в", MODEL_PATH.resolve())


if __name__ == "__main__":
    main()