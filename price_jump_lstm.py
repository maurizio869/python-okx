// price_jump_lstm.py
"""
Train an LSTM model to predict 0.35% price jumps within 5 minutes
based on the previous 20-minute minute-candle history.

Usage (training):
    python price_jump_lstm.py --train <10_day_history.json> --val_split 0.2

Usage (evaluation & visualisation):
    python price_jump_lstm.py --eval <2_day_history.json> --model_path <saved.pt>

The script will output candlestick charts with predicted jump moments marked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

SEQ_LEN = 20  # minutes history
PRED_WINDOW = 5  # minutes to look ahead
JUMP_THRESHOLD = 0.0035  # 0.35% in decimal


class CandleDataset(Dataset):
    """Torch dataset providing (seq_len, features) -> label pairs."""

    def __init__(self, df: pd.DataFrame):
        # Ensure df is sorted ascending by time
        df = df.sort_index()
        self.closes = df["c"].values.astype(np.float32)
        self.highs = df["h"].values.astype(np.float32)
        self.features = np.stack([
            df["o"].values,
            df["h"].values,
            df["l"].values,
            df["c"].values,
            df["v"].values,
        ], axis=1).astype(np.float32)

        # Normalise features per column (fit on entire dataset for simplicity)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        self.samples: List[Tuple[np.ndarray, int]] = []
        self._build_samples()

    def _build_samples(self):
        n = len(self.closes)
        for i in range(SEQ_LEN, n - PRED_WINDOW):
            start_close = self.closes[i]
            future_max_high = self.highs_slice(i + 1, i + PRED_WINDOW)
            label = 1 if future_max_high / start_close - 1 >= JUMP_THRESHOLD else 0
            seq = self.features[i - SEQ_LEN : i]
            self.samples.append((seq, label))

    def highs_slice(self, start: int, end: int) -> float:
        return self.highs[start : end + 1].max()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.lstm(x)
        # Use last layer's hidden state
        h_last = h_n[-1]
        return self.fc(h_last)


def prepare_dataframe(json_path: Path) -> pd.DataFrame:
    """Read json file and return DataFrame indexed by timestamp ascending."""
    with open(json_path) as f:
        raw = json.load(f)
    # Convert dict to list of records
    records = list(raw.values())
    df = pd.DataFrame(records)
    # Ensure column names exist
    expected_cols = {"x", "o", "h", "l", "c", "v"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")
    df["datetime"] = pd.to_datetime(df["x"], unit="s")
    df = df.set_index("datetime").sort_index()
    return df


def train_model(df: pd.DataFrame, val_split: float, epochs: int, batch_size: int, lr: float, device):
    dataset = CandleDataset(df)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = LSTMClassifier(n_features=dataset.features.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
        avg_loss = total_loss / train_size

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        val_acc = correct / total if total else 0
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - val_acc: {val_acc:.4f}")

    return model, dataset.scaler


def predict_jumps(df: pd.DataFrame, model: nn.Module, scaler: StandardScaler, device):
    model.eval()
    dataset = CandleDataset(df)
    # Overwrite scaler to provided (fitted on training data)
    dataset.scaler = scaler
    dataset.features = scaler.transform(dataset.features)
    loader = DataLoader(dataset, batch_size=512)
    preds = np.zeros(len(dataset), dtype=int)
    idx_offset = SEQ_LEN  # mapping dataset idx to df row idx
    with torch.no_grad():
        ptr = 0
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            preds[ptr : ptr + len(batch_preds)] = batch_preds
            ptr += len(batch_preds)
    # Map predictions back to original dataframe timestamps (align with the end of sequence)
    pred_series = pd.Series(preds, index=df.index[idx_offset : idx_offset + len(preds)])
    return pred_series


def visualize(df: pd.DataFrame, predictions: pd.Series, out_path: Path | None = None):
    """Plot candlestick chart with predicted jumps marked."""
    df_plot = df.copy()
    # Convert to mplfinance expected columns
    df_plot = df_plot.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})

    # Build marker dataframe
    jumps = predictions[predictions == 1]

    addplots = []
    if not jumps.empty:
        ap = mpf.make_addplot(
            df_plot.loc[jumps.index, "High"] * 1.002,  # slightly above high
            type="scatter",
            markersize=50,
            marker="^",
            color="red",
        )
        addplots.append(ap)

    mpf.plot(
        df_plot,
        type="candle",
        style="charles",
        addplot=addplots,
        volume=True,
        show_nontrading=True,
        datetime_format="%m-%d %H:%M",
        xrotation=15,
        savefig=str(out_path) if out_path else None,
    )


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate LSTM jump predictor")
    parser.add_argument("--train", type=Path, help="Path to 10-day JSON candle file", default=None)
    parser.add_argument("--eval", type=Path, help="Path to 2-day JSON candle file", default=None)
    parser.add_argument("--model_path", type=Path, help="Path to save/load model", default=Path("lstm_jump.pt"))
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if args.train:
        df_train = prepare_dataframe(args.train)
        model, scaler = train_model(
            df_train,
            val_split=args.val_split,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
        torch.save({"model_state": model.state_dict(), "scaler": scaler}, args.model_path)
        print(f"Model saved to {args.model_path}")

    if args.eval:
        if not args.model_path.exists():
            raise FileNotFoundError("Provide a trained model via --model_path")
        checkpoint = torch.load(args.model_path, map_location=args.device)
        model = LSTMClassifier(n_features=5)
        model.load_state_dict(checkpoint["model_state"])
        model.to(args.device)
        scaler = checkpoint["scaler"]

        df_eval = prepare_dataframe(args.eval)
        predictions = predict_jumps(df_eval, model, scaler, device=args.device)
        visualize(df_eval, predictions)


if __name__ == "__main__":
    main()