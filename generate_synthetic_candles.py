#!/usr/bin/env python3
import argparse
import json
import time
import random
from typing import Dict, Any, List


def generate_candles(num_minutes: int, seed: int = 42, start_ts: int | None = None) -> Dict[str, Any]:
    """Generate simple synthetic OHLCV minute candles with occasional upward jumps.

    Output format matches existing scripts: a JSON object mapping string index to
    dicts with keys: x (timestamp, seconds), o,h,l,c (floats), v (float).
    """
    random.seed(seed)
    if start_ts is None:
        start_ts = int(time.time()) - num_minutes * 60

    price = 100.0
    mu = 0.0
    sigma = 0.0008  # ~0.08% per minute std
    jump_prob = 0.003  # chance per minute to trigger a mini upward jump
    jump_strength = (0.004, 0.012)  # 0.4% .. 1.2%

    candles: Dict[str, Any] = {}
    for i in range(num_minutes):
        ts = start_ts + i * 60
        open_price = price

        # base return
        ret = random.gauss(mu, sigma)

        # occasional upward jump
        if random.random() < jump_prob:
            ret += random.uniform(*jump_strength)

        close_price = max(0.01, open_price * (1.0 + ret))
        high = max(open_price, close_price) * (1.0 + random.uniform(0.0, 0.002))
        low  = min(open_price, close_price) * (1.0 - random.uniform(0.0, 0.002))
        vol  = float(max(1.0, random.lognormvariate(8.0, 0.5)))

        candles[str(i)] = {
            "x": ts,
            "o": float(open_price),
            "h": float(high),
            "l": float(low),
            "c": float(close_price),
            "v": vol,
        }

        price = close_price

    return candles


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-out", type=str, default="candles_10d.json")
    p.add_argument("--eval-out", type=str, default="candles_2d.json")
    p.add_argument("--train-minutes", type=int, default=6000)
    p.add_argument("--eval-minutes", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-ts", type=int, default=None)
    args = p.parse_args()

    train = generate_candles(args.train_minutes, seed=args.seed, start_ts=args.start_ts)
    with open(args.train_out, "w") as f:
        json.dump(train, f)
    print(f"Wrote {len(train)} candles -> {args.train_out}")

    eval_data = generate_candles(
        args.eval_minutes,
        seed=args.seed + 1,
        start_ts=(args.start_ts or int(time.time())) + args.train_minutes * 60,
    )
    with open(args.eval_out, "w") as f:
        json.dump(eval_data, f)
    print(f"Wrote {len(eval_data)} candles -> {args.eval_out}")


if __name__ == "__main__":
    main()