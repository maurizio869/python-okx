# eval.py
# Last modified (MSK): 2025-09-03 19:51 — правка номер 5
# Changes:
# - правка 1: Создан единый eval скрипт для обеих моделей (jump и drop)
# - правка 2: Переименован из price_jump_drop_eval_OneCFocalL.py в eval.py
# - правка 3: Исправлена загрузка модели - параметры архитектуры берутся из checkpoint
# - правка 4: Исправлены дата и время в шапке на правильные из системы Linux
# - правка 5: Добавлен флаг USE_CONSTANT_THRESHOLD для использования фиксированных порогов вместо подбираемых
"""Единый eval скрипт для jump и drop моделей OneCFocalL"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve

# ─── ПАРАМЕТРЫ ────────────────────────────────────────────────────
EVAL_JSON = Path("candles_eval.json")       # файл свечей для теста
JUMP_MODEL_PATH = Path("lstm_jump.pt")      # модель для jump
DROP_MODEL_PATH = Path("lstm_drop.pt")      # модель для drop
OUT_DATA = Path("viz_data.npz")             # куда сохранить данные
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Комиссии
MAKER_FEE = 0.0002
TAKER_FEE = 0.0005
USE_MAKER_FEES = False
ENTRY_FEE = MAKER_FEE if USE_MAKER_FEES else TAKER_FEE
EXIT_FEE = MAKER_FEE if USE_MAKER_FEES else TAKER_FEE

# Defaults (будут перезаписаны из моделей если есть)
SEQ_LEN = 30
PRED_WINDOW = 5
JUMP_THRESHOLD = 0.0035
DROP_THRESHOLD = 0.0035

# Флаг для использования фиксированных порогов
USE_CONSTANT_THRESHOLD = False
CONSTANT_JUMP_THRESHOLD = 0.5
CONSTANT_DROP_THRESHOLD = 0.5

def load_df(path: Path) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)
    df = pd.DataFrame(list(raw.values()))
    df["datetime"] = pd.to_datetime(df["x"], unit="s")
    return df.set_index("datetime").sort_index()

class EvalDataset(Dataset):
    """Dataset для OneCFocalL моделей (7 фичей, без нормализации)"""
    def __init__(self, df: pd.DataFrame, seq_len: int):
        self.seq_len = seq_len
        self.df = df
        
        # Извлекаем данные
        self.closes = df['c'].astype(np.float32).values
        self.opens = df['o'].astype(np.float32).values
        self.highs = df['h'].astype(np.float32).values
        self.lows = df['l'].astype(np.float32).values
        self.volumes = df['v'].astype(np.float32).values
        
        # Количество возможных окон
        self.n_samples = len(df) - seq_len - PRED_WINDOW
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Индексы для окна
        start_idx = idx
        end_idx = idx + self.seq_len
        
        # Извлекаем окно данных
        closes_w = self.closes[start_idx:end_idx]
        opens_w = self.opens[start_idx:end_idx]
        highs_w = self.highs[start_idx:end_idx]
        lows_w = self.lows[start_idx:end_idx]
        vols_w = self.volumes[start_idx:end_idx]
        
        # Вычисляем дополнительные фичи (КАК В ОБУЧЕНИИ!)
        body = np.abs(closes_w - opens_w) + 1e-12
        
        # Обрезка отрицательных значений для хвостов
        upper_w = np.clip(highs_w - np.maximum(opens_w, closes_w), 0.0, None)
        lower_w = np.clip(np.minimum(opens_w, closes_w) - lows_w, 0.0, None)
        
        # Отношения
        ratio_up = upper_w / body
        ratio_dn = lower_w / body
        
        # Собираем фичи в правильном порядке (как в обучении)
        x_seq = np.stack([
            closes_w,    # 0
            opens_w,     # 1
            highs_w,     # 2
            lows_w,      # 3
            vols_w,      # 4
            ratio_up,    # 5
            ratio_dn     # 6
        ], axis=0).astype(np.float32)
        
        # Транспонируем для модели
        x_seq = x_seq.T  # (seq_len, 7)
        
        return torch.from_numpy(x_seq)

class LSTMClassifier(nn.Module):
    """Модель LSTM для OneCFocalL"""
    def __init__(self, input_size=7, hidden_size=128, num_layers=3, dropout=0.35):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 2)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

def load_model_and_predict(model_path: Path, df: pd.DataFrame, model_name: str):
    """Загружает модель и делает предсказания"""
    
    if not model_path.exists():
        print(f"⚠️ Модель {model_path} не найдена, пропускаем")
        return None, None, None
    
    print(f"\n📊 Обработка {model_name}:")
    print(f"Загружаем модель {model_path}")
    
    # Загружаем чекпойнт
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    # Извлекаем метаданные
    if isinstance(ckpt, dict) and "meta" in ckpt:
        meta = ckpt["meta"]
    else:
        meta = {}
    
    # Параметры из мета или дефолтные
    seq_len = int(meta.get("seq_len", SEQ_LEN))
    pred_window = int(meta.get("pred_window", PRED_WINDOW))
    
    # Параметры архитектуры модели (должны совпадать с train скриптом)
    hidden_size = int(meta.get("hidden_size", 64))  # По умолчанию как в OneCFocalL
    num_layers = int(meta.get("num_layers", 2))     # По умолчанию как в OneCFocalL
    dropout = float(meta.get("dropout", 0.25))      # По умолчанию как в OneCFocalL
    
    # Создаем модель
    model = LSTMClassifier(input_size=7, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    
    # Загружаем веса
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    
    model.to(DEVICE).eval()
    
    # Создаем DataLoader
    dataset = EvalDataset(df, seq_len)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    # Массивы для результатов
    probs = np.zeros(len(dataset), dtype=np.float32)
    
    print(f"Делаем предсказания для {len(dataset)} окон...")
    with torch.no_grad():
        ptr = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            outputs = model(batch)
            probs_batch = torch.softmax(outputs, dim=1).cpu().numpy()
            probs[ptr:ptr+len(probs_batch)] = probs_batch[:, 1]
            ptr += len(probs_batch)
    
    # Вычисляем оптимальный порог по ROC кривой
    # Для этого нужны истинные метки - берем из целевой функции обучения
    opens_arr = df["o"].astype(np.float32).values
    closes_arr = df["c"].astype(np.float32).values
    
    true_labels = []
    for i in range(seq_len, len(df) - pred_window):
        current_open = opens_arr[i]
        
        if "jump" in model_name.lower():
            # Для jump: максимальная цена в окне предсказания
            max_close = np.max(closes_arr[i+1:i+pred_window+1])
            label = 1 if (max_close / current_open - 1.0) >= JUMP_THRESHOLD else 0
        else:
            # Для drop: минимальная цена в окне предсказания
            min_close = np.min(closes_arr[i+1:i+pred_window+1])
            label = 1 if (min_close / current_open - 1.0) <= -DROP_THRESHOLD else 0
        
        true_labels.append(label)
    
    true_labels = np.array(true_labels)
    
    # Находим оптимальный порог
    fpr, tpr, thresholds = roc_curve(true_labels, probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Оптимальный порог (подобран): {optimal_threshold:.4f}")
    
    # Проверяем флаг USE_CONSTANT_THRESHOLD
    if USE_CONSTANT_THRESHOLD:
        if "jump" in model_name.lower():
            used_threshold = CONSTANT_JUMP_THRESHOLD
            print(f"Используем константу jump_threshold={used_threshold:.4f} для jump предсказаний")
        else:
            used_threshold = CONSTANT_DROP_THRESHOLD
            print(f"Используем константу drop_threshold={used_threshold:.4f} для drop предсказаний")
    else:
        used_threshold = optimal_threshold
    
    # Бинаризация предсказаний
    preds = (probs >= used_threshold).astype(np.int8)
    
    print(f"Предсказаний 0: {np.sum(preds == 0)}")
    print(f"Предсказаний 1: {np.sum(preds == 1)}")
    print(f"Процент предсказаний 1: {np.mean(preds == 1)*100:.2f}%")
    
    return probs, preds, used_threshold

# ─── ОСНОВНОЙ КОД ─────────────────────────────────────────────────
print("=" * 60)
print("Единый eval для Jump и Drop моделей OneCFocalL")
print("=" * 60)

print(f"\n📁 Читаем {EVAL_JSON}")
df = load_df(EVAL_JSON)
print(f"Загружено {len(df)} свечей")

# Обрабатываем модель Jump
probs_jump, preds_jump, threshold_jump = load_model_and_predict(
    JUMP_MODEL_PATH, df, "Jump"
)

# Обрабатываем модель Drop
probs_drop, preds_drop, threshold_drop = load_model_and_predict(
    DROP_MODEL_PATH, df, "Drop"
)

# Подготавливаем данные для сохранения
save_data = {
    "index": df.index.astype("int64").values,
    "o": df["o"].values.astype(np.float32),
    "h": df["h"].values.astype(np.float32),
    "l": df["l"].values.astype(np.float32),
    "c": df["c"].values.astype(np.float32),
    "v": df["v"].values.astype(np.float32),
    "seq_len": np.int32(SEQ_LEN),
    "pred_window": np.int32(PRED_WINDOW),
    "maker_fee": np.float32(MAKER_FEE),
    "taker_fee": np.float32(TAKER_FEE),
    "use_maker_fees": np.bool_(USE_MAKER_FEES),
    "entry_fee": np.float32(ENTRY_FEE),
    "exit_fee": np.float32(EXIT_FEE),
}

# Добавляем данные jump если есть
if probs_jump is not None:
    save_data.update({
        "probs_jump": probs_jump,
        "preds_jump": preds_jump,
        "threshold_jump": np.float32(threshold_jump),
    })

# Добавляем данные drop если есть
if probs_drop is not None:
    save_data.update({
        "probs_drop": probs_drop,
        "preds_drop": preds_drop,
        "threshold_drop": np.float32(threshold_drop),
    })

# Для обратной совместимости - если есть только jump, дублируем как старые поля
if probs_jump is not None and probs_drop is None:
    save_data.update({
        "probs": probs_jump,
        "preds": preds_jump,
        "threshold": np.float32(threshold_jump),
    })

print(f"\n💾 Сохраняем {OUT_DATA}")
np.savez_compressed(OUT_DATA, **save_data)
print(f"✅ Данные сохранены: {OUT_DATA}")
print("=" * 60)