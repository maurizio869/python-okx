# lstm_predictor_class.py
# Сводка обновлений проекта

"""
Обновить скрипты и заголовки

price_jump_train_OneCFocalL.py was updated:

avgdd_arr = np.asarray(avgdd_seq_list) added (line 666) to fix avg_dd metric plotting.
AUTOTUNE_WD_MULT changed from 1.4 to 1.0 (line 90).
ret_val_fixed_no_fee introduced (line 307) to remove fees from PnL in training epochs/curves, retaining them in threshold sweep.
DEFAULT_DROPOUT changed from 0.37 to 0.35 (line 69), and AUTOTUNE_GAMMA from 1.6 to 1.4 (line 92). All changes were pushed to origin/cursor/abc. A new file, lstm_predictor.py, was created, providing a class for model inference with feature preparation (specific order, clipping, no normalization) matching training.
claude-4.1-opus • cursor/bc-2cd0f2ad-abb3-410b-8a21-671a05b3303e-d98a
Diff
Chat
Diff
Chat
131 earlier messages
Summary
price_jump_train_OneCFocalL.py was updated:

avgdd_arr = np.asarray(avgdd_seq_list) added (line 666) to fix avg_dd metric plotting.
AUTOTUNE_WD_MULT changed from 1.4 to 1.0 (line 90).
ret_val_fixed_no_fee introduced (line 307) to remove fees from PnL in training epochs/curves, retaining them in threshold sweep.
DEFAULT_DROPOUT changed from 0.37 to 0.35 (line 69), and AUTOTUNE_GAMMA from 1.6 to 1.4 (line 92).
All changes were pushed to origin/cursor/abc.
A new file, lstm_predictor.py, was created, providing a class for model inference with feature preparation (specific order, clipping, no normalization) matching training.
Нтрмализация есть в обучении (onecfocall)?, standart scaler используется в обучении(вроде он там есть но не используется)? ​обрезка выбросов сейчас есть а обучении?
Отличный вопрос! Давайте проверим, что именно используется в скрипте обучения price_jump_train_OneCFocalL.py.

Used grep and ran terminal command
Теперь давайте посмотрим подробнее на подготовку фичей:

Ran: sed -n '148,175p' price_jump_train_OneCFocalL.py
Теперь проверим значение USE_STANDARD_SCALER и что происходит при обучении:

Ran: sed -n '217,225p' price_jump_train_OneCFocalL.py
Отлично! Теперь у меня полная картина. Вот что происходит в обучении:

Анализ нормализации в price_jump_train_OneCFocalL.py:
1. StandardScaler - НЕ ИСПОЛЬЗУЕТСЯ!
USE_STANDARD_SCALER = False  # Строка 78
StandardScaler выключен, хотя код для него есть. Это значит, что данные идут в модель БЕЗ стандартизации.

2. Обрезка выбросов - ЕСТЬ!
# Строки 159-160 в __getitem__:
upper_w = np.clip(highs_w - np.maximum(opens_w, closes_w), 0.0, None)  # Обрезка снизу от 0
lower_w = np.clip(np.minimum(opens_w, closes_w) - lows_w, 0.0, None)   # Обрезка снизу от 0
Обрезаются только отрицательные значения для хвостов свечей (что логично, они не могут быть отрицательными).

3. Порядок фичей в обучении:
# Строки 163-171:
x_seq = np.stack([
    closes_w,   # 0
    opens_w,    # 1
    highs_w,    # 2
    lows_w,     # 3
    vols_w,     # 4
    ratio_up,   # 5 (upper_wick/body)
    ratio_dn,   # 6 (lower_wick/body)
], axis=0)
Правильный код для инференса БЕЗ нормализации:
class LSTMPredictor:
    def __init__(self, model_path='lstm_jump_PRAUC.pt', seq_len=30):
        self.seq_len = seq_len
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._load_model(model_path)
        
    def prepare_features(self, candles_list):
        """
        Готовит фичи ТОЧНО КАК В ОБУЧЕНИИ - без нормализации!
        """
        if len(candles_list) < self.seq_len:
            return None
        
        # Берем последние seq_len свечей
        recent = list(candles_list)[-self.seq_len:]
        
        # Извлекаем массивы
        closes = np.array([c['close'] for c in recent], dtype=np.float32)
        opens = np.array([c['open'] for c in recent], dtype=np.float32)
        highs = np.array([c['high'] for c in recent], dtype=np.float32)
        lows = np.array([c['low'] for c in recent], dtype=np.float32)
        volumes = np.array([c['volume'] for c in recent], dtype=np.float32)
        
        # Вычисляем дополнительные фичи (КАК В ОБУЧЕНИИ!)
        body = np.abs(closes - opens) + 1e-12  # +1e-12 для избежания деления на 0
        
        # Обрезка отрицательных значений для хвостов
        upper_wick = np.clip(highs - np.maximum(opens, closes), 0.0, None)
        lower_wick = np.clip(np.minimum(opens, closes) - lows, 0.0, None)
        
        # Отношения (как в обучении)
        ratio_up = upper_wick / body
        ratio_dn = lower_wick / body
        
        # Собираем в правильном порядке (ВАЖНО! Как в обучении)
        features = np.stack([
            closes,    # 0
            opens,     # 1
            highs,     # 2
            lows,      # 3
            volumes,   # 4
            ratio_up,  # 5
            ratio_dn   # 6
        ], axis=0).astype(np.float32)  # Shape: (7, seq_len)
        
        # Транспонируем для модели
        features = features.T  # Shape: (seq_len, 7)
        
        return features
    
    def predict(self, candles):
        """Предсказание БЕЗ нормализации, как в обучении"""
        features = self.prepare_features(candles)
        if features is None:
            return None, "Недостаточно данных"
        
        # В тензор и добавляем batch dimension
        x = torch.from_numpy(features).unsqueeze(0).to(self.device)  # (1, seq_len, 7)
        
        # НЕ НУЖНА транспозиция, так как мы уже сделали .T выше
        
        with torch.no_grad():
            self.model.eval()
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            prob_jump = probs[0, 1].item()
        
        return prob_jump, "OK"
"""