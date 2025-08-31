# lstm_predictor_class.py
# Класс для инференса обученной LSTM модели
# БЕЗ нормализации (USE_STANDARD_SCALER = False в обучении)

import numpy as np
import torch

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