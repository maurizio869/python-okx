# lstm_predictor.py
# Класс для инференса обученной LSTM модели
# Версия без нормализации (как в обучении с USE_STANDARD_SCALER = False)

import numpy as np
import torch
import torch.nn as nn
import json
from pathlib import Path
from collections import deque
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional, Union


class LSTMModel(nn.Module):
    """Архитектура модели - должна точно совпадать с обучением!"""
    def __init__(self, input_size: int = 7, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.35):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.fc(out)


class LSTMPredictor:
    """
    Класс для предсказания скачков цены с использованием обученной LSTM модели.
    Работает БЕЗ нормализации, как в обучении с USE_STANDARD_SCALER = False.
    """
    
    def __init__(self, 
                 model_path: str = 'lstm_jump_PRAUC.pt',
                 meta_path: Optional[str] = None,
                 seq_len: int = 30,
                 pred_window: int = 5,
                 jump_threshold: float = 0.0035,
                 device: Optional[str] = None):
        """
        Инициализация предиктора.
        
        Args:
            model_path: Путь к файлу модели .pt
            meta_path: Путь к мета-информации (если None, будет model_path с .meta.json)
            seq_len: Длина последовательности (количество минутных свечей)
            pred_window: Окно предсказания в минутах
            jump_threshold: Порог для определения скачка (0.35% по умолчанию)
            device: Устройство для вычислений ('cuda', 'cpu' или None для авто)
        """
        self.seq_len = seq_len
        self.pred_window = pred_window
        self.jump_threshold = jump_threshold
        
        # Определяем устройство
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Пути к файлам
        self.model_path = Path(model_path)
        if meta_path is None:
            self.meta_path = self.model_path.with_suffix('.meta.json')
        else:
            self.meta_path = Path(meta_path)
        
        # Загружаем метаданные
        self.meta = self._load_meta()
        
        # Загружаем модель
        self.model = self._load_model()
        
        # Буфер для последних свечей
        self.candle_buffer = deque(maxlen=seq_len)
        
        # Порог классификации (из meta или дефолт)
        self.classification_threshold = self.meta.get('best_threshold', 0.5)
        
        # Статистика
        self.total_predictions = 0
        self.positive_predictions = 0
        
        print(f"✅ Модель загружена на {self.device}")
        print(f"   Порог классификации: {self.classification_threshold:.4f}")
        print(f"   Длина последовательности: {self.seq_len}")
        print(f"   Окно предсказания: {self.pred_window} минут")
    
    def _load_meta(self) -> Dict:
        """Загружает метаданные модели если файл существует"""
        if self.meta_path.exists():
            try:
                with open(self.meta_path, 'r') as f:
                    meta = json.load(f)
                print(f"📊 Метаданные загружены из {self.meta_path}")
                if 'best_threshold' in meta:
                    print(f"   Best threshold: {meta['best_threshold']:.4f}")
                if 'pr_auc' in meta:
                    print(f"   PR-AUC: {meta['pr_auc']:.4f}")
                return meta
            except Exception as e:
                print(f"⚠️ Не удалось загрузить метаданные: {e}")
                return {}
        else:
            print(f"ℹ️ Файл метаданных не найден: {self.meta_path}")
            return {}
    
    def _load_model(self) -> nn.Module:
        """Загружает обученную модель"""
        # Создаем модель с правильной архитектурой
        model = LSTMModel(
            input_size=7,  # Close, Open, High, Low, Volume + 2 ratio фичи
            hidden_size=64,
            num_layers=2,
            dropout=0.35  # DEFAULT_DROPOUT из обучения
        )
        
        # Загружаем веса
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Проверяем формат checkpoint
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✅ Загружены веса из checkpoint['model_state_dict']")
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    print(f"✅ Загружены веса из checkpoint['state_dict']")
                else:
                    # Пробуем загрузить напрямую если это state_dict
                    model.load_state_dict(checkpoint)
                    print(f"✅ Загружены веса напрямую из checkpoint")
            else:
                # Если checkpoint это сразу state_dict
                model.load_state_dict(checkpoint)
                print(f"✅ Загружены веса модели")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            raise
        
        model.to(self.device)
        model.eval()  # Переводим в режим инференса
        
        # Подсчет параметров
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📈 Всего параметров в модели: {total_params:,}")
        
        return model
    
    def add_candle(self, candle: Dict[str, float]) -> None:
        """
        Добавляет новую свечу в буфер.
        
        Args:
            candle: Словарь с ключами 'open', 'high', 'low', 'close', 'volume'
                   Опционально: 'timestamp' для логирования
        """
        required_keys = ['open', 'high', 'low', 'close', 'volume']
        for key in required_keys:
            if key not in candle:
                raise ValueError(f"Отсутствует обязательное поле '{key}' в свече")
        
        self.candle_buffer.append(candle)
    
    def prepare_features(self, candles_list: Union[deque, List]) -> Optional[np.ndarray]:
        """
        Подготавливает фичи из сырых OHLCV данных.
        ТОЧНО КАК В ОБУЧЕНИИ - без нормализации!
        
        Args:
            candles_list: Список или deque со свечами
            
        Returns:
            numpy array формы (seq_len, 7) или None если недостаточно данных
        """
        if len(candles_list) < self.seq_len:
            return None
        
        # Берем последние seq_len свечей
        recent = list(candles_list)[-self.seq_len:]
        
        # Извлекаем массивы (ВАЖНО: порядок как в обучении!)
        closes = np.array([c['close'] for c in recent], dtype=np.float32)
        opens = np.array([c['open'] for c in recent], dtype=np.float32)
        highs = np.array([c['high'] for c in recent], dtype=np.float32)
        lows = np.array([c['low'] for c in recent], dtype=np.float32)
        volumes = np.array([c['volume'] for c in recent], dtype=np.float32)
        
        # Вычисляем дополнительные фичи (КАК В ОБУЧЕНИИ!)
        body = np.abs(closes - opens) + 1e-12  # +1e-12 для избежания деления на 0
        
        # Обрезка отрицательных значений для хвостов (как в обучении)
        upper_wick = np.clip(highs - np.maximum(opens, closes), 0.0, None)
        lower_wick = np.clip(np.minimum(opens, closes) - lows, 0.0, None)
        
        # Отношения хвостов к телу
        ratio_up = upper_wick / body
        ratio_dn = lower_wick / body
        
        # Собираем в правильном порядке (КРИТИЧЕСКИ ВАЖНО!)
        # Порядок из обучения: closes, opens, highs, lows, volumes, ratio_up, ratio_dn
        features = np.stack([
            closes,    # 0
            opens,     # 1
            highs,     # 2
            lows,      # 3
            volumes,   # 4
            ratio_up,  # 5 (upper_wick/body)
            ratio_dn   # 6 (lower_wick/body)
        ], axis=0).astype(np.float32)  # Shape: (7, seq_len)
        
        # Транспонируем для правильной формы
        features = features.T  # Shape: (seq_len, 7)
        
        return features
    
    def predict(self, return_logits: bool = False) -> Tuple[Optional[float], str, Optional[Dict]]:
        """
        Делает предсказание на основе последних seq_len свечей из буфера.
        
        Args:
            return_logits: Если True, возвращает также сырые логиты
            
        Returns:
            Кортеж (вероятность_скачка, статус, дополнительная_информация)
            - вероятность_скачка: float от 0 до 1 или None при ошибке
            - статус: строка со статусом ("OK", "Недостаточно данных", etc.)
            - дополнительная_информация: словарь с дополнительными данными
        """
        # Проверяем достаточность данных
        if len(self.candle_buffer) < self.seq_len:
            return None, f"Недостаточно данных: {len(self.candle_buffer)}/{self.seq_len}", None
        
        # Подготавливаем фичи
        features = self.prepare_features(self.candle_buffer)
        if features is None:
            return None, "Ошибка подготовки фичей", None
        
        # Конвертируем в тензор и добавляем batch dimension
        x = torch.from_numpy(features).unsqueeze(0).to(self.device)  # (1, seq_len, 7)
        
        # Предсказание
        with torch.no_grad():
            self.model.eval()  # Убеждаемся что модель в eval режиме
            
            # Forward pass
            logits = self.model(x)  # (1, 2)
            probs = torch.softmax(logits, dim=1)  # (1, 2)
            
            # Вероятность класса 1 (скачок)
            prob_jump = probs[0, 1].item()
            prob_no_jump = probs[0, 0].item()
        
        # Обновляем статистику
        self.total_predictions += 1
        if prob_jump >= self.classification_threshold:
            self.positive_predictions += 1
        
        # Собираем дополнительную информацию
        extra_info = {
            'prob_no_jump': prob_no_jump,
            'prob_jump': prob_jump,
            'threshold': self.classification_threshold,
            'signal': 'BUY' if prob_jump >= self.classification_threshold else 'HOLD',
            'confidence': abs(prob_jump - self.classification_threshold),
            'last_close': self.candle_buffer[-1]['close'],
            'total_predictions': self.total_predictions,
            'positive_rate': self.positive_predictions / max(1, self.total_predictions)
        }
        
        if return_logits:
            extra_info['logits'] = logits.cpu().numpy().tolist()
        
        return prob_jump, "OK", extra_info
    
    def predict_batch(self, candles_history: List[Dict]) -> List[Dict]:
        """
        Делает предсказания для исторических данных (для бэктеста).
        Проходит по истории последовательно, делая предсказание для каждой точки.
        
        Args:
            candles_history: Список свечей для анализа
            
        Returns:
            Список предсказаний с метаданными
        """
        predictions = []
        
        # Очищаем буфер перед началом
        self.candle_buffer.clear()
        self.total_predictions = 0
        self.positive_predictions = 0
        
        # Проходим по истории
        for i, candle in enumerate(candles_history):
            self.add_candle(candle)
            
            # Начинаем предсказывать когда накопили достаточно данных
            if i >= self.seq_len - 1:
                prob, status, info = self.predict()
                
                if prob is not None:
                    predictions.append({
                        'index': i,
                        'timestamp': candle.get('timestamp', i),
                        'probability': prob,
                        'signal': info['signal'],
                        'confidence': info['confidence'],
                        'close': candle['close'],
                        'open': candle['open'],
                        'high': candle['high'],
                        'low': candle['low'],
                        'volume': candle['volume']
                    })
        
        return predictions
    
    def get_required_features_info(self) -> Dict:
        """
        Возвращает информацию о требуемых фичах для модели.
        Полезно для отладки и документации.
        """
        return {
            'num_features': 7,
            'feature_names': [
                'close',
                'open', 
                'high',
                'low',
                'volume',
                'upper_wick_to_body_ratio',
                'lower_wick_to_body_ratio'
            ],
            'sequence_length': self.seq_len,
            'normalization': 'None (USE_STANDARD_SCALER = False)',
            'clipping': 'Only negative wick values clipped to 0',
            'expected_shape': f'({self.seq_len}, 7)',
            'notes': [
                'Порядок фичей критически важен!',
                'Close идет первым, не OHLCV',
                'Хвосты обрезаются снизу от 0',
                'Нормализация НЕ используется',
                'Body вычисляется как |close - open| + 1e-12'
            ]
        }
    
    def reset_buffer(self) -> None:
        """Очищает буфер свечей и сбрасывает статистику"""
        self.candle_buffer.clear()
        self.total_predictions = 0
        self.positive_predictions = 0
        print("🔄 Буфер очищен, статистика сброшена")
    
    def get_stats(self) -> Dict:
        """Возвращает статистику предсказаний"""
        return {
            'total_predictions': self.total_predictions,
            'positive_predictions': self.positive_predictions,
            'positive_rate': self.positive_predictions / max(1, self.total_predictions),
            'buffer_size': len(self.candle_buffer),
            'threshold': self.classification_threshold,
            'device': self.device
        }


# ===== ПРИМЕР ИСПОЛЬЗОВАНИЯ =====
if __name__ == "__main__":
    # Пример использования класса
    
    # 1. Инициализация
    predictor = LSTMPredictor(
        model_path='lstm_jump_PRAUC.pt',
        seq_len=30,
        device='cuda'  # или 'cpu'
    )
    
    # 2. Информация о модели
    print("\n📋 Требования к фичам:")
    info = predictor.get_required_features_info()
    for key, value in info.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    - {item}")
        else:
            print(f"  {key}: {value}")
    
    # 3. Симуляция получения данных
    import random
    
    print("\n🔄 Симуляция торговли...")
    
    # Генерируем тестовые свечи
    for i in range(35):  # Нужно минимум 30 для первого предсказания
        base_price = 50000 + random.uniform(-1000, 1000)
        candle = {
            'open': base_price + random.uniform(-100, 100),
            'high': base_price + random.uniform(50, 200),
            'low': base_price - random.uniform(50, 200),
            'close': base_price + random.uniform(-100, 100),
            'volume': 100 + random.uniform(0, 50),
            'timestamp': i
        }
        
        # Добавляем свечу
        predictor.add_candle(candle)
        
        # Пробуем предсказать
        if i >= 29:  # После накопления 30 свечей
            prob, status, info = predictor.predict()
            
            if prob is not None:
                print(f"\n⏱️ Свеча #{i+1}")
                print(f"  Цена закрытия: ${candle['close']:.2f}")
                print(f"  Вероятность скачка: {prob:.1%}")
                print(f"  Сигнал: {info['signal']}")
                print(f"  Уверенность: {info['confidence']:.3f}")
                
                if info['signal'] == 'BUY':
                    print(f"  🚀 СИГНАЛ НА ПОКУПКУ!")
                    print(f"     Ожидаемый скачок: +{predictor.jump_threshold*100:.2f}%")
                    print(f"     Take Profit: ${candle['close'] * (1 + predictor.jump_threshold):.2f}")
    
    # 4. Финальная статистика
    print("\n📊 Статистика сессии:")
    stats = predictor.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")