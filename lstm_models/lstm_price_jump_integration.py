"""
Интеграция LSTM модели с существующими данными о ценовых скачках
Пример адаптации LSTM для работы с данными из основного проекта
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lstm_time_series import LSTMTimeSeriesPredictor
from lstm_multivariate import MultivariateLSTMPredictor
import json

class PriceJumpLSTMAnalyzer:
    """
    Анализатор ценовых скачков с использованием LSTM
    """
    
    def __init__(self):
        self.univariate_predictor = None
        self.multivariate_predictor = None
        self.data = None
        
    def load_okx_data(self, file_path):
        """Загрузка данных OKX из JSON файла"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Преобразование в DataFrame
            df = pd.DataFrame(data)
            
            # Преобразование timestamp в datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
            
            # Переименование колонок для совместимости
            column_mapping = {
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            return df
            
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            return None
    
    def create_price_jump_features(self, data):
        """Создание признаков для анализа ценовых скачков"""
        df = data.copy()
        
        # Основные признаки ценовых движений
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Признаки скачков
        df['price_jump'] = np.abs(df['returns']) > (2 * df['volatility'])
        df['jump_magnitude'] = np.abs(df['returns'])
        df['jump_direction'] = np.sign(df['returns'])
        
        # Технические индикаторы для скачков
        df['rsi'] = self.calculate_rsi(df['Close'])
        df['bollinger_position'] = self.calculate_bollinger_position(df['Close'])
        df['volume_spike'] = df['Volume'] > df['Volume'].rolling(window=20).mean() * 2
        
        # Временные признаки
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        return df.dropna()
    
    def calculate_rsi(self, prices, window=14):
        """Расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_position(self, prices, window=20):
        """Позиция цены относительно полос Боллинджера"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        
        position = (prices - lower_band) / (upper_band - lower_band)
        return position
    
    def predict_price_jumps(self, data, jump_threshold=0.02):
        """Предсказание ценовых скачков"""
        # Подготовка данных с признаками скачков
        df = self.create_price_jump_features(data)
        
        # Создание целевой переменной (будет ли скачок в следующий период)
        df['future_jump'] = (np.abs(df['returns'].shift(-1)) > jump_threshold).astype(int)
        
        # Признаки для модели
        feature_columns = [
            'Close', 'Volume', 'returns', 'volatility', 'rsi',
            'bollinger_position', 'jump_magnitude', 'hour'
        ]
        
        # Фильтрация доступных признаков
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Создание многомерной LSTM модели
        predictor = MultivariateLSTMPredictor(
            sequence_length=24,  # 24 часа для часовых данных
            lstm_units=32,
            dense_units=16,
            dropout_rate=0.3
        )
        
        # Построение модели
        model = predictor.build_model()
        
        # Обучение на предсказании скачков
        history, test_data = predictor.train(
            df, 
            target_column='future_jump',
            feature_columns=available_features,
            epochs=30,
            batch_size=16
        )
        
        return predictor, history, df
    
    def analyze_jump_patterns(self, data):
        """Анализ паттернов ценовых скачков"""
        df = self.create_price_jump_features(data)
        
        # Статистика скачков
        jump_stats = {
            'total_jumps': df['price_jump'].sum(),
            'jump_frequency': df['price_jump'].mean(),
            'avg_jump_magnitude': df[df['price_jump']]['jump_magnitude'].mean(),
            'max_jump_magnitude': df['jump_magnitude'].max()
        }
        
        print("=== Анализ ценовых скачков ===")
        print(f"Общее количество скачков: {jump_stats['total_jumps']}")
        print(f"Частота скачков: {jump_stats['jump_frequency']:.4f}")
        print(f"Средняя величина скачка: {jump_stats['avg_jump_magnitude']:.4f}")
        print(f"Максимальная величина скачка: {jump_stats['max_jump_magnitude']:.4f}")
        
        # Визуализация паттернов
        self.plot_jump_analysis(df)
        
        return jump_stats
    
    def plot_jump_analysis(self, df):
        """Визуализация анализа скачков"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # График цены с выделенными скачками
        axes[0, 0].plot(df.index, df['Close'], alpha=0.7, label='Цена')
        jump_points = df[df['price_jump']]
        axes[0, 0].scatter(jump_points.index, jump_points['Close'], 
                          color='red', s=20, alpha=0.7, label='Скачки')
        axes[0, 0].set_title('Цена с выделенными скачками')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Распределение величины скачков
        axes[0, 1].hist(df[df['price_jump']]['jump_magnitude'], bins=30, alpha=0.7)
        axes[0, 1].set_title('Распределение величины скачков')
        axes[0, 1].set_xlabel('Величина скачка')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Скачки по часам
        hourly_jumps = df.groupby('hour')['price_jump'].sum()
        axes[1, 0].bar(hourly_jumps.index, hourly_jumps.values, alpha=0.7)
        axes[1, 0].set_title('Распределение скачков по часам')
        axes[1, 0].set_xlabel('Час дня')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Волатильность и скачки
        axes[1, 1].scatter(df['volatility'], df['jump_magnitude'], alpha=0.5)
        axes[1, 1].set_title('Волатильность vs Величина скачка')
        axes[1, 1].set_xlabel('Волатильность')
        axes[1, 1].set_ylabel('Величина скачка')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_trading_strategy(self, data, predictor):
        """Создание торговой стратегии на основе предсказаний LSTM"""
        df = self.create_price_jump_features(data)
        
        # Получение предсказаний
        predictions = predictor.predict(df, target_column='Close')
        
        # Создание сигналов
        df['predicted_price'] = np.nan
        df.iloc[predictor.sequence_length:, df.columns.get_loc('predicted_price')] = predictions
        
        # Торговые сигналы
        df['price_change_prediction'] = df['predicted_price'] / df['Close'] - 1
        df['signal'] = 0
        df.loc[df['price_change_prediction'] > 0.01, 'signal'] = 1  # Покупка
        df.loc[df['price_change_prediction'] < -0.01, 'signal'] = -1  # Продажа
        
        # Расчет доходности стратегии
        df['strategy_returns'] = df['signal'].shift(1) * df['returns']
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()
        
        return df
    
    def plot_strategy_performance(self, strategy_df):
        """Визуализация результатов торговой стратегии"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Кумулятивная доходность
        ax1.plot(strategy_df.index, strategy_df['cumulative_returns'], 
                label='Buy & Hold', alpha=0.7)
        ax1.plot(strategy_df.index, strategy_df['cumulative_strategy'], 
                label='LSTM Strategy', alpha=0.7)
        ax1.set_title('Сравнение стратегий')
        ax1.set_ylabel('Кумулятивная доходность')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Торговые сигналы
        ax2.plot(strategy_df.index, strategy_df['Close'], alpha=0.7, label='Цена')
        buy_signals = strategy_df[strategy_df['signal'] == 1]
        sell_signals = strategy_df[strategy_df['signal'] == -1]
        
        ax2.scatter(buy_signals.index, buy_signals['Close'], 
                   color='green', marker='^', s=50, label='Покупка')
        ax2.scatter(sell_signals.index, sell_signals['Close'], 
                   color='red', marker='v', s=50, label='Продажа')
        
        ax2.set_title('Торговые сигналы')
        ax2.set_ylabel('Цена')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """Демонстрация работы с ценовыми скачками"""
    print("=== LSTM анализ ценовых скачков ===")
    
    analyzer = PriceJumpLSTMAnalyzer()
    
    # Попробуем загрузить реальные данные из проекта
    data_files = [
        '../okx/btc_data.json',
        '../test/sample_data.json',
        'sample_crypto_data.json'
    ]
    
    data = None
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"Загружаем данные из {file_path}")
            data = analyzer.load_okx_data(file_path)
            if data is not None:
                break
    
    # Если реальные данные не найдены, создаем синтетические
    if data is None:
        print("Создаем синтетические данные для демонстрации...")
        data = create_synthetic_crypto_data()
    
    print(f"Загружено {len(data)} записей данных")
    print(f"Период: {data.index[0]} - {data.index[-1]}")
    
    # Анализ паттернов скачков
    jump_stats = analyzer.analyze_jump_patterns(data)
    
    # Предсказание скачков с помощью LSTM
    print("\nОбучение LSTM модели для предсказания скачков...")
    predictor, history, processed_data = analyzer.predict_price_jumps(data)
    
    # Визуализация обучения
    predictor.plot_training_history(history)
    
    # Создание торговой стратегии
    print("\nСоздание торговой стратегии...")
    strategy_df = analyzer.create_trading_strategy(data, predictor)
    
    # Оценка стратегии
    total_return = strategy_df['cumulative_returns'].iloc[-1] - 1
    strategy_return = strategy_df['cumulative_strategy'].iloc[-1] - 1
    
    print(f"\nРезультаты стратегии:")
    print(f"Buy & Hold доходность: {total_return:.4f} ({total_return*100:.2f}%)")
    print(f"LSTM стратегия доходность: {strategy_return:.4f} ({strategy_return*100:.2f}%)")
    print(f"Превышение: {(strategy_return - total_return)*100:.2f}%")
    
    # Визуализация результатов
    analyzer.plot_strategy_performance(strategy_df)


def create_synthetic_crypto_data(n_points=2000):
    """Создание синтетических криптовалютных данных"""
    np.random.seed(42)
    
    # Базовая цена с трендом и волатильностью
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='H')
    
    # Создание реалистичных криптовалютных данных
    base_price = 30000  # Базовая цена BTC
    trend = np.cumsum(np.random.normal(0, 0.001, n_points))
    
    # Добавляем периодические паттерны и волатильность
    volatility = 0.02 + 0.01 * np.sin(np.arange(n_points) * 2 * np.pi / 168)  # Недельная сезонность
    returns = np.random.normal(0, volatility)
    
    # Добавляем редкие большие скачки (имитация новостных событий)
    jump_probability = 0.005
    jumps = np.random.binomial(1, jump_probability, n_points)
    jump_sizes = np.random.normal(0, 0.05, n_points) * jumps
    returns += jump_sizes
    
    # Создание цен
    log_prices = np.log(base_price) + np.cumsum(returns) + trend
    prices = np.exp(log_prices)
    
    # Создание OHLCV данных
    data = {
        'Open': prices * (1 + np.random.normal(0, 0.001, n_points)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
        'Close': prices,
        'Volume': np.random.exponential(100, n_points)
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Корректировка High и Low
    df['High'] = np.maximum(df[['Open', 'Close']].max(axis=1), df['High'])
    df['Low'] = np.minimum(df[['Open', 'Close']].min(axis=1), df['Low'])
    
    return df


if __name__ == "__main__":
    main()