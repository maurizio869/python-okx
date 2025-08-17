"""
Многомерная LSTM модель для временных рядов
Пример использования LSTM для предсказания с несколькими входными признаками
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
import ta

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Установим random seed для воспроизводимости
np.random.seed(42)
tf.random.set_seed(42)

class MultivariateLSTMPredictor:
    """
    Многомерный LSTM предиктор для временных рядов
    """
    
    def __init__(self, sequence_length=60, n_features=5, lstm_units=50, 
                 dense_units=25, dropout_rate=0.2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_names = []
        
    def build_model(self):
        """Создание архитектуры многомерной LSTM модели"""
        model = Sequential([
            # Первый LSTM слой
            LSTM(units=self.lstm_units, 
                 return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features)),
            Dropout(self.dropout_rate),
            
            # Второй LSTM слой
            LSTM(units=self.lstm_units, return_sequences=True),
            Dropout(self.dropout_rate),
            
            # Третий LSTM слой
            LSTM(units=self.lstm_units, return_sequences=False),
            Dropout(self.dropout_rate),
            
            # Dense слои
            Dense(units=self.dense_units, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(units=self.dense_units // 2, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(units=1)  # Выходной слой для предсказания одного значения
        ])
        
        # Компиляция модели
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def prepare_features(self, data):
        """Подготовка технических индикаторов и признаков"""
        df = data.copy()
        
        # Базовые признаки
        df['returns'] = df['Close'].pct_change()
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        # Технические индикаторы
        df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['Close'])
        df['bollinger_high'] = ta.volatility.bollinger_hband(df['Close'])
        df['bollinger_low'] = ta.volatility.bollinger_lband(df['Close'])
        df['volume_sma'] = ta.trend.sma_indicator(df['Volume'], window=20)
        
        # Отношения к скользящим средним
        df['close_sma_ratio'] = df['Close'] / df['sma_20']
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Волатильность
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Удаляем NaN значения
        df = df.dropna()
        
        return df
    
    def prepare_data(self, data, target_column='Close', feature_columns=None):
        """Подготовка многомерных данных для обучения LSTM"""
        # Добавление технических индикаторов
        df = self.prepare_features(data)
        
        # Выбор признаков
        if feature_columns is None:
            feature_columns = [
                'Close', 'Volume', 'returns', 'rsi', 'macd',
                'close_sma_ratio', 'volatility', 'high_low_ratio'
            ]
        
        # Фильтрация доступных колонок
        available_columns = [col for col in feature_columns if col in df.columns]
        self.feature_names = available_columns
        self.n_features = len(available_columns)
        
        # Извлечение данных
        feature_data = df[available_columns].values
        target_data = df[target_column].values
        
        # Нормализация данных
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Создание последовательностей
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(target_data[i])
        
        X, y = np.array(X), np.array(y)
        
        return X, y, df.index[self.sequence_length:]
    
    def train(self, data, target_column='Close', feature_columns=None, 
              test_size=0.2, epochs=100, batch_size=32, verbose=1):
        """Обучение модели"""
        # Подготовка данных
        X, y, dates = self.prepare_data(data, target_column, feature_columns)
        
        # Разделение на train/test
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Обучение
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False
        )
        
        return history, (X_test, y_test, dates[train_size:])
    
    def predict(self, data, target_column='Close', feature_columns=None):
        """Предсказание значений"""
        X, _, _ = self.prepare_data(data, target_column, feature_columns)
        predictions = self.model.predict(X)
        return predictions.flatten()
    
    def evaluate(self, data, target_column='Close', feature_columns=None, test_size=0.2):
        """Оценка модели"""
        X, y, dates = self.prepare_data(data, target_column, feature_columns)
        
        # Разделение на train/test
        train_size = int(len(X) * (1 - test_size))
        X_test, y_test = X[train_size:], y[train_size:]
        
        # Предсказания
        predictions = self.model.predict(X_test).flatten()
        
        # Метрики
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'actual': y_test,
            'dates': dates[train_size:]
        }
    
    def plot_feature_importance(self, data, target_column='Close', feature_columns=None):
        """Анализ важности признаков через корреляцию"""
        df = self.prepare_features(data)
        
        if feature_columns is None:
            feature_columns = self.feature_names
        
        # Вычисление корреляций с целевой переменной
        correlations = df[feature_columns + [target_column]].corr()[target_column].drop(target_column)
        correlations = correlations.abs().sort_values(ascending=True)
        
        plt.figure(figsize=(10, 8))
        correlations.plot(kind='barh')
        plt.title('Корреляция признаков с целевой переменной')
        plt.xlabel('Абсолютная корреляция')
        plt.tight_layout()
        plt.show()
        
        return correlations
    
    def plot_predictions(self, data, target_column='Close', feature_columns=None, 
                        test_size=0.2):
        """Визуализация предсказаний"""
        results = self.evaluate(data, target_column, feature_columns, test_size)
        
        plt.figure(figsize=(15, 8))
        
        # Исходные данные
        dates = results['dates']
        plt.plot(dates, results['actual'], label='Фактические значения', alpha=0.7)
        plt.plot(dates, results['predictions'], label='Предсказания', alpha=0.7)
        
        plt.title('Многомерное LSTM прогнозирование')
        plt.xlabel('Дата')
        plt.ylabel('Цена')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return results
    
    def plot_training_history(self, history):
        """Визуализация процесса обучения"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # MAE
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()


def load_enhanced_stock_data(symbol='AAPL', period='2y'):
    """Загрузка расширенных данных о ценах акций"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return create_synthetic_multivariate_data()

def create_synthetic_multivariate_data(n_points=1000):
    """Создание синтетических многомерных временных данных"""
    np.random.seed(42)
    
    # Создание базового временного ряда
    time = np.arange(n_points)
    base_price = 100 + 0.02 * time + 10 * np.sin(2 * np.pi * time / 50)
    
    # Создание OHLCV данных
    data = {
        'Open': base_price + np.random.normal(0, 1, n_points),
        'High': base_price + np.abs(np.random.normal(2, 1, n_points)),
        'Low': base_price - np.abs(np.random.normal(2, 1, n_points)),
        'Close': base_price + np.random.normal(0, 1.5, n_points),
        'Volume': np.random.exponential(1000000, n_points)
    }
    
    # Создание DataFrame
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='D')
    df = pd.DataFrame(data, index=dates)
    
    # Корректировка High и Low
    df['High'] = np.maximum(df[['Open', 'Close']].max(axis=1), df['High'])
    df['Low'] = np.minimum(df[['Open', 'Close']].min(axis=1), df['Low'])
    
    return df


def main():
    """Главная функция для демонстрации работы многомерного LSTM"""
    print("=== Multivariate LSTM Time Series Prediction Demo ===")
    
    # Загрузка данных
    print("Загрузка данных...")
    try:
        data = load_enhanced_stock_data('AAPL', '2y')
        print("Загружены данные по акциям Apple")
    except:
        print("Создание синтетических данных...")
        data = create_synthetic_multivariate_data()
    
    print(f"Размер данных: {data.shape}")
    print(f"Колонки: {list(data.columns)}")
    print(data.head())
    
    # Визуализация исходных данных
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(data.index, data['Close'])
    axes[0, 0].set_title('Цена закрытия')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(data.index, data['Volume'])
    axes[0, 1].set_title('Объем торгов')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(data.index, data['High'] - data['Low'])
    axes[1, 0].set_title('Дневной диапазон (High - Low)')
    axes[1, 0].grid(True, alpha=0.3)
    
    returns = data['Close'].pct_change().dropna()
    axes[1, 1].hist(returns, bins=50, alpha=0.7)
    axes[1, 1].set_title('Распределение доходности')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Создание и настройка модели
    predictor = MultivariateLSTMPredictor(
        sequence_length=60,
        lstm_units=64,
        dense_units=32,
        dropout_rate=0.2
    )
    
    # Построение модели
    model = predictor.build_model()
    print("\nАрхитектура модели:")
    model.summary()
    
    # Анализ важности признаков
    print("\nАнализ важности признаков:")
    correlations = predictor.plot_feature_importance(data)
    print("\nТоп-5 самых важных признаков:")
    print(correlations.tail().to_string())
    
    # Обучение
    print("\nНачинаем обучение...")
    history, test_data = predictor.train(
        data,
        target_column='Close',
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Визуализация процесса обучения
    predictor.plot_training_history(history)
    
    # Оценка и визуализация результатов
    print("\nОценка модели:")
    results = predictor.plot_predictions(data, target_column='Close')
    
    print(f"\nИспользованные признаки ({len(predictor.feature_names)}):")
    for i, feature in enumerate(predictor.feature_names, 1):
        print(f"{i}. {feature}")


if __name__ == "__main__":
    main()