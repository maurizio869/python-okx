"""
LSTM модель для прогнозирования временных рядов
Пример использования LSTM для предсказания цен акций или других временных данных
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Установим random seed для воспроизводимости
np.random.seed(42)
tf.random.set_seed(42)

class LSTMTimeSeriesPredictor:
    """
    LSTM предиктор для временных рядов
    """
    
    def __init__(self, sequence_length=60, n_features=1, lstm_units=50, 
                 dense_units=25, dropout_rate=0.2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def build_model(self):
        """Создание архитектуры LSTM модели"""
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
            Dense(units=1)  # Выходной слой для предсказания одного значения
        ])
        
        # Компиляция модели
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def prepare_data(self, data, target_column=None):
        """Подготовка данных для обучения LSTM"""
        if isinstance(data, pd.DataFrame):
            if target_column:
                values = data[target_column].values
            else:
                values = data.iloc[:, 0].values  # Берем первый столбец
        else:
            values = data
            
        # Нормализация данных
        values = values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(values)
        
        # Создание последовательностей
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Reshape для LSTM (samples, time steps, features)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def train(self, data, target_column=None, test_size=0.2, epochs=100, 
              batch_size=32, verbose=1):
        """Обучение модели"""
        # Подготовка данных
        X, y = self.prepare_data(data, target_column)
        
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
            shuffle=False  # Важно для временных рядов
        )
        
        return history, (X_test, y_test)
    
    def predict(self, data, target_column=None, steps_ahead=1):
        """Предсказание будущих значений"""
        X, _ = self.prepare_data(data, target_column)
        
        # Предсказание
        predictions = self.model.predict(X)
        
        # Обратная нормализация
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions
    
    def predict_future(self, data, target_column=None, steps_ahead=30):
        """Предсказание на несколько шагов вперед"""
        if isinstance(data, pd.DataFrame):
            if target_column:
                last_sequence = data[target_column].values[-self.sequence_length:]
            else:
                last_sequence = data.iloc[-self.sequence_length:, 0].values
        else:
            last_sequence = data[-self.sequence_length:]
        
        # Нормализация последней последовательности
        last_sequence = last_sequence.reshape(-1, 1)
        last_sequence_scaled = self.scaler.transform(last_sequence)
        
        # Предсказание будущих значений
        future_predictions = []
        current_sequence = last_sequence_scaled.flatten()
        
        for _ in range(steps_ahead):
            # Reshape для предсказания
            X_pred = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Предсказание следующего значения
            next_pred = self.model.predict(X_pred, verbose=0)[0, 0]
            future_predictions.append(next_pred)
            
            # Обновление последовательности
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        # Обратная нормализация
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = self.scaler.inverse_transform(future_predictions)
        
        return future_predictions.flatten()
    
    def evaluate(self, data, target_column=None, test_size=0.2):
        """Оценка модели"""
        X, y = self.prepare_data(data, target_column)
        
        # Разделение на train/test
        train_size = int(len(X) * (1 - test_size))
        X_test, y_test = X[train_size:], y[train_size:]
        
        # Предсказания
        predictions = self.model.predict(X_test)
        
        # Обратная нормализация для вычисления метрик
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        predictions_actual = self.scaler.inverse_transform(predictions).flatten()
        
        # Метрики
        mse = mean_squared_error(y_test_actual, predictions_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, predictions_actual)
        
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions_actual,
            'actual': y_test_actual
        }
    
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
    
    def plot_predictions(self, data, target_column=None, test_size=0.2, 
                        future_steps=30):
        """Визуализация предсказаний"""
        # Получение исходных данных
        if isinstance(data, pd.DataFrame):
            if target_column:
                original_data = data[target_column].values
                dates = data.index if hasattr(data, 'index') else range(len(data))
            else:
                original_data = data.iloc[:, 0].values
                dates = data.index if hasattr(data, 'index') else range(len(data))
        else:
            original_data = data
            dates = range(len(data))
        
        # Разделение на train/test
        train_size = int(len(original_data) * (1 - test_size))
        
        # Оценка модели
        results = self.evaluate(data, target_column, test_size)
        
        # Предсказание будущего
        future_predictions = self.predict_future(data, target_column, future_steps)
        
        # Создание дат для будущих предсказаний
        if hasattr(data, 'index') and hasattr(data.index, 'freq'):
            future_dates = pd.date_range(start=dates[-1], periods=future_steps+1, freq=data.index.freq)[1:]
        else:
            future_dates = range(len(dates), len(dates) + future_steps)
        
        # Визуализация
        plt.figure(figsize=(15, 8))
        
        # Исходные данные
        plt.plot(dates[:train_size], original_data[:train_size], 
                label='Обучающие данные', color='blue', alpha=0.7)
        plt.plot(dates[train_size:], original_data[train_size:], 
                label='Тестовые данные', color='green', alpha=0.7)
        
        # Предсказания на тестовых данных
        test_dates = dates[train_size + self.sequence_length:]
        plt.plot(test_dates, results['predictions'], 
                label='Предсказания', color='red', alpha=0.7)
        
        # Будущие предсказания
        plt.plot(future_dates, future_predictions, 
                label='Будущие предсказания', color='orange', alpha=0.8, linewidth=2)
        
        plt.title('LSTM Прогнозирование временных рядов')
        plt.xlabel('Время')
        plt.ylabel('Значение')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def load_stock_data(symbol='AAPL', period='2y'):
    """Загрузка данных о ценах акций"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data[['Close']]
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return create_synthetic_data()

def create_synthetic_data(n_points=1000):
    """Создание синтетических временных данных для демонстрации"""
    np.random.seed(42)
    
    # Создание временного ряда с трендом и сезонностью
    time = np.arange(n_points)
    trend = 0.02 * time
    seasonal = 10 * np.sin(2 * np.pi * time / 50) + 5 * np.cos(2 * np.pi * time / 25)
    noise = np.random.normal(0, 2, n_points)
    
    values = 100 + trend + seasonal + noise
    
    # Создание DataFrame
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='D')
    df = pd.DataFrame({'Close': values}, index=dates)
    
    return df


def main():
    """Главная функция для демонстрации работы LSTM предиктора"""
    print("=== LSTM Time Series Prediction Demo ===")
    
    # Загрузка данных (попробуем загрузить реальные данные, если не получится - синтетические)
    print("Загрузка данных...")
    try:
        data = load_stock_data('AAPL', '2y')
        print("Загружены данные по акциям Apple")
    except:
        print("Создание синтетических данных...")
        data = create_synthetic_data()
    
    print(f"Размер данных: {data.shape}")
    print(data.head())
    
    # Визуализация исходных данных
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'])
    plt.title('Исходные данные временного ряда')
    plt.xlabel('Дата')
    plt.ylabel('Цена')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Создание и настройка модели
    predictor = LSTMTimeSeriesPredictor(
        sequence_length=60,
        lstm_units=50,
        dense_units=25,
        dropout_rate=0.2
    )
    
    # Построение модели
    model = predictor.build_model()
    print("\nАрхитектура модели:")
    model.summary()
    
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
    
    # Оценка модели
    print("\nОценка модели:")
    results = predictor.evaluate(data, target_column='Close')
    
    # Визуализация предсказаний
    print("\nВизуализация предсказаний...")
    predictor.plot_predictions(data, target_column='Close', future_steps=30)
    
    # Пример предсказания будущих значений
    print("\nПредсказание будущих значений:")
    future_predictions = predictor.predict_future(data, target_column='Close', steps_ahead=10)
    
    for i, pred in enumerate(future_predictions, 1):
        print(f"День {i}: {pred:.2f}")


if __name__ == "__main__":
    main()