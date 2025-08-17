"""
LSTM модель для классификации текста
Пример использования LSTM для анализа настроений или классификации отзывов
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Установим random seed для воспроизводимости
np.random.seed(42)
tf.random.set_seed(42)

class LSTMTextClassifier:
    """
    LSTM классификатор для текста
    """
    
    def __init__(self, vocab_size=10000, embedding_dim=128, lstm_units=64, 
                 max_length=100, num_classes=2):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.max_length = max_length
        self.num_classes = num_classes
        self.tokenizer = None
        self.model = None
        
    def build_model(self):
        """Создание архитектуры LSTM модели"""
        model = Sequential([
            # Embedding слой для преобразования слов в векторы
            Embedding(input_dim=self.vocab_size, 
                     output_dim=self.embedding_dim, 
                     input_length=self.max_length),
            
            # Dropout для регуляризации
            Dropout(0.2),
            
            # Bidirectional LSTM слой
            Bidirectional(LSTM(self.lstm_units, return_sequences=True)),
            Dropout(0.3),
            
            # Второй LSTM слой
            LSTM(self.lstm_units // 2),
            Dropout(0.3),
            
            # Dense слои для классификации
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax' if self.num_classes > 2 else 'sigmoid')
        ])
        
        # Компиляция модели
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, texts, labels):
        """Подготовка текстовых данных для обучения"""
        # Токенизация текста
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(texts)
        
        # Преобразование текста в последовательности
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Padding последовательностей
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, 
                                       padding='post', truncating='post')
        
        return padded_sequences, np.array(labels)
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, 
              epochs=50, batch_size=32, verbose=1):
        """Обучение модели"""
        # Подготовка данных
        X_train, y_train = self.prepare_data(train_texts, train_labels)
        
        validation_data = None
        if val_texts is not None and val_labels is not None:
            X_val, y_val = self.prepare_data(val_texts, val_labels)
            validation_data = (X_val, y_val)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7)
        ]
        
        # Обучение
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, texts):
        """Предсказание для новых текстов"""
        X, _ = self.prepare_data(texts, [0] * len(texts))  # Dummy labels
        predictions = self.model.predict(X)
        return predictions
    
    def evaluate(self, test_texts, test_labels):
        """Оценка модели на тестовых данных"""
        X_test, y_test = self.prepare_data(test_texts, test_labels)
        
        # Предсказания
        predictions = self.model.predict(X_test)
        if self.num_classes == 2:
            y_pred = (predictions > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(predictions, axis=1)
        
        # Метрики
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return y_pred
    
    def plot_training_history(self, history):
        """Визуализация процесса обучения"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()


def create_sample_data():
    """Создание примерных данных для демонстрации"""
    # Примерные тексты и метки (положительные и отрицательные отзывы)
    texts = [
        "Этот фильм просто потрясающий! Отличная игра актеров.",
        "Ужасный фильм, потратил время зря.",
        "Очень интересный сюжет, рекомендую к просмотру.",
        "Скучно и предсказуемо, не советую.",
        "Великолепная работа режиссера, шедевр кинематографа.",
        "Полная ерунда, даже не знаю как это сняли.",
        "Захватывающий триллер с неожиданной развязкой.",
        "Банальный сюжет, ничего нового.",
        "Отличная комедия, смеялся от души.",
        "Депрессивный фильм, настроение испортил.",
        "Прекрасная игра актеров, очень эмоционально.",
        "Слабый сценарий, актеры играют плохо.",
        "Визуальные эффекты на высоте, зрелищно.",
        "Дешевые спецэффекты, выглядит нереалистично.",
        "Глубокий философский фильм, заставляет думать.",
        "Непонятный и запутанный сюжет.",
        "Отличный саундтрек, музыка дополняет картину.",
        "Ужасная музыка, режет слух.",
        "Красивые пейзажи и операторская работа.",
        "Плохое качество съемки, картинка мутная."
    ]
    
    # Метки: 1 - положительный отзыв, 0 - отрицательный
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    return texts, labels


def main():
    """Главная функция для демонстрации работы LSTM классификатора"""
    print("=== LSTM Text Classification Demo ===")
    
    # Создание примерных данных
    texts, labels = create_sample_data()
    
    # Разделение на train/test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Создание и настройка модели
    classifier = LSTMTextClassifier(
        vocab_size=1000,
        embedding_dim=64,
        lstm_units=32,
        max_length=50,
        num_classes=2
    )
    
    # Построение модели
    model = classifier.build_model()
    print("\nАрхитектура модели:")
    model.summary()
    
    # Обучение
    print("\nНачинаем обучение...")
    history = classifier.train(
        train_texts, train_labels,
        epochs=20,
        batch_size=4,
        verbose=1
    )
    
    # Визуализация процесса обучения
    classifier.plot_training_history(history)
    
    # Оценка на тестовых данных
    print("\nОценка модели на тестовых данных:")
    classifier.evaluate(test_texts, test_labels)
    
    # Пример предсказания
    print("\nПример предсказаний:")
    sample_texts = [
        "Отличный фильм, всем рекомендую!",
        "Ужасно скучно, не советую смотреть."
    ]
    
    predictions = classifier.predict(sample_texts)
    for text, pred in zip(sample_texts, predictions):
        sentiment = "Положительный" if pred[0] > 0.5 else "Отрицательный"
        confidence = pred[0] if pred[0] > 0.5 else 1 - pred[0]
        print(f"Текст: '{text}'")
        print(f"Настроение: {sentiment} (уверенность: {confidence:.2f})")
        print()


if __name__ == "__main__":
    main()