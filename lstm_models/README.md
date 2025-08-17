# LSTM Models для временных рядов и классификации текста

Этот проект содержит примеры реализации LSTM (Long Short-Term Memory) нейронных сетей для различных задач машинного обучения с использованием TensorFlow/Keras.

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Запуск примеров

```bash
# Классификация текста
python lstm_text_classification.py

# Прогнозирование временных рядов (одномерные)
python lstm_time_series.py

# Прогнозирование временных рядов (многомерные)
python lstm_multivariate.py
```

## 📁 Структура проекта

```
lstm_models/
├── lstm_text_classification.py    # LSTM для классификации текста
├── lstm_time_series.py           # LSTM для одномерных временных рядов
├── lstm_multivariate.py          # LSTM для многомерных временных рядов
├── requirements.txt              # Зависимости проекта
└── README.md                     # Документация
```

## 🔤 LSTM для классификации текста

### Описание
Модель `LSTMTextClassifier` предназначена для анализа настроений текста или любой другой задачи классификации текста.

### Основные возможности:
- Bidirectional LSTM архитектура
- Автоматическая токенизация текста
- Поддержка многоклассовой классификации
- Визуализация процесса обучения
- Метрики качества и confusion matrix

### Пример использования:

```python
from lstm_text_classification import LSTMTextClassifier

# Создание классификатора
classifier = LSTMTextClassifier(
    vocab_size=10000,
    embedding_dim=128,
    lstm_units=64,
    max_length=100,
    num_classes=2
)

# Построение модели
model = classifier.build_model()

# Обучение
history = classifier.train(train_texts, train_labels, epochs=50)

# Предсказание
predictions = classifier.predict(["Отличный фильм!", "Ужасно скучно"])
```

### Параметры модели:
- `vocab_size`: Размер словаря (по умолчанию 10000)
- `embedding_dim`: Размерность embedding векторов (по умолчанию 128)
- `lstm_units`: Количество LSTM юнитов (по умолчанию 64)
- `max_length`: Максимальная длина последовательности (по умолчанию 100)
- `num_classes`: Количество классов для классификации (по умолчанию 2)

## 📈 LSTM для временных рядов (одномерные)

### Описание
Модель `LSTMTimeSeriesPredictor` предназначена для прогнозирования одномерных временных рядов, таких как цены акций, температура, продажи и т.д.

### Основные возможности:
- Многослойная LSTM архитектура
- Автоматическая нормализация данных
- Предсказание на несколько шагов вперед
- Загрузка реальных данных через Yahoo Finance
- Создание синтетических данных для тестирования
- Визуализация результатов

### Пример использования:

```python
from lstm_time_series import LSTMTimeSeriesPredictor
import yfinance as yf

# Загрузка данных
data = yf.Ticker('AAPL').history(period='2y')

# Создание предиктора
predictor = LSTMTimeSeriesPredictor(
    sequence_length=60,
    lstm_units=50,
    dense_units=25
)

# Построение и обучение модели
model = predictor.build_model()
history, test_data = predictor.train(data, target_column='Close')

# Предсказание будущих значений
future_predictions = predictor.predict_future(data, steps_ahead=30)

# Визуализация результатов
predictor.plot_predictions(data, target_column='Close')
```

### Параметры модели:
- `sequence_length`: Длина входной последовательности (по умолчанию 60)
- `n_features`: Количество признаков (по умолчанию 1)
- `lstm_units`: Количество LSTM юнитов (по умолчанию 50)
- `dense_units`: Количество нейронов в dense слоях (по умолчанию 25)
- `dropout_rate`: Коэффициент dropout (по умолчанию 0.2)

## 📊 LSTM для многомерных временных рядов

### Описание
Модель `MultivariateLSTMPredictor` работает с многомерными временными рядами, используя несколько признаков одновременно для более точного прогнозирования.

### Основные возможности:
- Автоматическое создание технических индикаторов
- Анализ важности признаков
- Многомерная LSTM архитектура
- Корреляционный анализ
- Расширенная визуализация

### Технические индикаторы:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- MACD
- Bollinger Bands
- Волатильность
- Соотношения цен и объемов

### Пример использования:

```python
from lstm_multivariate import MultivariateLSTMPredictor
import yfinance as yf

# Загрузка расширенных данных
data = yf.Ticker('AAPL').history(period='2y')

# Создание многомерного предиктора
predictor = MultivariateLSTMPredictor(
    sequence_length=60,
    lstm_units=64,
    dense_units=32
)

# Анализ важности признаков
correlations = predictor.plot_feature_importance(data)

# Обучение модели
history, test_data = predictor.train(data, target_column='Close')

# Оценка и визуализация
results = predictor.plot_predictions(data)
```

### Используемые признаки:
1. **Цены**: Close, Open, High, Low
2. **Объем**: Volume, Volume SMA
3. **Доходность**: Returns
4. **Технические индикаторы**: RSI, MACD, SMA, EMA
5. **Соотношения**: Close/SMA, High/Low, Close/Open
6. **Волатильность**: Rolling standard deviation

## 🛠 Настройка и оптимизация

### Гиперпараметры для тюнинга:

1. **Архитектура сети**:
   - Количество LSTM слоев
   - Количество юнитов в каждом слое
   - Dropout rate для регуляризации

2. **Обучение**:
   - Learning rate
   - Batch size
   - Количество эпох
   - Early stopping patience

3. **Данные**:
   - Длина входной последовательности
   - Набор признаков
   - Методы нормализации

### Советы по улучшению качества:

1. **Больше данных**: LSTM модели хорошо работают с большими объемами данных
2. **Feature engineering**: Добавление релевантных признаков может значительно улучшить качество
3. **Регуляризация**: Используйте dropout и early stopping для предотвращения переобучения
4. **Валидация**: Всегда используйте отдельную валидационную выборку
5. **Нормализация**: Обязательно нормализуйте входные данные

## 📊 Метрики качества

### Для классификации текста:
- **Accuracy**: Доля правильно классифицированных примеров
- **Precision**: Точность по каждому классу
- **Recall**: Полнота по каждому классу
- **F1-score**: Гармоническое среднее точности и полноты
- **Confusion Matrix**: Матрица ошибок

### Для временных рядов:
- **MSE (Mean Squared Error)**: Среднеквадратичная ошибка
- **RMSE (Root Mean Squared Error)**: Корень из среднеквадратичной ошибки
- **MAE (Mean Absolute Error)**: Средняя абсолютная ошибка
- **MAPE**: Средняя абсолютная процентная ошибка (можно добавить)

## 🔧 Расширения и модификации

### Возможные улучшения:

1. **Attention механизм**: Добавление attention слоев для лучшего понимания важности различных временных моментов

2. **Transformer архитектура**: Замена LSTM на Transformer для некоторых задач

3. **Ensemble методы**: Комбинирование нескольких LSTM моделей

4. **Hyperparameter tuning**: Автоматический поиск оптимальных гиперпараметров

5. **Cross-validation**: Кросс-валидация для временных рядов

## 🐛 Устранение проблем

### Частые проблемы и решения:

1. **Модель не обучается**:
   - Проверьте learning rate (попробуйте 0.001 или 0.0001)
   - Убедитесь в правильной нормализации данных
   - Проверьте архитектуру модели

2. **Переобучение**:
   - Увеличьте dropout rate
   - Добавьте регуляризацию
   - Уменьшите количество параметров модели
   - Используйте early stopping

3. **Медленное обучение**:
   - Уменьшите batch_size
   - Используйте GPU если доступно
   - Оптимизируйте загрузку данных

4. **Плохое качество предсказаний**:
   - Добавьте больше данных
   - Попробуйте feature engineering
   - Настройте гиперпараметры
   - Проверьте качество исходных данных

## 📚 Дополнительные ресурсы

### Полезные ссылки:
- [TensorFlow LSTM Guide](https://www.tensorflow.org/guide/keras/rnn)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Time Series Forecasting with LSTM](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/)

### Библиотеки:
- **TensorFlow/Keras**: Основной фреймворк для глубокого обучения
- **yfinance**: Загрузка финансовых данных
- **ta**: Технические индикаторы
- **scikit-learn**: Метрики и предобработка данных
- **matplotlib/seaborn**: Визуализация

## 🤝 Вклад в проект

Если вы хотите улучшить проект:

1. Создайте fork репозитория
2. Создайте новую ветку для ваших изменений
3. Внесите изменения и добавьте тесты
4. Создайте pull request с описанием изменений

## 📄 Лицензия

Проект распространяется под лицензией MIT. Вы можете свободно использовать, изменять и распространять код.

---

**Удачи в изучении LSTM моделей! 🚀**