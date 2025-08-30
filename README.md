# 📈 Python OKX Trading Scripts

Набор Python скриптов для анализа и предсказания движений цен на криптовалютной бирже OKX.

## 🧠 Основные компоненты

### LSTM модели для предсказания скачков цен:
- `price_jump_train_OneCFocalL.py` - Основной скрипт тренировки LSTM модели
- `price_jump_eval_colab.py` - Оценка производительности модели
- `price_jump_visualize.py` - Визуализация результатов предсказаний

### Вспомогательные скрипты:
- `lstm_jump_dropout_p_find.py` - Поиск оптимальных параметров dropout
- `price_jump_max_lr_finder_plot.py` - Поиск оптимальной скорости обучения
- `price_jump_train_colab.py` - Версия для Google Colab
- `price_jump_train_colab_FOCAL_LOSS.py` - Версия с Focal Loss
- `price_jump_train_colab_FINDERandOneCycleLR.py` - Версия с OneCycleLR

### Устаревшие версии:
- `OLD_price_jump_train_OneCFocalL.py` - Старая версия основного скрипта

## 🚀 Использование

```bash
# Установка зависимостей
pip install torch pandas numpy matplotlib scikit-learn

# Запуск тренировки модели
python price_jump_train_OneCFocalL.py

# Оценка модели
python price_jump_eval_colab.py

# Визуализация результатов
python price_jump_visualize.py
```

## 🎯 Цель проекта

Создание и обучение LSTM нейронной сети для предсказания значительных движений цен (price jumps) на криптовалютном рынке с использованием данных с биржи OKX.

---

*Проект использует машинное обучение для анализа финансовых данных* 🤖💰