# 🐍 Python OKX Trading & Android Sound Scripts

Репозиторий содержит различные Python скрипты для работы с криптовалютами и звуком на Android.

## 📁 Структура проекта

### 📈 Торговые скрипты (OKX Trading)
Скрипты для анализа и предсказания движений цен на криптовалютной бирже OKX:

- `price_jump_train_OneCFocalL.py` - Тренировка LSTM модели для предсказания скачков цен
- `price_jump_eval_colab.py` - Оценка производительности модели
- `price_jump_visualize.py` - Визуализация результатов
- `lstm_jump_dropout_p_find.py` - Поиск оптимальных параметров dropout
- Другие вспомогательные скрипты

### 🔊 Android Sound Scripts
**📂 Папка: [`android-sound-scripts/`](./android-sound-scripts/)**

Набор Python скриптов для воспроизведения звуков на Android устройствах через IDE типа Pydroid:

- `quick_start.py` - Быстрый тест звука
- `android_sound_player.py` - Полнофункциональный плеер
- `advanced_sound_generator.py` - Продвинутый генератор звуков
- `audio_file_player.py` - Работа с аудиофайлами
- `requirements.txt` - Зависимости
- `README.md` - Подробная документация

## 🚀 Быстрый старт

### Для звуковых скриптов на Android:
```bash
cd android-sound-scripts/
pip install pygame numpy
python quick_start.py
```

### Для торговых скриптов:
```bash
# Установите зависимости для машинного обучения
pip install torch pandas numpy matplotlib
python price_jump_train_OneCFocalL.py
```

## 📖 Документация

- **Звуковые скрипты:** См. [`android-sound-scripts/README.md`](./android-sound-scripts/README.md)
- **Торговые скрипты:** Документация в комментариях к файлам

## 🤝 Вклад

Репозиторий содержит независимые проекты. Каждая папка имеет свои зависимости и документацию.

---

*Создано с помощью Python для различных задач автоматизации и анализа* 🐍