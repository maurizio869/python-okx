#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрый старт - простой скрипт для воспроизведения звуков на Android
Минимальные зависимости: только pygame и numpy
"""

import pygame
import numpy as np
import time
import sys

def init_sound():
    """Инициализация звуковой системы"""
    try:
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=512)
        pygame.mixer.init()
        print("✅ Звуковая система готова!")
        return True
    except Exception as e:
        print(f"❌ Ошибка инициализации звука: {e}")
        return False

def play_beep(frequency=800, duration=0.5, volume=0.5):
    """
    Простой сигнал
    
    Args:
        frequency: Частота в Hz
        duration: Длительность в секундах
        volume: Громкость (0.0-1.0)
    """
    sample_rate = 22050
    frames = int(duration * sample_rate)
    
    # Генерируем синусоиду
    t = np.linspace(0, duration, frames, False)
    wave = np.sin(2 * np.pi * frequency * t)
    
    # Конвертируем в 16-битный формат
    wave = (wave * volume * 32767).astype(np.int16)
    
    # Воспроизводим
    sound = pygame.sndarray.make_sound(wave)
    sound.play()
    time.sleep(duration)

def main():
    """Главная функция"""
    print("🔊 Быстрый тест звука на Android")
    print("=" * 35)
    
    if not init_sound():
        print("❌ Не удалось инициализировать звук")
        sys.exit(1)
    
    try:
        print("\n🎵 Тестируем звуки...")
        
        # Простой сигнал
        print("1. Простой сигнал (800 Hz)")
        play_beep(800, 1.0)
        
        # Низкий тон
        print("2. Низкий тон (200 Hz)")
        play_beep(200, 1.0)
        
        # Высокий тон
        print("3. Высокий тон (1500 Hz)")
        play_beep(1500, 1.0)
        
        # Последовательность сигналов
        print("4. Последовательность сигналов")
        for i in range(3):
            play_beep(1000, 0.2, 0.7)
            time.sleep(0.1)
        
        # Мелодия
        print("5. Простая мелодия")
        notes = [262, 294, 330, 349, 392, 440, 494, 523]  # C мажорная гамма
        for note in notes:
            play_beep(note, 0.3, 0.5)
        
        print("\n✅ Тест завершен успешно!")
        print("🎉 Звук работает на вашем Android устройстве!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Остановлено пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
    finally:
        pygame.mixer.quit()
        print("👋 До свидания!")

if __name__ == "__main__":
    main()