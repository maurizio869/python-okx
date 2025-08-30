#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для воспроизведения звуков на Android телефоне через Pydroid
Поддерживает различные способы генерации и воспроизведения звуков
"""

import pygame
import numpy as np
import time
import math
import os
from typing import Optional

class AndroidSoundPlayer:
    """Класс для работы со звуком на Android устройствах"""
    
    def __init__(self, sample_rate: int = 44100, channels: int = 1):
        """
        Инициализация звукового плеера
        
        Args:
            sample_rate: Частота дискретизации (Hz)
            channels: Количество каналов (1 - моно, 2 - стерео)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_initialized = False
        
        try:
            pygame.mixer.pre_init(
                frequency=sample_rate,
                size=-16,
                channels=channels,
                buffer=512
            )
            pygame.mixer.init()
            self.is_initialized = True
            print(f"✅ Pygame mixer инициализирован: {sample_rate}Hz, {channels} каналов")
        except Exception as e:
            print(f"❌ Ошибка инициализации pygame: {e}")
    
    def generate_tone(self, frequency: float, duration: float, 
                     wave_type: str = "sine", amplitude: float = 0.5) -> np.ndarray:
        """
        Генерация тонального сигнала
        
        Args:
            frequency: Частота в Hz
            duration: Длительность в секундах
            wave_type: Тип волны ("sine", "square", "sawtooth", "triangle")
            amplitude: Амплитуда (0.0 - 1.0)
        
        Returns:
            Массив numpy с аудиоданными
        """
        frames = int(duration * self.sample_rate)
        t = np.linspace(0, duration, frames, False)
        
        if wave_type == "sine":
            wave = np.sin(2 * np.pi * frequency * t)
        elif wave_type == "square":
            wave = np.sign(np.sin(2 * np.pi * frequency * t))
        elif wave_type == "sawtooth":
            wave = 2 * (t * frequency - np.floor(0.5 + t * frequency))
        elif wave_type == "triangle":
            wave = 2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1
        else:
            raise ValueError(f"Неподдерживаемый тип волны: {wave_type}")
        
        # Применяем амплитуду и конвертируем в 16-битный формат
        wave = (wave * amplitude * 32767).astype(np.int16)
        
        # Для стерео дублируем канал
        if self.channels == 2:
            wave = np.column_stack((wave, wave))
        
        return wave
    
    def play_tone(self, frequency: float, duration: float, 
                  wave_type: str = "sine", amplitude: float = 0.5):
        """
        Воспроизведение тонального сигнала
        
        Args:
            frequency: Частота в Hz
            duration: Длительность в секундах
            wave_type: Тип волны
            amplitude: Амплитуда
        """
        if not self.is_initialized:
            print("❌ Плеер не инициализирован")
            return
        
        try:
            wave_data = self.generate_tone(frequency, duration, wave_type, amplitude)
            sound = pygame.sndarray.make_sound(wave_data)
            sound.play()
            
            print(f"🔊 Воспроизводится: {frequency}Hz, {duration}с, {wave_type}")
            time.sleep(duration)  # Ждем окончания воспроизведения
            
        except Exception as e:
            print(f"❌ Ошибка воспроизведения: {e}")
    
    def play_melody(self, notes: list, note_duration: float = 0.5):
        """
        Воспроизведение мелодии
        
        Args:
            notes: Список частот нот
            note_duration: Длительность каждой ноты
        """
        print("🎵 Воспроизведение мелодии...")
        for i, freq in enumerate(notes):
            if freq == 0:  # Пауза
                time.sleep(note_duration)
            else:
                self.play_tone(freq, note_duration)
            print(f"Нота {i+1}/{len(notes)}")
    
    def play_beep_sequence(self, count: int = 3, frequency: float = 800, 
                          beep_duration: float = 0.2, pause_duration: float = 0.3):
        """
        Воспроизведение последовательности коротких сигналов
        
        Args:
            count: Количество сигналов
            frequency: Частота сигнала
            beep_duration: Длительность сигнала
            pause_duration: Длительность паузы между сигналами
        """
        print(f"📢 Воспроизведение {count} сигналов...")
        for i in range(count):
            self.play_tone(frequency, beep_duration)
            if i < count - 1:  # Пауза между сигналами (кроме последнего)
                time.sleep(pause_duration)
    
    def play_alarm(self, duration: float = 5.0):
        """
        Воспроизведение сигнала тревоги (чередование двух частот)
        
        Args:
            duration: Общая длительность сигнала
        """
        print("🚨 Воспроизведение сигнала тревоги...")
        end_time = time.time() + duration
        
        while time.time() < end_time:
            self.play_tone(800, 0.3)
            if time.time() >= end_time:
                break
            self.play_tone(600, 0.3)
    
    def cleanup(self):
        """Очистка ресурсов"""
        if self.is_initialized:
            pygame.mixer.quit()
            print("🔇 Pygame mixer остановлен")


def main():
    """Основная функция с примерами использования"""
    print("🎼 Android Sound Player - Тест звуков")
    print("=" * 40)
    
    # Создаем плеер
    player = AndroidSoundPlayer()
    
    if not player.is_initialized:
        print("❌ Не удалось инициализировать звуковую систему")
        return
    
    try:
        # Меню выбора
        while True:
            print("\n📱 Выберите действие:")
            print("1. Простой тон (440Hz)")
            print("2. Последовательность сигналов")
            print("3. Сигнал тревоги")
            print("4. Мелодия 'Twinkle Twinkle Little Star'")
            print("5. Тест разных типов волн")
            print("6. Пользовательский тон")
            print("0. Выход")
            
            choice = input("\n👆 Ваш выбор: ").strip()
            
            if choice == "1":
                print("\n🎵 Воспроизведение простого тона...")
                player.play_tone(440, 2.0)
                
            elif choice == "2":
                print("\n📢 Последовательность сигналов...")
                player.play_beep_sequence(5, 1000, 0.2, 0.3)
                
            elif choice == "3":
                print("\n🚨 Сигнал тревоги...")
                player.play_alarm(3.0)
                
            elif choice == "4":
                # Ноты для "Twinkle Twinkle Little Star"
                notes = [
                    262,  # C
                    262,  # C
                    392,  # G
                    392,  # G
                    440,  # A
                    440,  # A
                    392,  # G
                    0,    # пауза
                    349,  # F
                    349,  # F
                    330,  # E
                    330,  # E
                    294,  # D
                    294,  # D
                    262   # C
                ]
                player.play_melody(notes, 0.4)
                
            elif choice == "5":
                print("\n🌊 Тест разных типов волн...")
                wave_types = ["sine", "square", "sawtooth", "triangle"]
                for wave_type in wave_types:
                    print(f"Тип волны: {wave_type}")
                    player.play_tone(440, 1.0, wave_type, 0.3)
                    time.sleep(0.5)
                    
            elif choice == "6":
                try:
                    freq = float(input("Введите частоту (Hz): "))
                    duration = float(input("Введите длительность (сек): "))
                    player.play_tone(freq, duration)
                except ValueError:
                    print("❌ Неверный формат числа")
                    
            elif choice == "0":
                break
                
            else:
                print("❌ Неверный выбор")
    
    except KeyboardInterrupt:
        print("\n⏹️ Остановлено пользователем")
    
    finally:
        player.cleanup()
        print("👋 До свидания!")


if __name__ == "__main__":
    main()