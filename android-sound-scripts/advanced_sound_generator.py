#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Продвинутый генератор звуков для Android с использованием sounddevice
Требует установки: pip install sounddevice numpy scipy
"""

import numpy as np
import time
import math
from typing import Optional, List, Tuple

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️  sounddevice не установлен. Используйте: pip install sounddevice")

class AdvancedSoundGenerator:
    """Продвинутый генератор звуков с использованием sounddevice"""
    
    def __init__(self, sample_rate: int = 44100):
        """
        Инициализация генератора
        
        Args:
            sample_rate: Частота дискретизации
        """
        self.sample_rate = sample_rate
        self.is_available = SOUNDDEVICE_AVAILABLE
        
        if self.is_available:
            # Получаем информацию об аудиоустройствах
            try:
                devices = sd.query_devices()
                print("🔊 Доступные аудиоустройства:")
                for i, device in enumerate(devices):
                    if device['max_output_channels'] > 0:
                        print(f"  {i}: {device['name']} (выходов: {device['max_output_channels']})")
                
                # Устанавливаем устройство по умолчанию
                sd.default.samplerate = sample_rate
                print(f"✅ Sounddevice готов: {sample_rate}Hz")
            except Exception as e:
                print(f"❌ Ошибка инициализации sounddevice: {e}")
                self.is_available = False
    
    def generate_wave(self, frequency: float, duration: float, 
                     wave_type: str = "sine", amplitude: float = 0.5,
                     fade_in: float = 0.0, fade_out: float = 0.0) -> np.ndarray:
        """
        Генерация волны с дополнительными эффектами
        
        Args:
            frequency: Частота в Hz
            duration: Длительность в секундах
            wave_type: Тип волны
            amplitude: Амплитуда
            fade_in: Длительность нарастания громкости
            fade_out: Длительность затухания
        """
        frames = int(duration * self.sample_rate)
        t = np.linspace(0, duration, frames, False)
        
        # Генерация основной волны
        if wave_type == "sine":
            wave = np.sin(2 * np.pi * frequency * t)
        elif wave_type == "square":
            wave = np.sign(np.sin(2 * np.pi * frequency * t))
        elif wave_type == "sawtooth":
            wave = 2 * (t * frequency - np.floor(0.5 + t * frequency))
        elif wave_type == "triangle":
            wave = 2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1
        elif wave_type == "noise":
            wave = np.random.uniform(-1, 1, frames)
        else:
            raise ValueError(f"Неподдерживаемый тип волны: {wave_type}")
        
        # Применяем амплитуду
        wave *= amplitude
        
        # Применяем fade-in
        if fade_in > 0:
            fade_in_frames = int(fade_in * self.sample_rate)
            fade_in_curve = np.linspace(0, 1, fade_in_frames)
            wave[:fade_in_frames] *= fade_in_curve
        
        # Применяем fade-out
        if fade_out > 0:
            fade_out_frames = int(fade_out * self.sample_rate)
            fade_out_curve = np.linspace(1, 0, fade_out_frames)
            wave[-fade_out_frames:] *= fade_out_curve
        
        return wave
    
    def play_wave(self, wave_data: np.ndarray, blocking: bool = True):
        """
        Воспроизведение волны
        
        Args:
            wave_data: Аудиоданные
            blocking: Ждать окончания воспроизведения
        """
        if not self.is_available:
            print("❌ sounddevice недоступен")
            return
        
        try:
            sd.play(wave_data, self.sample_rate)
            if blocking:
                sd.wait()  # Ждем окончания воспроизведения
        except Exception as e:
            print(f"❌ Ошибка воспроизведения: {e}")
    
    def play_tone(self, frequency: float, duration: float, 
                  wave_type: str = "sine", amplitude: float = 0.5):
        """Воспроизведение простого тона"""
        wave = self.generate_wave(frequency, duration, wave_type, amplitude)
        print(f"🔊 Воспроизведение: {frequency}Hz, {duration}с, {wave_type}")
        self.play_wave(wave)
    
    def play_chord(self, frequencies: List[float], duration: float, 
                   amplitude: float = 0.3):
        """
        Воспроизведение аккорда (несколько частот одновременно)
        
        Args:
            frequencies: Список частот
            duration: Длительность
            amplitude: Амплитуда для каждой частоты
        """
        if not frequencies:
            return
        
        # Генерируем волну для каждой частоты
        combined_wave = np.zeros(int(duration * self.sample_rate))
        
        for freq in frequencies:
            wave = self.generate_wave(freq, duration, "sine", amplitude)
            combined_wave += wave
        
        # Нормализуем амплитуду
        max_amplitude = np.max(np.abs(combined_wave))
        if max_amplitude > 0:
            combined_wave /= max_amplitude
            combined_wave *= 0.8  # Оставляем запас
        
        print(f"🎼 Воспроизведение аккорда: {frequencies} Hz")
        self.play_wave(combined_wave)
    
    def play_frequency_sweep(self, start_freq: float, end_freq: float, 
                           duration: float, amplitude: float = 0.5):
        """
        Воспроизведение скользящего по частоте тона
        
        Args:
            start_freq: Начальная частота
            end_freq: Конечная частота
            duration: Длительность
            amplitude: Амплитуда
        """
        frames = int(duration * self.sample_rate)
        t = np.linspace(0, duration, frames, False)
        
        # Линейное изменение частоты
        frequencies = np.linspace(start_freq, end_freq, frames)
        
        # Генерируем волну с изменяющейся частотой
        phase = 2 * np.pi * np.cumsum(frequencies) / self.sample_rate
        wave = amplitude * np.sin(phase)
        
        print(f"🌊 Скользящий тон: {start_freq} → {end_freq} Hz")
        self.play_wave(wave)
    
    def play_dtmf_tone(self, digit: str, duration: float = 0.5):
        """
        Воспроизведение DTMF тона (тональный набор)
        
        Args:
            digit: Цифра или символ (0-9, *, #, A-D)
            duration: Длительность
        """
        # Частоты DTMF
        dtmf_freqs = {
            '1': (697, 1209), '2': (697, 1336), '3': (697, 1477), 'A': (697, 1633),
            '4': (770, 1209), '5': (770, 1336), '6': (770, 1477), 'B': (770, 1633),
            '7': (852, 1209), '8': (852, 1336), '9': (852, 1477), 'C': (852, 1633),
            '*': (941, 1209), '0': (941, 1336), '#': (941, 1477), 'D': (941, 1633)
        }
        
        if digit not in dtmf_freqs:
            print(f"❌ Неподдерживаемый DTMF символ: {digit}")
            return
        
        freq1, freq2 = dtmf_freqs[digit]
        
        # Генерируем две синусоиды и складываем
        wave1 = self.generate_wave(freq1, duration, "sine", 0.5)
        wave2 = self.generate_wave(freq2, duration, "sine", 0.5)
        combined_wave = (wave1 + wave2) / 2
        
        print(f"📞 DTMF тон для '{digit}': {freq1}Hz + {freq2}Hz")
        self.play_wave(combined_wave)
    
    def play_phone_number(self, number: str, digit_duration: float = 0.5, 
                         pause_duration: float = 0.1):
        """
        Воспроизведение телефонного номера в виде DTMF тонов
        
        Args:
            number: Номер телефона (строка)
            digit_duration: Длительность каждой цифры
            pause_duration: Пауза между цифрами
        """
        print(f"📱 Набор номера: {number}")
        
        for char in number:
            if char.isdigit() or char in '*#':
                self.play_dtmf_tone(char, digit_duration)
                time.sleep(pause_duration)
            elif char == ' ' or char == '-':
                time.sleep(pause_duration * 2)  # Длинная пауза
    
    def stop_all(self):
        """Остановка всех звуков"""
        if self.is_available:
            sd.stop()
            print("⏹️ Все звуки остановлены")


def demo_advanced_sounds():
    """Демонстрация продвинутых возможностей"""
    print("🎼 Демонстрация продвинутого генератора звуков")
    print("=" * 50)
    
    generator = AdvancedSoundGenerator()
    
    if not generator.is_available:
        print("❌ sounddevice недоступен. Установите: pip install sounddevice")
        return
    
    try:
        while True:
            print("\n🎵 Выберите демонстрацию:")
            print("1. Простой тон с затуханием")
            print("2. Аккорд (до-мажор)")
            print("3. Скользящий тон")
            print("4. DTMF тоны (0-9)")
            print("5. Набор телефонного номера")
            print("6. Белый шум")
            print("7. Разные типы волн")
            print("0. Выход")
            
            choice = input("\n👆 Ваш выбор: ").strip()
            
            if choice == "1":
                print("🔊 Тон с затуханием...")
                wave = generator.generate_wave(440, 3.0, "sine", 0.7, 
                                             fade_in=0.5, fade_out=1.0)
                generator.play_wave(wave)
                
            elif choice == "2":
                print("🎼 До-мажорный аккорд...")
                # C-E-G аккорд
                generator.play_chord([261.63, 329.63, 392.00], 2.0)
                
            elif choice == "3":
                print("🌊 Скользящий тон...")
                generator.play_frequency_sweep(200, 2000, 3.0)
                
            elif choice == "4":
                print("📞 DTMF тоны...")
                for digit in "0123456789":
                    generator.play_dtmf_tone(digit, 0.3)
                    time.sleep(0.1)
                    
            elif choice == "5":
                number = input("Введите номер телефона: ").strip()
                generator.play_phone_number(number)
                
            elif choice == "6":
                print("📻 Белый шум...")
                generator.play_tone(0, 2.0, "noise", 0.3)
                
            elif choice == "7":
                print("🌊 Разные типы волн...")
                wave_types = ["sine", "square", "sawtooth", "triangle"]
                for wave_type in wave_types:
                    print(f"  {wave_type.capitalize()} волна")
                    generator.play_tone(440, 1.0, wave_type, 0.4)
                    time.sleep(0.5)
                    
            elif choice == "0":
                break
                
            else:
                print("❌ Неверный выбор")
    
    except KeyboardInterrupt:
        print("\n⏹️ Остановлено пользователем")
    
    finally:
        generator.stop_all()
        print("👋 До свидания!")


if __name__ == "__main__":
    demo_advanced_sounds()