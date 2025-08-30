#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для работы с аудиофайлами на Android с использованием pydub
Требует установки: pip install pydub pygame
Для поддержки MP3: pip install pydub[mp3]
"""

import os
import time
from typing import Optional, List

try:
    from pydub import AudioSegment
    from pydub.generators import Sine, Square, Sawtooth, Triangle, WhiteNoise
    from pydub.playback import play
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠️  pydub не установлен. Используйте: pip install pydub")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  pygame не установлен. Используйте: pip install pygame")

class AudioFilePlayer:
    """Класс для работы с аудиофайлами и их воспроизведения"""
    
    def __init__(self):
        """Инициализация плеера"""
        self.pydub_available = PYDUB_AVAILABLE
        self.pygame_available = PYGAME_AVAILABLE
        
        if self.pygame_available:
            try:
                pygame.mixer.init()
                print("✅ Pygame mixer инициализирован")
            except Exception as e:
                print(f"❌ Ошибка инициализации pygame: {e}")
                self.pygame_available = False
        
        if self.pydub_available:
            print("✅ Pydub доступен")
        
        self.current_audio = None
    
    def generate_tone_file(self, frequency: float, duration_ms: int, 
                          wave_type: str = "sine", volume: float = 0.5) -> Optional[AudioSegment]:
        """
        Генерация тонального сигнала как AudioSegment
        
        Args:
            frequency: Частота в Hz
            duration_ms: Длительность в миллисекундах
            wave_type: Тип волны ("sine", "square", "sawtooth", "triangle", "noise")
            volume: Громкость (0.0 - 1.0)
        
        Returns:
            AudioSegment или None при ошибке
        """
        if not self.pydub_available:
            print("❌ pydub недоступен")
            return None
        
        try:
            if wave_type == "sine":
                audio = Sine(frequency).to_audio_segment(duration=duration_ms)
            elif wave_type == "square":
                audio = Square(frequency).to_audio_segment(duration=duration_ms)
            elif wave_type == "sawtooth":
                audio = Sawtooth(frequency).to_audio_segment(duration=duration_ms)
            elif wave_type == "triangle":
                audio = Triangle(frequency).to_audio_segment(duration=duration_ms)
            elif wave_type == "noise":
                audio = WhiteNoise().to_audio_segment(duration=duration_ms)
            else:
                raise ValueError(f"Неподдерживаемый тип волны: {wave_type}")
            
            # Применяем громкость (в dB)
            volume_db = 20 * math.log10(volume) if volume > 0 else -60
            audio = audio + volume_db
            
            return audio
            
        except Exception as e:
            print(f"❌ Ошибка генерации звука: {e}")
            return None
    
    def play_audio_segment(self, audio: AudioSegment):
        """
        Воспроизведение AudioSegment
        
        Args:
            audio: AudioSegment для воспроизведения
        """
        if not self.pydub_available:
            print("❌ pydub недоступен")
            return
        
        try:
            play(audio)
        except Exception as e:
            print(f"❌ Ошибка воспроизведения: {e}")
    
    def load_audio_file(self, file_path: str) -> Optional[AudioSegment]:
        """
        Загрузка аудиофайла
        
        Args:
            file_path: Путь к файлу
        
        Returns:
            AudioSegment или None при ошибке
        """
        if not self.pydub_available:
            print("❌ pydub недоступен")
            return None
        
        if not os.path.exists(file_path):
            print(f"❌ Файл не найден: {file_path}")
            return None
        
        try:
            # Автоматическое определение формата по расширению
            audio = AudioSegment.from_file(file_path)
            print(f"✅ Загружен файл: {file_path}")
            print(f"   Длительность: {len(audio)/1000:.2f} сек")
            print(f"   Частота: {audio.frame_rate} Hz")
            print(f"   Каналы: {audio.channels}")
            return audio
            
        except Exception as e:
            print(f"❌ Ошибка загрузки файла: {e}")
            return None
    
    def save_audio_segment(self, audio: AudioSegment, file_path: str, 
                          format: str = "wav"):
        """
        Сохранение AudioSegment в файл
        
        Args:
            audio: AudioSegment для сохранения
            file_path: Путь для сохранения
            format: Формат файла ("wav", "mp3", "ogg")
        """
        if not self.pydub_available:
            print("❌ pydub недоступен")
            return
        
        try:
            audio.export(file_path, format=format)
            print(f"✅ Файл сохранен: {file_path}")
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
    
    def create_melody_file(self, notes: List[tuple], output_path: str):
        """
        Создание мелодии из нот и сохранение в файл
        
        Args:
            notes: Список кортежей (частота, длительность_мс)
            output_path: Путь для сохранения
        """
        if not self.pydub_available:
            print("❌ pydub недоступен")
            return
        
        try:
            melody = AudioSegment.empty()
            
            for freq, duration in notes:
                if freq == 0:  # Пауза
                    note = AudioSegment.silent(duration=duration)
                else:
                    note = self.generate_tone_file(freq, duration)
                
                if note:
                    melody += note
            
            if melody:
                self.save_audio_segment(melody, output_path)
                print(f"🎵 Мелодия создана: {output_path}")
                return melody
            
        except Exception as e:
            print(f"❌ Ошибка создания мелодии: {e}")
        
        return None
    
    def apply_effects(self, audio: AudioSegment, effects: dict) -> AudioSegment:
        """
        Применение эффектов к аудио
        
        Args:
            audio: Исходное аудио
            effects: Словарь с эффектами
        
        Returns:
            Обработанное аудио
        """
        if not self.pydub_available:
            return audio
        
        result = audio
        
        try:
            # Изменение громкости
            if "volume_db" in effects:
                result = result + effects["volume_db"]
                print(f"🔊 Применен эффект громкости: {effects['volume_db']} dB")
            
            # Fade in/out
            if "fade_in_ms" in effects:
                result = result.fade_in(effects["fade_in_ms"])
                print(f"📈 Применен fade-in: {effects['fade_in_ms']} мс")
            
            if "fade_out_ms" in effects:
                result = result.fade_out(effects["fade_out_ms"])
                print(f"📉 Применен fade-out: {effects['fade_out_ms']} мс")
            
            # Повтор
            if "repeat" in effects:
                result = result * effects["repeat"]
                print(f"🔄 Повтор: {effects['repeat']} раз")
            
            # Реверс
            if effects.get("reverse", False):
                result = result.reverse()
                print("🔄 Применен реверс")
            
        except Exception as e:
            print(f"❌ Ошибка применения эффектов: {e}")
        
        return result
    
    def create_notification_sounds(self):
        """Создание набора звуков уведомлений"""
        if not self.pydub_available:
            print("❌ pydub недоступен")
            return
        
        print("🔔 Создание звуков уведомлений...")
        
        # Простой сигнал
        beep = self.generate_tone_file(800, 200)
        if beep:
            self.save_audio_segment(beep, "notification_beep.wav")
        
        # Двойной сигнал
        double_beep = beep + AudioSegment.silent(100) + beep
        if double_beep:
            self.save_audio_segment(double_beep, "notification_double_beep.wav")
        
        # Мелодичный сигнал
        melody_notes = [(523, 150), (659, 150), (784, 300)]  # C5-E5-G5
        melody = AudioSegment.empty()
        for freq, duration in melody_notes:
            note = self.generate_tone_file(freq, duration)
            if note:
                melody += note
        
        if melody:
            self.save_audio_segment(melody, "notification_melody.wav")
        
        # Сигнал ошибки
        error_sound = self.generate_tone_file(300, 500, "square", 0.6)
        if error_sound:
            self.save_audio_segment(error_sound, "notification_error.wav")
        
        print("✅ Звуки уведомлений созданы")


def demo_audio_files():
    """Демонстрация работы с аудиофайлами"""
    print("🎵 Демонстрация работы с аудиофайлами")
    print("=" * 40)
    
    player = AudioFilePlayer()
    
    if not player.pydub_available:
        print("❌ pydub недоступен. Установите: pip install pydub")
        return
    
    try:
        while True:
            print("\n📁 Выберите действие:")
            print("1. Создать и воспроизвести тон")
            print("2. Создать мелодию 'Happy Birthday'")
            print("3. Создать звуки уведомлений")
            print("4. Загрузить и воспроизвести файл")
            print("5. Создать тон с эффектами")
            print("6. Создать DTMF последовательность")
            print("0. Выход")
            
            choice = input("\n👆 Ваш выбор: ").strip()
            
            if choice == "1":
                freq = float(input("Частота (Hz): ") or "440")
                duration = int(input("Длительность (мс): ") or "1000")
                wave_type = input("Тип волны (sine/square/sawtooth/triangle): ") or "sine"
                
                audio = player.generate_tone_file(freq, duration, wave_type)
                if audio:
                    print("🔊 Воспроизведение...")
                    player.play_audio_segment(audio)
            
            elif choice == "2":
                # Happy Birthday мелодия
                notes = [
                    (262, 400), (262, 200), (294, 600), (262, 600), (349, 600), (330, 1200),  # Happy birthday to
                    (262, 400), (262, 200), (294, 600), (262, 600), (392, 600), (349, 1200),  # you, happy birthday to
                    (262, 400), (262, 200), (523, 600), (440, 600), (349, 600), (330, 600), (294, 600),  # you, happy birthday dear
                    (466, 400), (466, 200), (440, 600), (349, 600), (392, 600), (349, 1200)   # [name], happy birthday to you
                ]
                
                melody = player.create_melody_file(notes, "happy_birthday.wav")
                if melody:
                    print("🎂 Воспроизведение Happy Birthday...")
                    player.play_audio_segment(melody)
            
            elif choice == "3":
                player.create_notification_sounds()
            
            elif choice == "4":
                file_path = input("Путь к аудиофайлу: ").strip()
                audio = player.load_audio_file(file_path)
                if audio:
                    print("🔊 Воспроизведение файла...")
                    player.play_audio_segment(audio)
            
            elif choice == "5":
                # Создание тона с эффектами
                audio = player.generate_tone_file(440, 2000)
                if audio:
                    effects = {
                        "fade_in_ms": 500,
                        "fade_out_ms": 500,
                        "volume_db": -10
                    }
                    
                    processed_audio = player.apply_effects(audio, effects)
                    print("🎛️ Воспроизведение с эффектами...")
                    player.play_audio_segment(processed_audio)
            
            elif choice == "6":
                # DTMF последовательность
                number = input("Введите номер для DTMF: ") or "123456"
                dtmf_freqs = {
                    '1': (697, 1209), '2': (697, 1336), '3': (697, 1477),
                    '4': (770, 1209), '5': (770, 1336), '6': (770, 1477),
                    '7': (852, 1209), '8': (852, 1336), '9': (852, 1477),
                    '0': (941, 1336), '*': (941, 1209), '#': (941, 1477)
                }
                
                dtmf_sequence = AudioSegment.empty()
                
                for digit in number:
                    if digit in dtmf_freqs:
                        freq1, freq2 = dtmf_freqs[digit]
                        tone1 = player.generate_tone_file(freq1, 200)
                        tone2 = player.generate_tone_file(freq2, 200)
                        
                        if tone1 and tone2:
                            # Смешиваем два тона
                            combined = tone1.overlay(tone2)
                            dtmf_sequence += combined + AudioSegment.silent(100)
                
                if dtmf_sequence:
                    player.save_audio_segment(dtmf_sequence, f"dtmf_{number}.wav")
                    print(f"📞 Воспроизведение DTMF для {number}...")
                    player.play_audio_segment(dtmf_sequence)
            
            elif choice == "0":
                break
            
            else:
                print("❌ Неверный выбор")
    
    except KeyboardInterrupt:
        print("\n⏹️ Остановлено пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    print("👋 До свидания!")


if __name__ == "__main__":
    import math
    demo_audio_files()