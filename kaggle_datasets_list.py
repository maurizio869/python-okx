#!/usr/bin/env python3
"""
Скрипт для вывода всех доступных датасетов в текущей Kaggle сессии
"""

import os
from pathlib import Path

def list_kaggle_datasets():
    """
    Выводит пути и названия всех датасетов в текущей Kaggle сессии
    """
    # Стандартная директория для датасетов в Kaggle
    kaggle_input_dir = Path('/kaggle/input')
    
    # Проверяем, существует ли директория (работаем ли мы в Kaggle)
    if not kaggle_input_dir.exists():
        print("❌ Директория /kaggle/input не найдена.")
        print("Возможно, скрипт запущен не в среде Kaggle.")
        return
    
    # Получаем список всех датасетов
    datasets = [item for item in kaggle_input_dir.iterdir() if item.is_dir()]
    
    if not datasets:
        print("📂 В текущей сессии нет подключенных датасетов.")
        return
    
    print(f"📊 Найдено датасетов в сессии: {len(datasets)}\n")
    print("=" * 60)
    
    for i, dataset_path in enumerate(sorted(datasets), 1):
        dataset_name = dataset_path.name
        
        print(f"{i}. Название: {dataset_name}")
        print(f"   Путь: {dataset_path}")
        
        # Показываем размер датасета
        try:
            total_size = sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            print(f"   Размер: {size_mb:.2f} MB")
        except:
            print(f"   Размер: не удалось определить")
        
        # Показываем количество файлов
        try:
            file_count = len([f for f in dataset_path.rglob('*') if f.is_file()])
            print(f"   Файлов: {file_count}")
        except:
            print(f"   Файлов: не удалось определить")
            
        # Показываем несколько первых файлов
        try:
            files = list(dataset_path.rglob('*'))[:5]  # Первые 5 файлов
            if files:
                print(f"   Примеры файлов:")
                for file_path in files:
                    if file_path.is_file():
                        print(f"     - {file_path.name}")
        except:
            pass
            
        print("-" * 60)

def list_datasets_simple():
    """
    Упрощенная версия - только названия и пути
    """
    kaggle_input_dir = Path('/kaggle/input')
    
    if not kaggle_input_dir.exists():
        print("❌ Не в среде Kaggle")
        return
    
    datasets = [item for item in kaggle_input_dir.iterdir() if item.is_dir()]
    
    print("📊 Датасеты в сессии:")
    for dataset in sorted(datasets):
        print(f"• {dataset.name} -> {dataset}")

if __name__ == "__main__":
    print("🔍 Поиск датасетов Kaggle в текущей сессии...\n")
    
    # Запускаем подробную версию
    list_kaggle_datasets()
    
    print("\n" + "=" * 60)
    print("💡 Для использования датасета в коде:")
    print("   import pandas as pd")
    print("   df = pd.read_csv('/kaggle/input/НАЗВАНИЕ_ДАТАСЕТА/файл.csv')")