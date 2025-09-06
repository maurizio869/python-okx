# Last modified (MSK): 2025-09-06 09:00 — правка номер 1
import os
from pathlib import Path
from dotenv import load_dotenv

# путь к .env: та же папка, где лежит текущий файл
dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path)

def check_api_keys_detailed():
    """Детальная проверка API ключей"""
    print("🔍 Детальная проверка API ключей...")
    
    # Читаем содержимое .env файла
    print(f"\n=== Содержимое .env файла ===")
    print(f"Путь к .env: {dotenv_path}")
    print(f"Файл существует: {dotenv_path.exists()}")
    
    if dotenv_path.exists():
        with open(dotenv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"Содержимое файла:")
        print(repr(content))
    
    # Получаем ключи
    api_key = os.getenv('api_key1')
    api_secret = os.getenv('api_secret1')
    
    print(f"\n=== Загруженные ключи ===")
    print(f"API Key: {repr(api_key)}")
    print(f"API Secret: {repr(api_secret)}")
    
    if not api_key or not api_secret:
        print("❌ Ключи не найдены!")
        return False
    
    # Проверяем длину
    print(f"\n=== Проверка длины ===")
    print(f"API Key длина: {len(api_key)}")
    print(f"API Secret длина: {len(api_secret)}")
    
    # Проверяем символы
    print(f"\n=== Проверка символов ===")
    import string
    allowed_chars = string.ascii_letters + string.digits
    
    api_key_clean = True
    api_secret_clean = True
    
    for i, char in enumerate(api_key):
        if char not in allowed_chars:
            print(f"❌ API Key символ {i}: {repr(char)} (недопустимый)")
            api_key_clean = False
    
    for i, char in enumerate(api_secret):
        if char not in allowed_chars:
            print(f"❌ API Secret символ {i}: {repr(char)} (недопустимый)")
            api_secret_clean = False
    
    if api_key_clean and api_secret_clean:
        print("✅ Все символы допустимы")
    
    # Проверяем начало и конец
    print(f"\n=== Проверка начала и конца ===")
    print(f"API Key начинается с: {repr(api_key[:5])}")
    print(f"API Key заканчивается на: {repr(api_key[-5:])}")
    print(f"API Secret начинается с: {repr(api_secret[:5])}")
    print(f"API Secret заканчивается на: {repr(api_secret[-5:])}")
    
    # Проверяем на скрытые символы
    print(f"\n=== Проверка на скрытые символы ===")
    api_key_bytes = api_key.encode('utf-8')
    api_secret_bytes = api_secret.encode('utf-8')
    
    print(f"API Key байты: {api_key_bytes}")
    print(f"API Secret байты: {api_secret_bytes}")
    
    # Проверяем на пробелы в начале/конце
    api_key_stripped = api_key.strip()
    api_secret_stripped = api_secret.strip()
    
    if api_key != api_key_stripped:
        print(f"❌ API Key содержит пробелы в начале/конце!")
        print(f"Оригинал: {repr(api_key)}")
        print(f"Без пробелов: {repr(api_key_stripped)}")
    else:
        print("✅ API Key не содержит пробелов в начале/конце")
    
    if api_secret != api_secret_stripped:
        print(f"❌ API Secret содержит пробелы в начале/конце!")
        print(f"Оригинал: {repr(api_secret)}")
        print(f"Без пробелов: {repr(api_secret_stripped)}")
    else:
        print("✅ API Secret не содержит пробелов в начале/конце")
    
    return True

def test_api_key_with_different_formats():
    """Тестируем API ключи с разными форматами"""
    print("\n🔍 Тестирование API ключей с разными форматами...")
    
    api_key = os.getenv('api_key1')
    api_secret = os.getenv('api_secret1')
    
    if not api_key or not api_secret:
        print("❌ Ключи не найдены!")
        return
    
    # Разные варианты обработки ключей
    key_variants = [
        ("Оригинал", api_key, api_secret),
        ("Без пробелов", api_key.strip(), api_secret.strip()),
        ("Верхний регистр", api_key.upper(), api_secret.upper()),
        ("Нижний регистр", api_key.lower(), api_secret.lower()),
    ]
    
    for name, test_key, test_secret in key_variants:
        print(f"\n--- Тестируем {name} ---")
        print(f"API Key: {repr(test_key)}")
        print(f"API Secret: {repr(test_secret)}")
        
        # Простой тест подписи
        import time
        import hmac
        import hashlib
        
        timestamp = str(int(time.time() * 1000))
        signature = hmac.new(
            test_secret.encode('utf-8'),
            timestamp.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        print(f"Тестовая подпись: {signature}")
        print(f"Длина подписи: {len(signature)}")

if __name__ == "__main__":
    print("🔍 Проверка новых API ключей...")
    
    if check_api_keys_detailed():
        test_api_key_with_different_formats()
    
    print("\n🔍 Проверка завершена")