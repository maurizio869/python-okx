import os
import hmac
import hashlib
import time
from pathlib import Path
from dotenv import load_dotenv

# путь к .env: та же папка, где лежит текущий файл
dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path)           # <-- читаем файл

def debug_api_keys():
    """Диагностика API ключей"""
    
    print(f"Путь к .env: {dotenv_path}")
    print(f"Файл существует: {dotenv_path.exists()}")
    
    if not dotenv_path.exists():
        print("❌ .env файл не найден!")
        return
    
    # Читаем содержимое файла
    with open(dotenv_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"Содержимое .env:")
        print(repr(content))
    
    print("✅ .env загружен")
    
    # Получаем ключи
    api_key = os.getenv('api_key1')
    api_secret = os.getenv('api_secret1')
    
    print(f"API Key: {repr(api_key)}")
    print(f"API Secret: {repr(api_secret)}")
    
    if not api_key or not api_secret:
        print("❌ Ключи не найдены!")
        return
    
    # Проверяем длину ключей
    print(f"Длина API Key: {len(api_key)}")
    print(f"Длина API Secret: {len(api_secret)}")
    
    # Проверяем символы
    print(f"API Key содержит только ASCII: {api_key.isascii()}")
    print(f"API Secret содержит только ASCII: {api_secret.isascii()}")
    
    # Тестируем создание подписи
    test_params = {
        "symbol": "HBAR-USDT",
        "side": "SELL",
        "type": "MARKET", 
        "quantity": "3",
        "leverage": "10",
        "positionSide": "SHORT",
        "timeInForce": "IOC"
    }
    
    timestamp = str(int(time.time() * 1000))
    query_string = "&".join([f"{k}={v}" for k, v in test_params.items()])
    query_string += f"&timestamp={timestamp}"
    
    print(f"Query String: {query_string}")
    
    try:
        signature = hmac.new(
            api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        print(f"Signature: {signature}")
        print(f"Длина подписи: {len(signature)}")
        print("✅ Подпись создана успешно")
        
    except Exception as e:
        print(f"❌ Ошибка создания подписи: {e}")

if __name__ == "__main__":
    debug_api_keys()