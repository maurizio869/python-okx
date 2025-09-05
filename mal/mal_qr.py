# Last modified (MSK): 2025-09-05 15:26 — правка номер 2
import requests
import time
import hmac
import hashlib
import os
from pathlib import Path
from dotenv import load_dotenv
# путь к .env: та же папка, где лежит текущий файл
dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path)           # <-- читаем файл

def place_futures_order():
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')

    # Параметры ордера
    params = {
        "symbol": "HBAR-USDT",
        "side": "SELL",
        "type": "MARKET", 
        "quantity": "3",
        "leverage": "10",
        "positionSide": "SHORT",
        "timeInForce": "IOC"
    }

    # Создание подписи согласно BingX API документации
    timestamp = str(int(time.time() * 1000))
    
    # Сортируем параметры по ключу и создаем query string
    sorted_params = sorted(params.items())
    query_string = "&".join([f"{k}={v}" for k, v in sorted_params])
    query_string += f"&timestamp={timestamp}"
    
    # Генерируем подпись: HMAC-SHA256(query_string, secret_key)
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    # Заголовки (только ASCII символы)
    headers = {
        "X-BX-APIKEY": api_key,
        "X-BX-SIGNATURE": signature,
        "X-BX-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }

    try:
        # Отправка запроса
        response = requests.post(
            "https://open-api.bingx.com/openApi/swap/v2/trade/order",
            json=params,
            headers=headers,
            timeout=10
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"General Error: {e}")
        return {"error": str(e)}

# Тестируем функцию
if __name__ == "__main__":
    print("Запуск теста API...")
    result = place_futures_order()
    print(f"Результат: {result}")