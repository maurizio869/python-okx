# Last modified (MSK): 2025-09-05 15:45 — правка номер 1
import requests
import time
import hmac
import hashlib
import os
from pathlib import Path
from dotenv import load_dotenv

# путь к .env: та же папка, где лежит текущий файл
dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path)

def test_api_keys():
    """Тестируем валидность API ключей"""
    print("=== Тестируем валидность API ключей ===")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    print(f"API Key длина: {len(api_key)}")
    print(f"API Secret длина: {len(api_secret)}")
    print(f"API Key начинается с: {api_key[:5]}...")
    print(f"API Secret начинается с: {api_secret[:5]}...")
    
    # Проверяем, что ключи не являются дефолтными
    if api_key == '1' or api_secret == '1':
        print("❌ Используются дефолтные ключи! Нужно заменить на реальные.")
        return False
    
    print("✅ API ключи загружены")
    return True

def test_headers_format():
    """Тестируем разные форматы заголовков"""
    print("\n=== Тестируем форматы заголовков ===")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    timestamp = str(int(time.time() * 1000))
    
    # Простой запрос для получения баланса
    params = {}
    query_string = f"timestamp={timestamp}"
    
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # Формат 1: Стандартный
    headers1 = {
        "X-BX-APIKEY": api_key,
        "X-BX-SIGNATURE": signature,
        "X-BX-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    
    # Формат 2: Строчные буквы
    headers2 = {
        "x-bx-apikey": api_key,
        "x-bx-signature": signature,
        "x-bx-timestamp": timestamp,
        "content-type": "application/json"
    }
    
    # Формат 3: Другой порядок
    headers3 = {
        "Content-Type": "application/json",
        "X-BX-TIMESTAMP": timestamp,
        "X-BX-SIGNATURE": signature,
        "X-BX-APIKEY": api_key
    }
    
    url = "https://open-api.bingx.com/openApi/swap/v2/user/getBalance"
    
    for i, headers in enumerate([headers1, headers2, headers3], 1):
        print(f"\n--- Тестируем формат заголовков {i} ---")
        print(f"Заголовки: {headers}")
        
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=10
            )
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') != 100412:  # Не "Null signature"
                    print(f"✅ Формат {i} работает!")
                    return headers
                else:
                    print(f"❌ Формат {i}: Null signature")
            else:
                print(f"❌ Формат {i}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Формат {i}: Ошибка - {e}")
    
    return None

def test_different_requests():
    """Тестируем разные типы запросов"""
    print("\n=== Тестируем разные типы запросов ===")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    timestamp = str(int(time.time() * 1000))
    
    # Тест 1: GET запрос с параметрами в URL
    print("\n--- Тест 1: GET с параметрами в URL ---")
    params = {"timestamp": timestamp}
    query_string = f"timestamp={timestamp}"
    
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "X-BX-APIKEY": api_key,
        "X-BX-SIGNATURE": signature,
        "X-BX-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(
            "https://open-api.bingx.com/openApi/swap/v2/user/getBalance",
            params=params,
            headers=headers,
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Ошибка: {e}")
    
    # Тест 2: POST запрос с JSON телом
    print("\n--- Тест 2: POST с JSON телом ---")
    order_params = {
        "symbol": "HBAR-USDT",
        "side": "SELL",
        "type": "MARKET", 
        "quantity": "3",
        "leverage": "10",
        "positionSide": "SHORT",
        "timeInForce": "IOC"
    }
    
    # Сортируем параметры
    sorted_params = sorted(order_params.items())
    query_string = "&".join([f"{k}={v}" for k, v in sorted_params])
    query_string += f"&timestamp={timestamp}"
    
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "X-BX-APIKEY": api_key,
        "X-BX-SIGNATURE": signature,
        "X-BX-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            "https://open-api.bingx.com/openApi/swap/v2/trade/order",
            json=order_params,
            headers=headers,
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Ошибка: {e}")
    
    # Тест 3: POST запрос с данными в теле (form-data)
    print("\n--- Тест 3: POST с form-data ---")
    try:
        response = requests.post(
            "https://open-api.bingx.com/openApi/swap/v2/trade/order",
            data=order_params,
            headers=headers,
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Ошибка: {e}")

def test_raw_request():
    """Тестируем сырой HTTP запрос"""
    print("\n=== Тестируем сырой HTTP запрос ===")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    timestamp = str(int(time.time() * 1000))
    
    # Создаем подпись
    query_string = f"timestamp={timestamp}"
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # Создаем сырой запрос
    url = "https://open-api.bingx.com/openApi/swap/v2/user/getBalance"
    params = {"timestamp": timestamp}
    
    print(f"URL: {url}")
    print(f"Параметры: {params}")
    print(f"Query String: {query_string}")
    print(f"Signature: {signature}")
    
    headers = {
        "X-BX-APIKEY": api_key,
        "X-BX-SIGNATURE": signature,
        "X-BX-TIMESTAMP": timestamp,
        "Content-Type": "application/json",
        "User-Agent": "Python/3.13"
    }
    
    print(f"Заголовки: {headers}")
    
    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response: {response.text}")
        
        # Проверяем, что заголовки действительно отправлены
        print(f"\nОтправленные заголовки:")
        for key, value in headers.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    print("🔍 Глубокая диагностика BingX API...")
    
    # Тестируем API ключи
    if not test_api_keys():
        print("❌ Проблема с API ключами")
        exit(1)
    
    # Тестируем форматы заголовков
    working_headers = test_headers_format()
    if working_headers:
        print(f"✅ Найден рабочий формат заголовков: {working_headers}")
    else:
        print("❌ Ни один формат заголовков не работает")
    
    # Тестируем разные типы запросов
    test_different_requests()
    
    # Тестируем сырой запрос
    test_raw_request()
    
    print("\n🔍 Диагностика завершена")