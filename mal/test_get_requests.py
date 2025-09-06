# Last modified (MSK): 2025-09-06 08:10 — правка номер 1
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

def test_get_requests():
    """Тестируем GET запросы (они проще для подписи)"""
    print("🔍 Тестирование GET запросов...")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    timestamp = str(int(time.time() * 1000))
    
    # Тест 1: Получение баланса (простой GET запрос)
    print("\n=== Тест 1: Получение баланса ===")
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
        
        if response.status_code == 200:
            result = response.json()
            if result.get('code') != 100412:  # Не "Null signature"
                print("✅ GET запрос работает!")
                return True
            else:
                print("❌ GET запрос: Null signature")
        else:
            print(f"❌ GET запрос: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ GET запрос: Ошибка - {e}")
    
    return False

def test_different_headers():
    """Тестируем разные форматы заголовков"""
    print("\n🔍 Тестирование разных форматов заголовков...")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    timestamp = str(int(time.time() * 1000))
    query_string = f"timestamp={timestamp}"
    
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # Разные форматы заголовков
    header_formats = [
        # Формат 1: Стандартный
        {
            "X-BX-APIKEY": api_key,
            "X-BX-SIGNATURE": signature,
            "X-BX-TIMESTAMP": timestamp,
            "Content-Type": "application/json"
        },
        # Формат 2: Без Content-Type
        {
            "X-BX-APIKEY": api_key,
            "X-BX-SIGNATURE": signature,
            "X-BX-TIMESTAMP": timestamp
        },
        # Формат 3: Строчные буквы
        {
            "x-bx-apikey": api_key,
            "x-bx-signature": signature,
            "x-bx-timestamp": timestamp
        },
        # Формат 4: Другой порядок
        {
            "Content-Type": "application/json",
            "X-BX-TIMESTAMP": timestamp,
            "X-BX-SIGNATURE": signature,
            "X-BX-APIKEY": api_key
        }
    ]
    
    for i, headers in enumerate(header_formats, 1):
        print(f"\n--- Тестируем формат заголовков {i} ---")
        print(f"Заголовки: {headers}")
        
        try:
            response = requests.get(
                "https://open-api.bingx.com/openApi/swap/v2/user/getBalance",
                params={"timestamp": timestamp},
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

def test_simple_post():
    """Тестируем простой POST запрос"""
    print("\n🔍 Тестирование простого POST запроса...")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    timestamp = str(int(time.time() * 1000))
    
    # Простой POST запрос без сложных параметров
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
        response = requests.post(
            "https://open-api.bingx.com/openApi/swap/v2/user/getBalance",
            json=params,
            headers=headers,
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('code') != 100412:  # Не "Null signature"
                print("✅ Простой POST работает!")
                return True
            else:
                print("❌ Простой POST: Null signature")
        else:
            print(f"❌ Простой POST: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Простой POST: Ошибка - {e}")
    
    return False

if __name__ == "__main__":
    print("🔍 Тестирование разных подходов к API...")
    
    # Тестируем GET запросы
    if test_get_requests():
        print("\n✅ GET запросы работают!")
    else:
        print("\n❌ GET запросы не работают")
    
    # Тестируем разные форматы заголовков
    working_headers = test_different_headers()
    if working_headers:
        print(f"\n✅ Найден рабочий формат заголовков: {working_headers}")
    else:
        print("\n❌ Ни один формат заголовков не работает")
    
    # Тестируем простой POST
    if test_simple_post():
        print("\n✅ Простой POST работает!")
    else:
        print("\n❌ Простой POST не работает")
    
    print("\n🔍 Тестирование завершено")