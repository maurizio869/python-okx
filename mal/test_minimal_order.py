# Last modified (MSK): 2025-09-06 08:55 — правка номер 1
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

def test_minimal_order():
    """Тестируем минимальный ордер с разными подходами"""
    print("🔍 Тестирование минимального ордера...")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    timestamp = str(int(time.time() * 1000))
    
    # Минимальные параметры ордера
    minimal_params = {
        "symbol": "BTC-USDT",
        "side": "BUY",
        "type": "MARKET",
        "quantity": "0.001"
    }
    
    print(f"API Key: {api_key[:10]}...")
    print(f"API Secret: {api_secret[:10]}...")
    print(f"Timestamp: {timestamp}")
    
    # Тест 1: Query string подпись
    print("\n=== Тест 1: Query string подпись ===")
    query_params = minimal_params.copy()
    query_params['timestamp'] = timestamp
    sorted_params = sorted(query_params.items())
    query_string = "&".join([f"{k}={v}" for k, v in sorted_params])
    
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    print(f"Query String: {query_string}")
    print(f"Signature: {signature}")
    
    headers = {
        "X-BX-APIKEY": api_key,
        "X-BX-SIGNATURE": signature,
        "X-BX-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            "https://open-api.bingx.com/openApi/swap/v2/trade/order",
            json=minimal_params,
            headers=headers,
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('code') != 100412:
                print("✅ Тест 1: Успех!")
                return True
            else:
                print("❌ Тест 1: Null signature")
        else:
            print(f"❌ Тест 1: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Тест 1: Ошибка - {e}")
    
    # Тест 2: Только timestamp в подписи
    print("\n=== Тест 2: Только timestamp в подписи ===")
    signature2 = hmac.new(
        api_secret.encode('utf-8'),
        timestamp.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    print(f"Signature (только timestamp): {signature2}")
    
    headers2 = {
        "X-BX-APIKEY": api_key,
        "X-BX-SIGNATURE": signature2,
        "X-BX-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            "https://open-api.bingx.com/openApi/swap/v2/trade/order",
            json=minimal_params,
            headers=headers2,
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('code') != 100412:
                print("✅ Тест 2: Успех!")
                return True
            else:
                print("❌ Тест 2: Null signature")
        else:
            print(f"❌ Тест 2: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Тест 2: Ошибка - {e}")
    
    # Тест 3: GET запрос вместо POST
    print("\n=== Тест 3: GET запрос вместо POST ===")
    try:
        response = requests.get(
            "https://open-api.bingx.com/openApi/swap/v2/trade/order",
            params=query_params,
            headers=headers,
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('code') != 100412:
                print("✅ Тест 3: Успех!")
                return True
            else:
                print("❌ Тест 3: Null signature")
        else:
            print(f"❌ Тест 3: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Тест 3: Ошибка - {e}")
    
    return False

def test_api_key_validity():
    """Проверяем валидность API ключей"""
    print("\n🔍 Проверка валидности API ключей...")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    # Проверяем, что ключи не дефолтные
    if api_key == '1' or api_secret == '1':
        print("❌ Используются дефолтные ключи!")
        return False
    
    # Проверяем длину
    if len(api_key) < 20 or len(api_secret) < 20:
        print("❌ API ключи слишком короткие!")
        return False
    
    # Проверяем символы
    import string
    allowed_chars = string.ascii_letters + string.digits
    
    for char in api_key:
        if char not in allowed_chars:
            print(f"❌ API Key содержит недопустимый символ: {char}")
            return False
    
    for char in api_secret:
        if char not in allowed_chars:
            print(f"❌ API Secret содержит недопустимый символ: {char}")
            return False
    
    print("✅ API ключи выглядят корректно")
    print(f"API Key длина: {len(api_key)}")
    print(f"API Secret длина: {len(api_secret)}")
    print(f"API Key начинается с: {api_key[:10]}...")
    print(f"API Secret начинается с: {api_secret[:10]}...")
    
    return True

def test_different_base_urls():
    """Тестируем разные базовые URL"""
    print("\n🔍 Тестирование разных базовых URL...")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    timestamp = str(int(time.time() * 1000))
    signature = hmac.new(
        api_secret.encode('utf-8'),
        timestamp.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "X-BX-APIKEY": api_key,
        "X-BX-SIGNATURE": signature,
        "X-BX-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    
    base_urls = [
        "https://open-api.bingx.com",
        "https://api.bingx.com",
        "https://openapi.bingx.com",
        "https://api.bingx.com/openApi"
    ]
    
    for base_url in base_urls:
        print(f"\n--- Тестируем {base_url} ---")
        try:
            response = requests.get(
                f"{base_url}/openApi/swap/v2/user/getBalance",
                params={"timestamp": timestamp},
                headers=headers,
                timeout=10
            )
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') != 100400:  # Не "api is not exist"
                    print(f"✅ {base_url} работает!")
                    return base_url
                else:
                    print(f"❌ {base_url}: api is not exist")
            else:
                print(f"❌ {base_url}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ {base_url}: Ошибка - {e}")
    
    return None

if __name__ == "__main__":
    print("🔍 Комплексное тестирование API...")
    
    # Проверяем валидность API ключей
    if not test_api_key_validity():
        print("\n❌ Проблема с API ключами")
        exit(1)
    
    # Тестируем разные базовые URL
    working_url = test_different_base_urls()
    if working_url:
        print(f"\n✅ Найден рабочий URL: {working_url}")
    else:
        print("\n❌ Не найден рабочий URL")
    
    # Тестируем минимальный ордер
    if test_minimal_order():
        print("\n✅ Минимальный ордер работает!")
    else:
        print("\n❌ Минимальный ордер не работает")
    
    print("\n🔍 Тестирование завершено")