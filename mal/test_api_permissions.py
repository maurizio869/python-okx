# Last modified (MSK): 2025-09-06 08:20 — правка номер 1
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

def test_read_only_operations():
    """Тестируем операции только для чтения"""
    print("🔍 Тестирование операций только для чтения...")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    timestamp = str(int(time.time() * 1000))
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
    
    # Тестируем разные операции только для чтения
    read_only_endpoints = [
        "/openApi/swap/v2/user/getBalance",
        "/openApi/swap/v2/user/getAccount",
        "/openApi/swap/v2/user/getPositions",
        "/openApi/swap/v2/user/getOrders",
        "/openApi/swap/v2/market/getTicker",
        "/openApi/swap/v2/market/getDepth"
    ]
    
    working_endpoints = []
    
    for endpoint in read_only_endpoints:
        print(f"\n--- Тестируем {endpoint} ---")
        try:
            response = requests.get(
                f"https://open-api.bingx.com{endpoint}",
                params={"timestamp": timestamp},
                headers=headers,
                timeout=10
            )
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') != 100412:  # Не "Null signature"
                    print(f"✅ {endpoint} работает!")
                    working_endpoints.append(endpoint)
                else:
                    print(f"❌ {endpoint}: Null signature")
            else:
                print(f"❌ {endpoint}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ {endpoint}: Ошибка - {e}")
    
    return working_endpoints

def test_without_signature():
    """Тестируем запросы без подписи (публичные endpoints)"""
    print("\n🔍 Тестирование публичных endpoints без подписи...")
    
    # Публичные endpoints (не требуют подписи)
    public_endpoints = [
        "/openApi/swap/v2/market/getTicker",
        "/openApi/swap/v2/market/getDepth",
        "/openApi/swap/v2/market/getKlines",
        "/openApi/swap/v2/market/getTrades"
    ]
    
    working_public = []
    
    for endpoint in public_endpoints:
        print(f"\n--- Тестируем {endpoint} (публичный) ---")
        try:
            response = requests.get(
                f"https://open-api.bingx.com{endpoint}",
                params={"symbol": "BTC-USDT"},
                timeout=10
            )
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text[:200]}...")  # Показываем только первые 200 символов
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 0 or 'data' in result:  # Успешный ответ
                    print(f"✅ {endpoint} работает!")
                    working_public.append(endpoint)
                else:
                    print(f"❌ {endpoint}: {result.get('msg', 'Unknown error')}")
            else:
                print(f"❌ {endpoint}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ {endpoint}: Ошибка - {e}")
    
    return working_public

def test_api_key_format():
    """Проверяем формат API ключей"""
    print("\n🔍 Проверка формата API ключей...")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    print(f"API Key: {api_key}")
    print(f"API Secret: {api_secret}")
    print(f"API Key длина: {len(api_key)}")
    print(f"API Secret длина: {len(api_secret)}")
    
    # Проверяем, что ключи не являются дефолтными
    if api_key == '1' or api_secret == '1':
        print("❌ Используются дефолтные ключи!")
        return False
    
    # Проверяем длину (обычно API ключи имеют определенную длину)
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
    return True

def test_simple_signature():
    """Тестируем простейшую подпись"""
    print("\n🔍 Тестирование простейшей подписи...")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    timestamp = str(int(time.time() * 1000))
    
    # Самая простая подпись - только timestamp
    signature = hmac.new(
        api_secret.encode('utf-8'),
        timestamp.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "X-BX-APIKEY": api_key,
        "X-BX-SIGNATURE": signature,
        "X-BX-TIMESTAMP": timestamp
    }
    
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
                print("✅ Простейшая подпись работает!")
                return True
            else:
                print("❌ Простейшая подпись: Null signature")
        else:
            print(f"❌ Простейшая подпись: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Простейшая подпись: Ошибка - {e}")
    
    return False

if __name__ == "__main__":
    print("🔍 Тестирование API ключей и разрешений...")
    
    # Проверяем формат API ключей
    if not test_api_key_format():
        print("\n❌ Проблема с форматом API ключей")
        exit(1)
    
    # Тестируем простейшую подпись
    if test_simple_signature():
        print("\n✅ Простейшая подпись работает!")
    else:
        print("\n❌ Простейшая подпись не работает")
    
    # Тестируем публичные endpoints
    working_public = test_without_signature()
    if working_public:
        print(f"\n✅ Работают публичные endpoints: {working_public}")
    else:
        print("\n❌ Публичные endpoints не работают")
    
    # Тестируем операции только для чтения
    working_read = test_read_only_operations()
    if working_read:
        print(f"\n✅ Работают операции для чтения: {working_read}")
    else:
        print("\n❌ Операции для чтения не работают")
    
    print("\n🔍 Тестирование завершено")