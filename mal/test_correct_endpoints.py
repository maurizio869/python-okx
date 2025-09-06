# Last modified (MSK): 2025-09-06 08:40 — правка номер 1
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

def test_different_endpoint_formats():
    """Тестируем разные форматы endpoints"""
    print("🔍 Тестирование разных форматов endpoints...")
    
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
    
    # Разные варианты endpoints для тестирования
    endpoint_variants = [
        # Вариант 1: Текущий (не работает)
        "/openApi/swap/v2/user/getBalance",
        
        # Вариант 2: Без v2
        "/openApi/swap/user/getBalance",
        
        # Вариант 3: С v1
        "/openApi/swap/v1/user/getBalance",
        
        # Вариант 4: Другой путь
        "/openApi/futures/user/getBalance",
        
        # Вариант 5: Еще один вариант
        "/openApi/derivatives/user/getBalance",
        
        # Вариант 6: Простой путь
        "/openApi/user/getBalance",
        
        # Вариант 7: С api
        "/api/swap/v2/user/getBalance",
        
        # Вариант 8: С v3
        "/openApi/swap/v3/user/getBalance",
        
        # Вариант 9: Другой формат
        "/openApi/swap/user/balance",
        
        # Вариант 10: Еще один
        "/openApi/swap/v2/user/balance"
    ]
    
    working_endpoints = []
    
    for endpoint in endpoint_variants:
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
                # Ищем успешные ответы (не "api is not exist" и не "Null signature")
                if (result.get('code') != 100400 and 
                    result.get('code') != 100412 and
                    'msg' not in result.get('msg', '').lower() or 
                    'success' in result.get('msg', '').lower() or
                    'data' in result):
                    print(f"✅ {endpoint} работает!")
                    working_endpoints.append(endpoint)
                else:
                    print(f"❌ {endpoint}: {result.get('msg', 'Unknown error')}")
            else:
                print(f"❌ {endpoint}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ {endpoint}: Ошибка - {e}")
    
    return working_endpoints

def test_trading_endpoints():
    """Тестируем endpoints для торговли"""
    print("\n🔍 Тестирование endpoints для торговли...")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    timestamp = str(int(time.time() * 1000))
    
    # Параметры для торгового ордера
    params = {
        "symbol": "BTC-USDT",
        "side": "BUY",
        "type": "MARKET",
        "quantity": "0.001",
        "leverage": "1",
        "positionSide": "LONG",
        "timeInForce": "IOC"
    }
    
    # Создаем подпись для POST запроса
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
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
    
    # Разные варианты торговых endpoints
    trading_endpoints = [
        "/openApi/swap/v2/trade/order",
        "/openApi/swap/v1/trade/order",
        "/openApi/swap/trade/order",
        "/openApi/futures/trade/order",
        "/openApi/derivatives/trade/order",
        "/openApi/trade/order",
        "/api/swap/v2/trade/order",
        "/openApi/swap/v3/trade/order"
    ]
    
    working_trading = []
    
    for endpoint in trading_endpoints:
        print(f"\n--- Тестируем торговый endpoint {endpoint} ---")
        try:
            response = requests.post(
                f"https://open-api.bingx.com{endpoint}",
                json=params,
                headers=headers,
                timeout=10
            )
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                # Ищем успешные ответы или ошибки, отличные от "api is not exist"
                if (result.get('code') != 100400 and 
                    'msg' not in result.get('msg', '').lower() or
                    'insufficient' in result.get('msg', '').lower() or
                    'invalid' in result.get('msg', '').lower()):
                    print(f"✅ {endpoint} работает!")
                    working_trading.append(endpoint)
                else:
                    print(f"❌ {endpoint}: {result.get('msg', 'Unknown error')}")
            else:
                print(f"❌ {endpoint}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ {endpoint}: Ошибка - {e}")
    
    return working_trading

def test_public_endpoints():
    """Тестируем публичные endpoints"""
    print("\n🔍 Тестирование публичных endpoints...")
    
    # Публичные endpoints (без подписи)
    public_endpoints = [
        "/openApi/swap/v2/market/ticker",
        "/openApi/swap/v1/market/ticker",
        "/openApi/swap/market/ticker",
        "/openApi/market/ticker",
        "/api/swap/v2/market/ticker",
        "/openApi/swap/v2/market/price",
        "/openApi/swap/v1/market/price",
        "/openApi/swap/market/price"
    ]
    
    working_public = []
    
    for endpoint in public_endpoints:
        print(f"\n--- Тестируем публичный endpoint {endpoint} ---")
        try:
            response = requests.get(
                f"https://open-api.bingx.com{endpoint}",
                params={"symbol": "BTC-USDT"},
                timeout=10
            )
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            
            if response.status_code == 200:
                result = response.json()
                if (result.get('code') != 100400 and 
                    'data' in result or
                    'success' in str(result).lower()):
                    print(f"✅ {endpoint} работает!")
                    working_public.append(endpoint)
                else:
                    print(f"❌ {endpoint}: {result.get('msg', 'Unknown error')}")
            else:
                print(f"❌ {endpoint}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ {endpoint}: Ошибка - {e}")
    
    return working_public

if __name__ == "__main__":
    print("🔍 Поиск правильных endpoints для BingX API...")
    
    # Тестируем разные форматы endpoints для чтения
    working_read = test_different_endpoint_formats()
    if working_read:
        print(f"\n✅ Найдены рабочие endpoints для чтения: {working_read}")
    else:
        print("\n❌ Не найдены рабочие endpoints для чтения")
    
    # Тестируем торговые endpoints
    working_trading = test_trading_endpoints()
    if working_trading:
        print(f"\n✅ Найдены рабочие торговые endpoints: {working_trading}")
    else:
        print("\n❌ Не найдены рабочие торговые endpoints")
    
    # Тестируем публичные endpoints
    working_public = test_public_endpoints()
    if working_public:
        print(f"\n✅ Найдены рабочие публичные endpoints: {working_public}")
    else:
        print("\n❌ Не найдены рабочие публичные endpoints")
    
    print("\n🔍 Поиск endpoints завершен")