# Last modified (MSK): 2025-09-05 15:30 — правка номер 1
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

def test_public_endpoint():
    """Тестируем публичный endpoint без авторизации"""
    print("=== Тестируем публичный endpoint ===")
    try:
        response = requests.get("https://open-api.bingx.com/openApi/swap/v1/server/time", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

def test_account_info():
    """Тестируем получение информации об аккаунте"""
    print("\n=== Тестируем получение информации об аккаунте ===")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    timestamp = str(int(time.time() * 1000))
    
    # Простой запрос для получения информации об аккаунте
    params = {}
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
        return response.status_code == 200
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

def test_different_endpoints():
    """Тестируем разные endpoints для торговли"""
    print("\n=== Тестируем разные endpoints ===")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    timestamp = str(int(time.time() * 1000))
    
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
    
    # Сортируем параметры
    sorted_params = sorted(params.items())
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
    
    # Список endpoints для тестирования
    endpoints = [
        "https://open-api.bingx.com/openApi/swap/v2/trade/order",
        "https://open-api.bingx.com/openApi/swap/v1/trade/order",
        "https://open-api.bingx.com/openApi/spot/v1/trade/order",
        "https://open-api.bingx.com/openApi/spot/v2/trade/order"
    ]
    
    for endpoint in endpoints:
        print(f"\n--- Тестируем {endpoint} ---")
        try:
            response = requests.post(
                endpoint,
                json=params,
                headers=headers,
                timeout=10
            )
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') != 100412:  # Не "Null signature"
                    print(f"✅ Endpoint работает!")
                else:
                    print(f"❌ Null signature")
            else:
                print(f"❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    print("🔍 Тестирование endpoints BingX API...")
    
    # Тестируем публичный endpoint
    if test_public_endpoint():
        print("✅ Публичный endpoint работает")
    else:
        print("❌ Публичный endpoint не работает")
    
    # Тестируем получение информации об аккаунте
    if test_account_info():
        print("✅ Получение информации об аккаунте работает")
    else:
        print("❌ Получение информации об аккаунте не работает")
    
    # Тестируем разные endpoints для торговли
    test_different_endpoints()