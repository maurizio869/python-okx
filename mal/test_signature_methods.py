# Last modified (MSK): 2025-09-06 07:50 — правка номер 1
import requests
import time
import hmac
import hashlib
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# путь к .env: та же папка, где лежит текущий файл
dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path)

def test_signature_methods():
    """Тестируем разные методы генерации подписи"""
    print("🔍 Тестирование методов генерации подписи...")
    
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
    
    timestamp = str(int(time.time() * 1000))
    url = "https://open-api.bingx.com/openApi/swap/v2/trade/order"
    
    # Метод 1: Текущий (timestamp + JSON body)
    print("\n=== Метод 1: timestamp + JSON body ===")
    json_body = json.dumps(params, separators=(',', ':'))
    signature_string = timestamp + json_body
    signature1 = hmac.new(
        api_secret.encode('utf-8'),
        signature_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    print(f"JSON Body: {json_body}")
    print(f"Signature String: {signature_string}")
    print(f"Signature: {signature1}")
    
    # Метод 2: Только JSON body
    print("\n=== Метод 2: только JSON body ===")
    signature2 = hmac.new(
        api_secret.encode('utf-8'),
        json_body.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    print(f"Signature: {signature2}")
    
    # Метод 3: Query string (старый способ)
    print("\n=== Метод 3: Query string ===")
    sorted_params = sorted(params.items())
    query_string = "&".join([f"{k}={v}" for k, v in sorted_params])
    query_string += f"&timestamp={timestamp}"
    signature3 = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    print(f"Query String: {query_string}")
    print(f"Signature: {signature3}")
    
    # Метод 4: JSON body + timestamp (обратный порядок)
    print("\n=== Метод 4: JSON body + timestamp ===")
    signature_string4 = json_body + timestamp
    signature4 = hmac.new(
        api_secret.encode('utf-8'),
        signature_string4.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    print(f"Signature String: {signature_string4}")
    print(f"Signature: {signature4}")
    
    # Метод 5: С параметрами в URL
    print("\n=== Метод 5: Параметры в URL ===")
    params_with_timestamp = params.copy()
    params_with_timestamp['timestamp'] = timestamp
    sorted_params5 = sorted(params_with_timestamp.items())
    query_string5 = "&".join([f"{k}={v}" for k, v in sorted_params5])
    signature5 = hmac.new(
        api_secret.encode('utf-8'),
        query_string5.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    print(f"Query String: {query_string5}")
    print(f"Signature: {signature5}")
    
    # Тестируем все методы
    headers_template = {
        "X-BX-APIKEY": api_key,
        "X-BX-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    
    signatures = [signature1, signature2, signature3, signature4, signature5]
    
    for i, signature in enumerate(signatures, 1):
        print(f"\n--- Тестируем метод {i} ---")
        headers = headers_template.copy()
        headers["X-BX-SIGNATURE"] = signature
        
        try:
            response = requests.post(
                url,
                json=params,
                headers=headers,
                timeout=10
            )
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') != 100412:  # Не "Null signature"
                    print(f"✅ Метод {i} работает!")
                    return signature, i
                else:
                    print(f"❌ Метод {i}: Null signature")
            else:
                print(f"❌ Метод {i}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Метод {i}: Ошибка - {e}")
    
    print("\n❌ Все методы не работают")
    return None, None

def test_different_endpoints():
    """Тестируем разные endpoints"""
    print("\n🔍 Тестирование разных endpoints...")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    params = {
        "symbol": "HBAR-USDT",
        "side": "SELL",
        "type": "MARKET", 
        "quantity": "3",
        "leverage": "10",
        "positionSide": "SHORT",
        "timeInForce": "IOC"
    }
    
    timestamp = str(int(time.time() * 1000))
    json_body = json.dumps(params, separators=(',', ':'))
    signature_string = timestamp + json_body
    signature = hmac.new(
        api_secret.encode('utf-8'),
        signature_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "X-BX-APIKEY": api_key,
        "X-BX-SIGNATURE": signature,
        "X-BX-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    
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
    # Тестируем методы генерации подписи
    working_signature, method_num = test_signature_methods()
    
    if working_signature:
        print(f"\n🎉 Найден рабочий метод {method_num}!")
        print(f"Рабочая подпись: {working_signature}")
    else:
        print("\n❌ Рабочий метод не найден")
    
    # Тестируем разные endpoints
    test_different_endpoints()