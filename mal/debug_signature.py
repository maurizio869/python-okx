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

def debug_signature():
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    print(f"API Key: {api_key[:10]}...")
    print(f"API Secret: {api_secret[:10]}...")
    
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
    print(f"Timestamp: {timestamp}")
    
    # Метод 1: Текущий способ (с сортировкой)
    sorted_params = sorted(params.items())
    query_string_1 = "&".join([f"{k}={v}" for k, v in sorted_params])
    query_string_1 += f"&timestamp={timestamp}"
    
    signature_1 = hmac.new(
        api_secret.encode('utf-8'),
        query_string_1.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    print(f"\n=== Метод 1 (текущий) ===")
    print(f"Query String: {query_string_1}")
    print(f"Signature: {signature_1}")
    
    # Метод 2: Без сортировки (как было изначально)
    query_string_2 = "&".join([f"{k}={v}" for k, v in params.items()])
    query_string_2 += f"&timestamp={timestamp}"
    
    signature_2 = hmac.new(
        api_secret.encode('utf-8'),
        query_string_2.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    print(f"\n=== Метод 2 (без сортировки) ===")
    print(f"Query String: {query_string_2}")
    print(f"Signature: {signature_2}")
    
    # Метод 3: Попробуем другой формат (как в некоторых примерах)
    # Сначала создаем строку для подписи, затем добавляем timestamp
    params_with_timestamp = params.copy()
    params_with_timestamp['timestamp'] = timestamp
    
    sorted_params_3 = sorted(params_with_timestamp.items())
    query_string_3 = "&".join([f"{k}={v}" for k, v in sorted_params_3])
    
    signature_3 = hmac.new(
        api_secret.encode('utf-8'),
        query_string_3.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    print(f"\n=== Метод 3 (timestamp в параметрах) ===")
    print(f"Query String: {query_string_3}")
    print(f"Signature: {signature_3}")
    
    # Тестируем все три метода
    headers_template = {
        "X-BX-APIKEY": api_key,
        "X-BX-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    
    url = "https://open-api.bingx.com/openApi/swap/v2/trade/order"
    
    for i, signature in enumerate([signature_1, signature_2, signature_3], 1):
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
                    return signature
                else:
                    print(f"❌ Метод {i}: Null signature")
            else:
                print(f"❌ Метод {i}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Метод {i}: Ошибка - {e}")
    
    print("\n❌ Все методы не работают")
    return None

if __name__ == "__main__":
    print("🔍 Диагностика подписи BingX API...")
    result = debug_signature()
    if result:
        print(f"\n✅ Рабочая подпись: {result}")
    else:
        print("\n❌ Не удалось найти рабочую подпись")