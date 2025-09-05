import requests
import time
import hmac
import hashlib

def test_api_connection():
    """Тест подключения к API без реальных ключей"""
    
    # Тестовые данные
    api_key = "test_key"
    api_secret = "test_secret"
    
    # Параметры ордера
    params = {
        "symbol": "HBAR-USDT",
        "side": "BUY",
        "type": "MARKET", 
        "quantity": "100",
        "leverage": "5",
        "positionSide": "LONG",
        "timeInForce": "IOC"
    }
    
    # Создание подписи
    timestamp = str(int(time.time() * 1000))
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    query_string += f"&timestamp={timestamp}"
    
    print(f"Query String: {query_string}")
    
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    print(f"Signature: {signature}")
    
    # Заголовки
    headers = {
        "X-BX-APIKEY": api_key,
        "X-BX-SIGNATURE": signature,
        "X-BX-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    
    print(f"Headers: {headers}")
    
    try:
        # Тест подключения
        response = requests.post(
            "https://open-api.bingx.com/openApi/swap/v2/trade/order",
            json=params,
            headers=headers,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Text: {response.text}")
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"General Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("Тест API подключения...")
    result = test_api_connection()
    print(f"Результат: {result}")