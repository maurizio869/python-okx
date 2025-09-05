import requests
import time
import hmac
import hashlib

def place_futures_order():
    api_key = "ваш_api_ключ"
    api_secret = "ваш_api_секрет"
    
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
    
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # Заголовки
    headers = {
        "X-BX-APIKEY": api_key,
        "X-BX-SIGNATURE": signature,
        "X-BX-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    
    # Отправка запроса
    response = requests.post(
        "https://open-api.bingx.com/openApi/swap/v2/trade/order",
        json=params,
        headers=headers
    )
    
    return response.json()

a = place_futures_order()