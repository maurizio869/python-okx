# Last modified (MSK): 2025-09-06 09:42:27 MSK — правка номер 5
import requests
import time
import hmac
import hashlib
import os
from pathlib import Path
from dotenv import load_dotenv
# путь к .env: та же папка, где лежит текущий файл
dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path)           # <-- читаем файл

def place_futures_order():
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')

    # Параметры ордера
    params = {
        "symbol": "HBAR-USDT",
        "side": "SELL",
        "type": "MARKET", 
        "quantity": "3",
        "positionSide": "SHORT"
    }

    # Создание подписи согласно BingX API документации
    timestamp = str(int(time.time() * 1000))
    
    import json
    json_body = json.dumps(params, separators=(',', ':'))
    signature = hmac.new(
        api_secret.encode('utf-8'),
        (timestamp + json_body).encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    print(f"Timestamp: {timestamp}")

    # Заголовки (только ASCII символы)
    headers = {
        "X-BX-APIKEY": api_key,
        "X-BX-SIGNATURE": signature,
        "X-BX-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }

    # вспомогательные функции для режима позиций и плеча
    def _signed_post(path, body: dict):
        ts = str(int(time.time() * 1000))
        jb = json.dumps(body, separators=(',', ':'))
        sig = hmac.new(api_secret.encode('utf-8'), (ts + jb).encode('utf-8'), hashlib.sha256).hexdigest()
        hdr = {
            "X-BX-APIKEY": api_key,
            "X-BX-SIGNATURE": sig,
            "X-BX-TIMESTAMP": ts,
            "Content-Type": "application/json"
        }
        return requests.post(f"https://open-api.bingx.com{path}", data=jb, headers=hdr, timeout=10)

    def get_position_mode():
        ts = str(int(time.time() * 1000))
        sig = hmac.new(api_secret.encode('utf-8'), (ts + '').encode('utf-8'), hashlib.sha256).hexdigest()
        hdr = {
            "X-BX-APIKEY": api_key,
            "X-BX-SIGNATURE": sig,
            "X-BX-TIMESTAMP": ts,
            "Content-Type": "application/json"
        }
        for path in ("/openApi/swap/v2/user/positionMode", "/openApi/swap/v1/positionSide/dual"):
            try:
                r = requests.get(f"https://open-api.bingx.com{path}", headers=hdr, timeout=10)
                if r.status_code == 200:
                    try:
                        data = r.json().get('data', {})
                        if 'positionMode' in data:
                            return str(data['positionMode']).upper()
                        if 'dualSidePosition' in data:
                            return 'HEDGE' if bool(data['dualSidePosition']) else 'ONE_WAY'
                    except Exception:
                        pass
            except Exception:
                continue
        return None

    def ensure_position_mode(desired: str):
        cur = get_position_mode()
        if cur == desired:
            return
        variants = [
            ("/openApi/swap/v2/user/positionMode", {"positionMode": desired}),
            ("/openApi/swap/v1/positionSide/dual", {"dualSidePosition": desired.upper() == "HEDGE"})
        ]
        for path, body in variants:
            r = _signed_post(path, body)
            if r.status_code == 200:
                time.sleep(0.2)
                if get_position_mode() == desired:
                    return
        raise RuntimeError(f"Не удалось установить режим позиций: {desired}; текущий: {cur}")

    def set_leverage(symbol: str, position_side: str, leverage: int | str):
        candidates = [
            ("/openApi/swap/v2/trade/leverage", {"symbol": symbol, "leverage": str(leverage)}),
            ("/openApi/swap/v2/user/leverage", {"symbol": symbol, "leverage": str(leverage)}),
            ("/openApi/swap/v2/user/leverage", {"symbol": symbol, "positionSide": position_side, "leverage": str(leverage)})
        ]
        last = None
        for path, body in candidates:
            r = _signed_post(path, body)
            if r.status_code == 200:
                return
            last = f"{r.status_code} {r.text}"
        raise RuntimeError(f"Установка плеча не удалась: {last}")

    try:
        # Отправка запроса
        response = requests.post(
            "https://open-api.bingx.com/openApi/swap/v2/trade/order",
            json=params,
            headers=headers,
            timeout=10
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"General Error: {e}")
        return {"error": str(e)}

# Тестируем функцию
if __name__ == "__main__":
    print("Запуск теста API...")
    try:
        ensure_position_mode('HEDGE')
        print('Режим позиций установлен/подтверждён: HEDGE')
    except Exception as e:
        print(f'Не удалось установить режим позиций: {e}')
    try:
        set_leverage('HBAR-USDT', 'SHORT', 10)
        print('Плечо установлено: 10x')
    except Exception as e:
        print(f'Не удалось установить плечо: {e}')
    result = place_futures_order()
    print(f"Результат: {result}")