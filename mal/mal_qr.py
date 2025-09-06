# Last modified (MSK): 2025-09-06 10:47:12 MSK — правка номер 8
import requests
import time
import hmac
import hashlib
import os
from pathlib import Path
from dotenv import load_dotenv
import json
# путь к .env: та же папка, где лежит текущий файл
dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path)           # <-- читаем файл

# Глобальные ключи для подписи и заголовков
api_key = os.getenv('api_key1', '1')
api_secret = os.getenv('api_secret1', '1')

# Переключаемые режимы подписи и заголовков
SIGN_MODE = os.getenv('BINGX_SIGN_MODE', 'METHOD_PATH_TS_BODY')  # METHOD_PATH_TS_BODY | TS_PATH_BODY | TS_BODY
ALT_HEADER = os.getenv('BINGX_ALT_HEADER', '0')  # если '1', добавлять X-BX-API-KEY

def _build_signature(method: str, path: str, timestamp: str, body_json: str) -> str:
    method = method.upper()
    if SIGN_MODE == 'METHOD_PATH_TS_BODY':
        base = method + path + timestamp + body_json
    elif SIGN_MODE == 'TS_PATH_BODY':
        base = timestamp + path + body_json
    else:  # TS_BODY (по умолчанию раньше было timestamp+json)
        base = timestamp + body_json
    return hmac.new(api_secret.encode('utf-8'), base.encode('utf-8'), hashlib.sha256).hexdigest()

def _signed_post(path, body: dict):
    ts = str(int(time.time() * 1000))
    jb = json.dumps(body, separators=(',', ':'))
    sig = _build_signature('POST', path, ts, jb)
    hdr = {
        "X-BX-APIKEY": api_key,
        "X-BX-SIGNATURE": sig,
        "X-BX-TIMESTAMP": ts,
        "Content-Type": "application/json"
    }
    if ALT_HEADER == '1':
        hdr["X-BX-API-KEY"] = api_key
    # Отладка
    print(f"DBG POST path={path} ts={ts} sign_mode={SIGN_MODE} alt_header={ALT_HEADER}")
    print(f"DBG POST string_signed={('POST'+path+ts+jb) if SIGN_MODE=='METHOD_PATH_TS_BODY' else (ts+path+jb) if SIGN_MODE=='TS_PATH_BODY' else (ts+jb)}")
    return requests.post(f"https://open-api.bingx.com{path}", data=jb, headers=hdr, timeout=10)

def _signed_get(path: str):
    ts = str(int(time.time() * 1000))
    jb = ''
    sig = _build_signature('GET', path, ts, jb)
    hdr = {
        "X-BX-APIKEY": api_key,
        "X-BX-SIGNATURE": sig,
        "X-BX-TIMESTAMP": ts,
        "Content-Type": "application/json"
    }
    if ALT_HEADER == '1':
        hdr["X-BX-API-KEY"] = api_key
    print(f"DBG GET path={path} ts={ts} sign_mode={SIGN_MODE} alt_header={ALT_HEADER}")
    print(f"DBG GET string_signed={('GET'+path+ts+jb) if SIGN_MODE=='METHOD_PATH_TS_BODY' else (ts+path+jb) if SIGN_MODE=='TS_PATH_BODY' else (ts+jb)}")
    return requests.get(f"https://open-api.bingx.com{path}", headers=hdr, timeout=10)

def get_position_mode():
    for path in ("/openApi/swap/v2/user/positionMode", "/openApi/swap/v1/positionSide/dual"):
        try:
            r = _signed_get(path)
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

def place_futures_order():

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
    
    json_body = json.dumps(params, separators=(',', ':'))
    signature = _build_signature('POST', '/openApi/swap/v2/trade/order', timestamp, json_body)
    print(f"Timestamp: {timestamp}")

    # Заголовки (только ASCII символы)
    headers = {
        "X-BX-APIKEY": api_key,
        "X-BX-SIGNATURE": signature,
        "X-BX-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }


    try:
        # Отправка запроса
        response = requests.post(
            "https://open-api.bingx.com/openApi/swap/v2/trade/order",
            data=json_body,
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