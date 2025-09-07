# Last modified (MSK): 2025-09-07 16:31:38 MSK — правка номер 13
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

# Конфигурация
SIGN_MODE = 'TS_BODY'  # METHOD_PATH_TS_BODY | TS_PATH_BODY | TS_BODY
ALT_HEADER = '0'
BASE_URL = 'https://open-api.bingx.com'
TIMEOUT = int(os.getenv('TIMEOUT','20'))

_time_skew_ms = 0

def _now_ms_str() -> str:
    return str(int(time.time() * 1000) + int(_time_skew_ms))

def _sync_server_time() -> None:
    global _time_skew_ms
    candidates = [
        f"{BASE_URL}/api/v1/common/timestamp",
        "https://api.bingx.com/api/v1/common/timestamp",
    ]
    for url in candidates:
        try:
            t0 = int(time.time() * 1000)
            r = requests.get(url, timeout=5)
            t1 = int(time.time() * 1000)
            if r.status_code == 200:
                data = r.json()
                server_ts = int(data.get('timestamp') or data.get('data') or 0)
                if server_ts:
                    rtt = t1 - t0
                    _time_skew_ms = server_ts - (t0 + rtt // 2)
                    print(f"DBG time_sync url={url} server_ts={server_ts} rtt={rtt} skew_ms={_time_skew_ms}")
                    return
        except Exception:
            continue
    print("DBG time_sync failed, using local time")

def _build_signature(mode: str, method: str, path: str, timestamp: str, body_json: str) -> str:
    mode = mode.upper()
    method = method.upper()
    if mode == 'METHOD_PATH_TS_BODY':
        base = method + path + timestamp + body_json
    elif mode == 'TS_PATH_BODY':
        base = timestamp + path + body_json
    else:
        base = timestamp + body_json
    return hmac.new(api_secret.encode('utf-8'), base.encode('utf-8'), hashlib.sha256).hexdigest()

def _signed_string(mode: str, method: str, path: str, timestamp: str, body_json: str) -> str:
    mode = mode.upper()
    method = method.upper()
    if mode == 'METHOD_PATH_TS_BODY':
        return method + path + timestamp + body_json
    elif mode == 'TS_PATH_BODY':
        return timestamp + path + body_json
    else:
        return timestamp + body_json

def _signed_post(path: str, body: dict, timeout_sec: int | None = None, retries: int = 2) -> requests.Response:
    if timeout_sec is None:
        timeout_sec = TIMEOUT
    jb = json.dumps(body, separators=(',', ':'))
    last_exc = None
    for attempt in range(retries + 1):
        ts = _now_ms_str()
        sig = _build_signature(SIGN_MODE, 'POST', path, ts, jb)
        headers = {
            'X-BX-APIKEY': api_key,
            'X-BX-API-KEY': api_key,
            'X-BX-SIGNATURE': sig,
            'X-BX-TIMESTAMP': ts,
            'Content-Type': 'application/json'
        }
        print(f"DBG POST path={path} ts={ts} sign_mode={SIGN_MODE} attempt={attempt}")
        print("DBG POST string_signed=" + _signed_string(SIGN_MODE, 'POST', path, ts, jb))
        try:
            return requests.post(f"{BASE_URL}{path}", data=jb, headers=headers, timeout=timeout_sec)
        except requests.exceptions.RequestException as e:
            last_exc = e
            time.sleep(0.5 * (attempt + 1))
    raise last_exc

def _signed_get(path: str, timeout_sec: int | None = None, retries: int = 2) -> requests.Response:
    if timeout_sec is None:
        timeout_sec = TIMEOUT
    jb = ''
    last_exc = None
    for attempt in range(retries + 1):
        ts = _now_ms_str()
        sig = _build_signature(SIGN_MODE, 'GET', path, ts, jb)
        headers = {
            'X-BX-APIKEY': api_key,
            'X-BX-API-KEY': api_key,
            'X-BX-SIGNATURE': sig,
            'X-BX-TIMESTAMP': ts,
            'Content-Type': 'application/json'
        }
        print(f"DBG GET path={path} ts={ts} sign_mode={SIGN_MODE} attempt={attempt}")
        print("DBG GET string_signed=" + _signed_string(SIGN_MODE, 'GET', path, ts, jb))
        try:
            return requests.get(f"{BASE_URL}{path}", headers=headers, timeout=timeout_sec)
        except requests.exceptions.RequestException as e:
            last_exc = e
            time.sleep(0.5 * (attempt + 1))
    raise last_exc

def get_position_mode() -> str | None:
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

def ensure_position_mode(desired: str) -> None:
    try:
        cur = get_position_mode()
    except Exception as e:
        print(f"DBG get_position_mode error: {e}")
        cur = None
    if cur == desired:
        return
    variants = [
        ("/openApi/swap/v2/user/positionMode", {"positionMode": desired}),
        ("/openApi/swap/v1/positionSide/dual", {"dualSidePosition": desired.upper() == "HEDGE"})
    ]
    for path, body in variants:
        try:
            r = _signed_post(path, body)
            print(f"DBG ensure_mode POST {path} status={r.status_code} text={r.text[:200]}")
            if r.status_code == 200:
                time.sleep(0.2)
                try:
                    if get_position_mode() == desired:
                        return
                except Exception:
                    pass
        except Exception as e:
            print(f"DBG ensure_mode POST error: {e}")
    print(f"WARN: режим позиций не подтверждён как {desired}, продолжаем без блокировки")

def set_leverage(symbol: str, position_side: str, leverage: int | str) -> None:
    candidates = [
        ("/openApi/swap/v2/trade/leverage", {"symbol": symbol, "leverage": str(leverage)}),
        ("/openApi/swap/v2/user/leverage", {"symbol": symbol, "leverage": str(leverage)}),
        ("/openApi/swap/v2/user/leverage", {"symbol": symbol, "positionSide": position_side, "leverage": str(leverage)})
    ]
    last = None
    for path, body in candidates:
        try:
            r = _signed_post(path, body)
            if r.status_code == 200:
                return
            last = f"{r.status_code} {r.text}"
        except Exception as e:
            last = str(e)
    raise RuntimeError(f"Установка плеча не удалась: {last}")

def place_futures_order() -> dict:
    # Параметры ордера
    params = {
        "symbol": "HBAR-USDT",
        "side": "SELL",
        "type": "MARKET",
        "quantity": "3",
        "positionSide": "SHORT"
    }

    # Автоподбор режима подписи
    _sync_server_time()
    json_body = json.dumps(params, separators=(',', ':'))
    modes_order = ['TS_BODY', 'TS_PATH_BODY', 'METHOD_PATH_TS_BODY']

    response = None
    last_exc = None
    last_timestamp = ''

    for mode in modes_order:
        ts = _now_ms_str()
        last_timestamp = ts
        sig = _build_signature(mode, 'POST', '/openApi/swap/v2/trade/order', ts, json_body)
        headers = {
            'X-BX-APIKEY': api_key,
            'X-BX-API-KEY': api_key,
            'X-BX-SIGNATURE': sig,
            'X-BX-TIMESTAMP': ts,
            'Content-Type': 'application/json'
        }
        print(f"DBG ORDER try mode={mode} ts={ts}")
        print("DBG ORDER string_signed=" + _signed_string(mode, 'POST', '/openApi/swap/v2/trade/order', ts, json_body))
        for attempt in range(3):
            try:
                response = requests.post(f"{BASE_URL}/openApi/swap/v2/trade/order", data=json_body, headers=headers, timeout=TIMEOUT)
                break
            except requests.exceptions.RequestException as e:
                last_exc = e
                print(f"WARN order post attempt={attempt} error={e}")
                time.sleep(0.7 * (attempt + 1))
        if response is None:
            continue
        try:
            data = response.json()
        except Exception:
            data = None
        code = data.get('code') if isinstance(data, dict) else None
        if response.status_code == 200 and code not in (100412, 100400):
            print(f"DBG ORDER accepted mode={mode} http={response.status_code} code={code}")
            break
        else:
            msg = (data.get('msg') if isinstance(data, dict) else str(response.text))
            print(f"DBG ORDER reject mode={mode} http={response.status_code} code={code} msg={str(msg)[:300]}")

    print(f"Timestamp: {last_timestamp}")
    if response is not None:
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        try:
            return response.json()
        except Exception:
            return {"raw": response.text}
    if last_exc:
        raise last_exc
    return {"error": "No response"}

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
