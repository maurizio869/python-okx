# Last modified (MSK): 2025-09-07 16:56:16 MSK — правка номер 17
import requests
import time
import hmac
import hashlib
import os
from pathlib import Path
from dotenv import load_dotenv
import json
import base64

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
TIMEOUT = 8

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
    params_base = {
        "symbol": "HBAR-USDT",
        "side": "SELL",
        "type": "MARKET",
        "quantity": "8",
        "positionSide": "SHORT"
    }
    _sync_server_time()
    last_ts = ''
    # 1) JSON body, sign query-string
    ts = _now_ms_str()
    last_ts = ts
    params = dict(params_base)
    params['timestamp'] = ts
    query = '&'.join([f"{k}={params[k]}" for k in sorted(params.keys())])
    import hashlib, hmac
    sig = hmac.new(api_secret.encode('utf-8'), query.encode('utf-8'), hashlib.sha256).hexdigest()
    headers = {
        'X-BX-APIKEY': api_key,
        'X-BX-SIGNATURE': sig,
        'X-BX-TIMESTAMP': ts,
        'Content-Type': 'application/json'
    }
    print(f"DBG ORDER QSTR sign JSON ct application/json ts={ts}")
    print(f"DBG ORDER query_string={query}")
    body_json = json.dumps(params, separators=(',', ':'))
    try:
        r = requests.post(f"{BASE_URL}/openApi/swap/v2/trade/order", headers=headers, data=body_json, timeout=TIMEOUT)
        try:
            data = r.json()
        except Exception:
            data = None
        code = data.get('code') if isinstance(data, dict) else None
        if r.status_code == 200 and code not in (100412, 100400):
            print(f"Timestamp: {ts}")
            print(f"Status Code: {r.status_code}")
            print(f"Response: {r.text}")
            return data if isinstance(data, dict) else {"raw": r.text}
        print(f"DBG ORDER reject JSON http={r.status_code} code={code} msg={str(data)[:200]}")
    except requests.exceptions.RequestException as e:
        print(f"WARN JSON post error: {e}")

    # 2) Form (x-www-form-urlencoded), sign same query, send form
    ts = _now_ms_str()
    last_ts = ts
    params = dict(params_base)
    params['timestamp'] = ts
    query = '&'.join([f"{k}={params[k]}" for k in sorted(params.keys())])
    sig = hmac.new(api_secret.encode('utf-8'), query.encode('utf-8'), hashlib.sha256).hexdigest()
    headers = {
        'X-BX-APIKEY': api_key,
        'X-BX-SIGNATURE': sig,
        'X-BX-TIMESTAMP': ts,
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    print(f"DBG ORDER QSTR sign FORM ct x-www-form-urlencoded ts={ts}")
    print(f"DBG ORDER query_string={query}")
    try:
        r = requests.post(f"{BASE_URL}/openApi/swap/v2/trade/order", headers=headers, data=query, timeout=TIMEOUT)
        try:
            data = r.json()
        except Exception:
            data = None
        code = data.get('code') if isinstance(data, dict) else None
        if r.status_code == 200 and code not in (100412, 100400):
            print(f"Timestamp: {ts}")
            print(f"Status Code: {r.status_code}")
            print(f"Response: {r.text}")
            return data if isinstance(data, dict) else {"raw": r.text}
        print(f"DBG ORDER reject FORM http={r.status_code} code={code} msg={str(data)[:200]}")
    except requests.exceptions.RequestException as e:
        print(f"WARN FORM post error: {e}")

    # 3) FORM with signature as param
    ts = _now_ms_str()
    last_ts = ts
    params = dict(params_base)
    params['timestamp'] = ts
    query = '&'.join([f"{k}={params[k]}" for k in sorted(params.keys())])
    sig = hmac.new(api_secret.encode('utf-8'), query.encode('utf-8'), hashlib.sha256).hexdigest()
    signed_query = query + f"&signature={sig}"
    headers = {
        'X-BX-APIKEY': api_key,
        'X-BX-SIGNATURE': sig,
        'X-BX-TIMESTAMP': ts,
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    print(f"DBG ORDER QSTR sign FORM+signature-param ts={ts}")
    print(f"DBG ORDER signed_query={signed_query}")
    try:
        r = requests.post(f"{BASE_URL}/openApi/swap/v2/trade/order", headers=headers, data=signed_query, timeout=TIMEOUT)
        try:
            data = r.json()
        except Exception:
            data = None
        print(f"Timestamp: {ts}")
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.text}")
        return data if isinstance(data, dict) else {"raw": r.text}
    except requests.exceptions.RequestException as e:
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
