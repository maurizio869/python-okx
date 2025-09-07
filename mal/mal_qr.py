# Last modified (MSK): 2025-09-07 17:01:59 MSK — правка номер 18
import requests
import time
import hmac
import hashlib
import os
from pathlib import Path
from dotenv import load_dotenv
import json

# Загружаем .env из той же папки (api_key1, api_secret1)
dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path)

# Ключи API
api_key = os.getenv('api_key1', '1')
api_secret = os.getenv('api_secret1', '1')

# Константы
BASE_URL = 'https://open-api.bingx.com'
TIMEOUT = 8  # секунды

# ------------------------- Вспомогательные функции -------------------------

def _now_ms_str() -> str:
    """Текущий timestamp в мс (строка)."""
    return str(int(time.time() * 1000))

def _sync_server_time() -> None:
    """Best-effort синхронизация: не обязательна, но помогает уменьшить дрейф.
    Оставлена как no-op/логирующая, чтобы не усложнять код.
    """
    try:
        r = requests.get(f"{BASE_URL}/api/v1/common/timestamp", timeout=3)
        if r.status_code == 200:
            data = r.json()
            server_ts = int(data.get('timestamp') or data.get('data') or 0)
            if server_ts:
                print(f"DBG server_ts={server_ts}")
    except Exception:
        pass

# --------------------------- Рабочая схема подписи --------------------------
# Для эндпоинта создания ордера работает схема:
# 1) Строим отсортированный по ключам query-string (включая timestamp)
# 2) Подписываем его HMAC-SHA256(secret, query_string) -> hex
# 3) Отправляем как application/x-www-form-urlencoded:
#    body = query_string + '&signature=' + signature
# 4) В заголовках указываем X-BX-APIKEY, X-BX-SIGNATURE, X-BX-TIMESTAMP


def _build_query_string(params: dict) -> str:
    """Возвращает отсортированный query-string вида k=v&k2=v2..."""
    return '&'.join([f"{k}={params[k]}" for k in sorted(params.keys())])


def _sign_query_string(query_string: str) -> str:
    """HMAC-SHA256 по query-string, hex."""
    return hmac.new(api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()


# ------------------------------ Леверидж (ok) -------------------------------
# Установка плеча у нас ранее проходила успешно через JSON+подпись HMAC(secret, timestamp+body)
# Оставляем краткую реализацию, так как она рабочая.

def _sign_ts_plus_body(ts: str, body_json: str) -> str:
    return hmac.new(api_secret.encode('utf-8'), (ts + body_json).encode('utf-8'), hashlib.sha256).hexdigest()


def set_leverage(symbol: str, leverage: int | str) -> None:
    body = {"symbol": symbol, "leverage": str(leverage)}
    ts = _now_ms_str()
    body_json = json.dumps(body, separators=(',', ':'))
    sig = _sign_ts_plus_body(ts, body_json)
    headers = {
        'X-BX-APIKEY': api_key,
        'X-BX-SIGNATURE': sig,
        'X-BX-TIMESTAMP': ts,
        'Content-Type': 'application/json'
    }
    try:
        r = requests.post(f"{BASE_URL}/openApi/swap/v2/trade/leverage", headers=headers, data=body_json, timeout=TIMEOUT)
        if r.status_code == 200:
            print('Плечо установлено: 10x')
        else:
            print(f"WARN leverage status={r.status_code} resp={r.text[:200]}")
    except requests.exceptions.RequestException as e:
        print(f"WARN leverage request error: {e}")


# ---------------------------- Создание ордера -------------------------------

def place_futures_order() -> dict:
    # Параметры ордера. Количество >= 8 для HBAR согласно ответу биржи.
    params_base = {
        'symbol': 'HBAR-USDT',
        'side': 'SELL',
        'type': 'MARKET',
        'quantity': '8',
        'positionSide': 'SHORT'
    }

    _sync_server_time()

    # Добавляем timestamp, строим подпись по query-string и отправляем форму с signature
    ts = _now_ms_str()
    params = dict(params_base)
    params['timestamp'] = ts

    query = _build_query_string(params)
    signature = _sign_query_string(query)
    signed_form = query + f"&signature={signature}"

    headers = {
        'X-BX-APIKEY': api_key,
        'X-BX-SIGNATURE': signature,
        'X-BX-TIMESTAMP': ts,
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    try:
        r = requests.post(f"{BASE_URL}/openApi/swap/v2/trade/order", headers=headers, data=signed_form, timeout=TIMEOUT)
        print(f"Timestamp: {ts}")
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.text}")
        try:
            return r.json()
        except Exception:
            return {'raw': r.text}
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}


# ------------------------------- Тест-запуск --------------------------------
if __name__ == '__main__':
    print('Запуск теста API...')
    # Плечо 10x (опционально; у нас это работало)
    set_leverage('HBAR-USDT', 10)
    # Размещаем ордер
    result = place_futures_order()
    print(f"Результат: {result}")
