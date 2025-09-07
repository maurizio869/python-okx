import os
import time
import hmac
import json
import hashlib
from pathlib import Path

import requests
from dotenv import load_dotenv


# Last modified (MSK): 2025-09-06 08:20 — правка номер 4

# Загрузка переменных окружения из .env, расположенного рядом с этим файлом
dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path)


API_KEY = os.getenv('api_key1', '')
API_SECRET = os.getenv('api_secret1', '')

# Базовый URL BingX OpenAPI
BASE_URL = 'https://open-api.bingx.com'


def _now_ms() -> str:
    """Возвращает текущий timestamp в миллисекундах как строку."""
    return str(int(time.time() * 1000))


def _sign_json(timestamp: str, json_body: str) -> str:
    """Создает подпись HMAC-SHA256 для JSON POST/authorized GET по правилу: HMAC(secret, timestamp + body)."""
    return hmac.new(
        API_SECRET.encode('utf-8'),
        (timestamp + json_body).encode('utf-8'),
        hashlib.sha256,
    ).hexdigest()


def _headers(timestamp: str, signature: str) -> dict:
    return {
        'X-BX-APIKEY': API_KEY,
        'X-BX-TIMESTAMP': timestamp,
        'X-BX-SIGNATURE': signature,
        'Content-Type': 'application/json',
    }


def _request(method: str, path: str, json_payload: dict | None = None, timeout: int = 10) -> requests.Response:
    """Отправляет подписанный запрос на BingX. Для GET без тела используем пустой JSON при расчете подписи."""
    if json_payload is None:
        json_str = ''
    else:
        # Важно: без пробелов для корректной подписи
        json_str = json.dumps(json_payload, separators=(',', ':'))

    timestamp = _now_ms()
    signature = _sign_json(timestamp, json_str)
    url = BASE_URL + path
    headers = _headers(timestamp, signature)

    if method.upper() == 'GET':
        # Авторизованный GET без query: тело пустое, подпись по timestamp + ''
        resp = requests.get(url, headers=headers, timeout=timeout)
    elif method.upper() == 'POST':
        resp = requests.post(url, data=json_str, headers=headers, timeout=timeout)
    else:
        raise ValueError(f'Unsupported method: {method}')

    return resp


def get_server_time() -> dict:
    """Возвращает серверное время, если эндпоинт доступен. Необязательная вспомогательная функция."""
    # На некоторых версиях API доступен общий эндпоинт времени; оставим как best-effort.
    try:
        resp = requests.get('https://api.bingx.com/api/v1/common/timestamp', timeout=5)
        return {'ok': resp.status_code == 200, 'status': resp.status_code, 'text': resp.text}
    except Exception as exc:
        return {'ok': False, 'error': str(exc)}


def get_position_mode() -> str | None:
    """Возвращает текущий режим позиций: 'HEDGE' или 'ONE_WAY' если поддерживается. Иначе None."""
    # Официальные пути могут отличаться у разных ревизий API; используем наиболее вероятный v2 user endpoint
    path_candidates = [
        '/openApi/swap/v2/user/positionMode',
        '/openApi/swap/v1/positionSide/dual',  # альтернативный путь у старых ревизий
    ]

    for path in path_candidates:
        try:
            resp = _request('GET', path)
            if resp.status_code == 200:
                data = {}
                try:
                    data = resp.json()
                except Exception:
                    pass
                # Пытаемся извлечь значение режима из разных возможных форматов
                # Примеры: { data: { positionMode: 'HEDGE' } } или { data: { dualSidePosition: true } }
                d = data.get('data') if isinstance(data, dict) else None
                if isinstance(d, dict):
                    if 'positionMode' in d:
                        return str(d.get('positionMode')).upper()
                    if 'dualSidePosition' in d:
                        return 'HEDGE' if bool(d.get('dualSidePosition')) else 'ONE_WAY'
        except Exception:
            continue

    return None


def ensure_position_mode(desired_mode: str) -> None:
    """Проверяет текущий режим позиций и, если необходимо, пытается переключить его.

    desired_mode: 'HEDGE' или 'ONE_WAY'
    """
    current = get_position_mode()
    if current == desired_mode:
        return

    # Попытка переключить режим. Важно: биржа может требовать отсутствие позиций/ордеров для переключения.
    payload_variants = [
        ('/openApi/swap/v2/user/positionMode', {'positionMode': desired_mode}),
        ('/openApi/swap/v1/positionSide/dual', {'dualSidePosition': desired_mode.upper() == 'HEDGE'}),
    ]

    for path, body in payload_variants:
        resp = _request('POST', path, body)
        # Успехи у BingX обычно содержат код 200/0 в json; проверим HTTP-код и попытаемся перечитать режим
        if resp.status_code == 200:
            # Подтверждаем
            time.sleep(0.2)
            if get_position_mode() == desired_mode:
                return

    raise RuntimeError(f'Не удалось установить режим позиций: {desired_mode}. Текущий: {current}')


def set_leverage(symbol: str, position_side: str, leverage: int | str) -> None:
    """Устанавливает кредитное плечо для указанного символа и стороны позиции, если это необходимо API.

    Некоторые ревизии API требуют указания стороны позиции, другие — нет. Попробуем оба варианта.
    """
    body_variants = [
        # v2 trade leverage (часто без positionSide)
        ('/openApi/swap/v2/trade/leverage', {'symbol': symbol, 'leverage': str(leverage)}),
        # альтернативный пользовательский путь и/или с позицией
        ('/openApi/swap/v2/user/leverage', {'symbol': symbol, 'leverage': str(leverage)}),
        ('/openApi/swap/v2/user/leverage', {'symbol': symbol, 'positionSide': position_side, 'leverage': str(leverage)}),
    ]

    last_error = None
    for path, body in body_variants:
        try:
            resp = _request('POST', path, body)
            if resp.status_code == 200:
                return
            last_error = f'{resp.status_code} {resp.text}'
        except Exception as exc:
            last_error = str(exc)

    raise RuntimeError(f'Установка плеча не удалась: {last_error}')


def place_futures_order(symbol: str,
                        side: str,
                        order_type: str,
                        quantity: str | int | float,
                        position_side: str = 'BOTH',
                        price: str | int | float | None = None,
                        reduce_only: bool | None = None,
                        client_order_id: str | None = None) -> dict:
    """Размещает ордер Perpetual Swap V2.

    Примечания:
    - leverage и timeInForce не передаем тут; плечо настраивается отдельно через set_leverage.
    - Для MARKET ордеров price не нужен.
    - position_side: 'BOTH' в one-way, 'LONG'/'SHORT' в hedge.
    """
    # Проверяем режим позиций и соответствие position_side
    mode = get_position_mode()
    if mode == 'HEDGE':
        if position_side not in ('LONG', 'SHORT'):
            raise ValueError("В режиме HEDGE параметр position_side должен быть 'LONG' или 'SHORT'")
    elif mode == 'ONE_WAY':
        position_side = 'BOTH'
    else:
        # Если режим не удалось определить, не будем навязывать правило, но предупредим в логе
        pass

    params: dict = {
        'symbol': symbol,
        'side': side,
        'type': order_type,
        'quantity': str(quantity),
        'positionSide': position_side,
    }
    if price is not None:
        params['price'] = str(price)
    if reduce_only is not None:
        params['reduceOnly'] = bool(reduce_only)
    if client_order_id is not None:
        params['clientOrderId'] = str(client_order_id)

    resp = _request('POST', '/openApi/swap/v2/trade/order', params)
    try:
        content = resp.json()
    except Exception:
        content = {'raw': resp.text}

    # Быстрая диагностическая печать
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")

    if resp.status_code >= 400:
        raise RuntimeError(f'API Error {resp.status_code}: {resp.text}')

    return content if isinstance(content, dict) else {'data': content}


if __name__ == '__main__':
    print('Запуск теста API...')

    # Опционально: привести режим к HEDGE, чтобы использовать SHORT/LONG отдельно
    try:
        ensure_position_mode('HEDGE')
        print('Режим позиций: HEDGE')
    except Exception as e:
        print(f'Не удалось установить/подтвердить режим HEDGE: {e}')

    # Попытка установить плечо 10x
    try:
        set_leverage('HBAR-USDT', 'SHORT', 10)
        print('Плечо установлено: 10x')
    except Exception as e:
        print(f'Не удалось установить плечо: {e}')

    # Тестовый MARKET ордер на продажу 3 шт в SHORT
    try:
        result = place_futures_order(
            symbol='HBAR-USDT',
            side='SELL',
            order_type='MARKET',
            quantity='3',
            position_side='SHORT',
        )
        print(f'Результат: {result}')
    except Exception as e:
        print(f'Ошибка при размещении ордера: {e}')

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
    result = place_futures_order()
    print(f"Результат: {result}")