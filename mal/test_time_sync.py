# Last modified (MSK): 2025-09-06 08:15 — правка номер 1
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

def test_time_sync():
    """Тестируем синхронизацию времени"""
    print("🔍 Тестирование синхронизации времени...")
    
    # Получаем время сервера BingX
    try:
        response = requests.get("https://open-api.bingx.com/openApi/swap/v1/server/time", timeout=10)
        if response.status_code == 200:
            server_time = response.json().get('data', {}).get('serverTime')
            if server_time:
                print(f"Время сервера BingX: {server_time}")
                print(f"Время сервера (человекочитаемое): {time.ctime(server_time/1000)}")
                
                # Получаем локальное время
                local_time = int(time.time() * 1000)
                print(f"Локальное время: {local_time}")
                print(f"Локальное время (человекочитаемое): {time.ctime(local_time/1000)}")
                
                # Вычисляем разность
                time_diff = abs(server_time - local_time)
                print(f"Разность времени: {time_diff} мс")
                
                if time_diff > 5000:  # Больше 5 секунд
                    print("❌ Время не синхронизировано! Разность больше 5 секунд")
                    return False
                else:
                    print("✅ Время синхронизировано")
                    return True
            else:
                print("❌ Не удалось получить время сервера")
                return False
        else:
            print(f"❌ Ошибка получения времени сервера: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_with_server_time():
    """Тестируем подпись с временем сервера"""
    print("\n🔍 Тестирование подписи с временем сервера...")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    # Получаем время сервера
    try:
        response = requests.get("https://open-api.bingx.com/openApi/swap/v1/server/time", timeout=10)
        if response.status_code == 200:
            server_time = response.json().get('data', {}).get('serverTime')
            if server_time:
                timestamp = str(server_time)
                print(f"Используем время сервера: {timestamp}")
                
                # Создаем подпись с временем сервера
                query_string = f"timestamp={timestamp}"
                signature = hmac.new(
                    api_secret.encode('utf-8'),
                    query_string.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                
                headers = {
                    "X-BX-APIKEY": api_key,
                    "X-BX-SIGNATURE": signature,
                    "X-BX-TIMESTAMP": timestamp,
                    "Content-Type": "application/json"
                }
                
                # Тестируем запрос
                try:
                    response = requests.get(
                        "https://open-api.bingx.com/openApi/swap/v2/user/getBalance",
                        params={"timestamp": timestamp},
                        headers=headers,
                        timeout=10
                    )
                    print(f"Status Code: {response.status_code}")
                    print(f"Response: {response.text}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('code') != 100412:  # Не "Null signature"
                            print("✅ Подпись с временем сервера работает!")
                            return True
                        else:
                            print("❌ Подпись с временем сервера: Null signature")
                    else:
                        print(f"❌ HTTP {response.status_code}")
                        
                except Exception as e:
                    print(f"❌ Ошибка запроса: {e}")
            else:
                print("❌ Не удалось получить время сервера")
        else:
            print(f"❌ Ошибка получения времени сервера: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    return False

def test_different_base_urls():
    """Тестируем разные базовые URL"""
    print("\n🔍 Тестирование разных базовых URL...")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    timestamp = str(int(time.time() * 1000))
    query_string = f"timestamp={timestamp}"
    
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "X-BX-APIKEY": api_key,
        "X-BX-SIGNATURE": signature,
        "X-BX-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    
    base_urls = [
        "https://open-api.bingx.com",
        "https://api.bingx.com",
        "https://open-api.bingx.com/openApi",
        "https://api.bingx.com/openApi"
    ]
    
    for base_url in base_urls:
        print(f"\n--- Тестируем {base_url} ---")
        try:
            response = requests.get(
                f"{base_url}/swap/v2/user/getBalance",
                params={"timestamp": timestamp},
                headers=headers,
                timeout=10
            )
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') != 100412:  # Не "Null signature"
                    print(f"✅ {base_url} работает!")
                    return base_url
                else:
                    print(f"❌ {base_url}: Null signature")
            else:
                print(f"❌ {base_url}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ {base_url}: Ошибка - {e}")
    
    return None

if __name__ == "__main__":
    print("🔍 Тестирование времени и базовых URL...")
    
    # Тестируем синхронизацию времени
    if test_time_sync():
        print("\n✅ Время синхронизировано")
    else:
        print("\n❌ Проблема с синхронизацией времени")
    
    # Тестируем подпись с временем сервера
    if test_with_server_time():
        print("\n✅ Подпись с временем сервера работает!")
    else:
        print("\n❌ Подпись с временем сервера не работает")
    
    # Тестируем разные базовые URL
    working_url = test_different_base_urls()
    if working_url:
        print(f"\n✅ Найден рабочий базовый URL: {working_url}")
    else:
        print("\n❌ Ни один базовый URL не работает")
    
    print("\n🔍 Тестирование завершено")