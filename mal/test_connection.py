# Last modified (MSK): 2025-09-06 07:45 — правка номер 1
import requests
import socket

def test_connection():
    """Тестируем подключение к BingX API"""
    print("🔍 Тестирование подключения к BingX API...")
    
    # Тест 1: Проверяем DNS разрешение
    print("\n=== Тест 1: DNS разрешение ===")
    try:
        ip = socket.gethostbyname('open-api.bingx.com')
        print(f"✅ DNS разрешен: open-api.bingx.com -> {ip}")
    except Exception as e:
        print(f"❌ DNS ошибка: {e}")
        return False
    
    # Тест 2: Проверяем доступность основного сайта
    print("\n=== Тест 2: Доступность основного сайта ===")
    try:
        response = requests.get("https://bingx.com", timeout=10)
        print(f"✅ bingx.com доступен: {response.status_code}")
    except Exception as e:
        print(f"❌ bingx.com недоступен: {e}")
    
    # Тест 3: Проверяем API endpoint
    print("\n=== Тест 3: API endpoint ===")
    try:
        response = requests.get("https://open-api.bingx.com/openApi/swap/v1/server/time", timeout=10)
        print(f"✅ API endpoint доступен: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ API endpoint недоступен: {e}")
    
    # Тест 4: Проверяем альтернативные endpoints
    print("\n=== Тест 4: Альтернативные endpoints ===")
    endpoints = [
        "https://open-api.bingx.com/openApi/swap/v2/server/time",
        "https://open-api.bingx.com/openApi/spot/v1/server/time",
        "https://api.bingx.com/openApi/swap/v1/server/time"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=5)
            print(f"✅ {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint}: {e}")
    
    return True

if __name__ == "__main__":
    test_connection()