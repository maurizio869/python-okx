# Last modified (MSK): 2025-09-06 09:05 — правка номер 1
import requests
import time

def test_public_endpoints():
    """Тестируем публичные endpoints без подписи"""
    print("🔍 Тестирование публичных endpoints...")
    
    # Публичные endpoints (не требуют подписи)
    public_endpoints = [
        # Разные варианты для получения информации о рынке
        "/openApi/swap/v2/market/ticker",
        "/openApi/swap/v1/market/ticker", 
        "/openApi/swap/market/ticker",
        "/openApi/market/ticker",
        "/api/swap/v2/market/ticker",
        "/openApi/swap/v2/market/price",
        "/openApi/swap/v1/market/price",
        "/openApi/swap/market/price",
        "/openApi/market/price",
        "/openApi/swap/v2/market/depth",
        "/openApi/swap/v1/market/depth",
        "/openApi/swap/market/depth",
        "/openApi/market/depth"
    ]
    
    working_endpoints = []
    
    for endpoint in public_endpoints:
        print(f"\n--- Тестируем {endpoint} ---")
        try:
            response = requests.get(
                f"https://open-api.bingx.com{endpoint}",
                params={"symbol": "BTC-USDT"},
                timeout=10
            )
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text[:300]}...")  # Показываем первые 300 символов
            
            if response.status_code == 200:
                result = response.json()
                # Ищем успешные ответы
                if (result.get('code') == 0 or 
                    'data' in result or
                    'success' in str(result).lower() or
                    'ticker' in str(result).lower() or
                    'price' in str(result).lower()):
                    print(f"✅ {endpoint} работает!")
                    working_endpoints.append(endpoint)
                else:
                    print(f"❌ {endpoint}: {result.get('msg', 'Unknown error')}")
            else:
                print(f"❌ {endpoint}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ {endpoint}: Ошибка - {e}")
    
    return working_endpoints

def test_different_base_urls():
    """Тестируем разные базовые URL"""
    print("\n🔍 Тестирование разных базовых URL...")
    
    base_urls = [
        "https://open-api.bingx.com",
        "https://api.bingx.com", 
        "https://openapi.bingx.com",
        "https://api.bingx.com/openApi",
        "https://open-api.bingx.com/openApi"
    ]
    
    working_urls = []
    
    for base_url in base_urls:
        print(f"\n--- Тестируем {base_url} ---")
        try:
            response = requests.get(
                f"{base_url}/openApi/swap/v2/market/ticker",
                params={"symbol": "BTC-USDT"},
                timeout=10
            )
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            
            if response.status_code == 200:
                result = response.json()
                if (result.get('code') == 0 or 
                    'data' in result or
                    'success' in str(result).lower()):
                    print(f"✅ {base_url} работает!")
                    working_urls.append(base_url)
                else:
                    print(f"❌ {base_url}: {result.get('msg', 'Unknown error')}")
            else:
                print(f"❌ {base_url}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ {base_url}: Ошибка - {e}")
    
    return working_urls

def test_simple_api_call():
    """Простой тест API вызова"""
    print("\n🔍 Простой тест API вызова...")
    
    try:
        # Простейший запрос
        response = requests.get(
            "https://open-api.bingx.com/openApi/swap/v2/market/ticker",
            params={"symbol": "BTC-USDT"},
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"JSON Response: {result}")
            
            if result.get('code') == 0:
                print("✅ API работает!")
                return True
            else:
                print(f"❌ API ошибка: {result.get('msg', 'Unknown')}")
                return False
        else:
            print(f"❌ HTTP ошибка: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Тестирование подключения к BingX API...")
    
    # Простой тест
    if test_simple_api_call():
        print("\n✅ Базовое подключение работает!")
    else:
        print("\n❌ Базовое подключение не работает!")
    
    # Тестируем разные базовые URL
    working_urls = test_different_base_urls()
    if working_urls:
        print(f"\n✅ Рабочие URL: {working_urls}")
    else:
        print("\n❌ Не найдены рабочие URL")
    
    # Тестируем публичные endpoints
    working_endpoints = test_public_endpoints()
    if working_endpoints:
        print(f"\n✅ Рабочие endpoints: {working_endpoints}")
    else:
        print("\n❌ Не найдены рабочие endpoints")
    
    print("\n🔍 Тестирование завершено")