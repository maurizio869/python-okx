# Last modified (MSK): 2025-09-05 15:45 — правка номер 1
import os
from pathlib import Path
from dotenv import load_dotenv

# путь к .env: та же папка, где лежит текущий файл
dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path)

def check_api_keys():
    """Проверяем API ключи"""
    print("=== Проверка API ключей ===")
    
    api_key = os.getenv('api_key1', '1')
    api_secret = os.getenv('api_secret1', '1')
    
    print(f"API Key: {api_key}")
    print(f"API Secret: {api_secret}")
    
    # Проверяем длину
    print(f"\nДлина API Key: {len(api_key)}")
    print(f"Длина API Secret: {len(api_secret)}")
    
    # Проверяем, что это не дефолтные значения
    if api_key == '1' or api_secret == '1':
        print("\n❌ ПРОБЛЕМА: Используются дефолтные ключи!")
        print("Нужно заменить на реальные API ключи из BingX")
        return False
    
    # Проверяем формат (обычно API ключи имеют определенную длину)
    if len(api_key) < 20 or len(api_secret) < 20:
        print("\n❌ ПРОБЛЕМА: API ключи слишком короткие!")
        print("Возможно, это не настоящие API ключи")
        return False
    
    # Проверяем, что ключи содержат только допустимые символы
    import string
    allowed_chars = string.ascii_letters + string.digits
    
    for char in api_key:
        if char not in allowed_chars:
            print(f"\n❌ ПРОБЛЕМА: API Key содержит недопустимый символ: {char}")
            return False
    
    for char in api_secret:
        if char not in allowed_chars:
            print(f"\n❌ ПРОБЛЕМА: API Secret содержит недопустимый символ: {char}")
            return False
    
    print("\n✅ API ключи выглядят корректно")
    return True

def check_env_file():
    """Проверяем .env файл"""
    print("\n=== Проверка .env файла ===")
    
    env_path = Path(__file__).resolve().parent / '.env'
    print(f"Путь к .env: {env_path}")
    print(f"Файл существует: {env_path.exists()}")
    
    if env_path.exists():
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"Содержимое .env файла:")
            print(content)
            
            # Проверяем, что файл содержит нужные ключи
            if 'api_key1=' in content and 'api_secret1=' in content:
                print("✅ .env файл содержит нужные ключи")
            else:
                print("❌ .env файл не содержит нужные ключи")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка чтения .env файла: {e}")
            return False
    else:
        print("❌ .env файл не найден")
        return False
    
    return True

if __name__ == "__main__":
    print("🔍 Проверка API ключей и .env файла...")
    
    # Проверяем .env файл
    if not check_env_file():
        print("\n❌ Проблема с .env файлом")
        exit(1)
    
    # Проверяем API ключи
    if not check_api_keys():
        print("\n❌ Проблема с API ключами")
        print("\n📝 Инструкция по получению API ключей:")
        print("1. Зайдите на https://bingx.com")
        print("2. Войдите в свой аккаунт")
        print("3. Перейдите в 'API Management'")
        print("4. Создайте новый API ключ")
        print("5. Скопируйте API Key и Secret Key")
        print("6. Вставьте их в файл mal/.env")
        print("7. Убедитесь, что включены права на торговлю")
        exit(1)
    
    print("\n✅ Все проверки пройдены успешно!")