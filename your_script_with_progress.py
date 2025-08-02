from okx.exceptions import OkxAPIException
import os
import sys
import threading

from datetime import datetime
from time import sleep

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pytz
from okx.MarketData import MarketAPI

import logging

import ta.trend
import numpy as np
import math
import time
import json
from dotenv import load_dotenv

# В начале скрипта
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.flush()
load_dotenv()

logger = logging.getLogger('azzraelcode-yt')
Account_QTY = 0.0

inst_id = os.getenv('SYMBOL', 'HBAR-USDT')
tf = os.getenv('TIMEFRAME', '1m')
limit = int(os.getenv('limit', 900))
limitB = 100
pages = 30 #15 on d

# Глобальные переменные для статуса
current_status = "Инициализация..."
current_progress = 0
is_running = True

# Функция, которая будет работать в фоне
def background_logger():
    counter = 0
    while is_running:
        counter += 1
        try:
            with open("/storage/emulated/0/status.txt", "w", encoding="utf-8") as f:
                f.write(f"Статус: {current_status}\n")
                f.write(f"Прогресс: {current_progress}%\n")
                f.write(f"Счётчик: {counter}\n")
                f.write(f"Время: {time.strftime('%H:%M:%S')}\n")
                f.write("Если счётчик растёт - скрипт работает!")
        except Exception as e:
            try:
                with open("/storage/emulated/0/error_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"Ошибка в фоновой задаче: {e}\n")
            except:
                pass
        time.sleep(1)

# Функция для обновления статуса
def update_status(status, progress=None):
    global current_status, current_progress
    current_status = status
    if progress is not None:
        current_progress = progress
    print(f"[{time.strftime('%H:%M:%S')}] {status}")  # Пытаемся вывести на экран

# Запускаем фоновую задачу
print("=== НАЧАЛО СКРИПТА ===")
print("Запускаем фоновую задачу...")
thread = threading.Thread(target=background_logger, daemon=True)
thread.start()
time.sleep(2)
print("Фоновая задача запущена!")

class BackT: 
    
    def __init__(self):
        super(BackT, self).__init__()
        
    def from_str_ms(self, time : str) -> datetime:
        return datetime.fromtimestamp(time+10800, tz=pytz.UTC)
        
    def write_read(self):
        update_status("*** OKX Candles History ***")
        candles = {}
        second_values = {}
        cl = MarketAPI(
            flag=os.getenv('IS_DEMO'),
            debug=False
            )

        update_status(f"Время: {self.from_str_ms(time.time())}")
        for page in range(0, pages):
            progress = int((page / pages) * 100)
            update_status(f"Обработка страницы {page+1}/{pages}", progress)
            
            open_times = sorted([*candles.keys()])

            # https://www.okx.com/docs-v5/en/#order-book-trading-market-data-get-candlesticks
            r = cl.get_history_candlesticks(
                instId=inst_id,
                bar=tf,
                limit=limitB,
                after=str(int(open_times[0])*1000) if len(open_times) > 0 else '',
                )
            data = r.get('data', [])
            
            for c in data:
                candles[int(c[0])//1000] = dict(
                    x=(int(c[0])+60000)//1000,
                    o=float(c[1]),
                    h=float(c[2]),
                    l=float(c[3]),
                    c=float(c[4]),
                    v=float(c[7]),
                    conf=int(c[8]),
                    )
                  
            # получили меньше чем могли, соотв на след странице ничего нет
            if len(data) < limitB:
                update_status(f"Завершено на странице {page}")
                break

            # задержка чтобы не нарушать лимиты
            if page > 0 and page % 20 == 0:
                update_status(f"Страница {page} из {pages}, {int(page/pages*100)}% прогресс, задержка безопасности...")
                sleep(1.1)
                
        update_status("Загрузка файлов...")
        with open('candles_10d.json', 'r') as file1: # 1d2806 2d2706 3d2606 6d1605  10d2106 30d2206
            self.candles_r=json.load(file1)
        with open('candles_keys_10d.json', 'r') as file1:
            self.candles_keys_r=json.load(file1)

        update_status(f"len(candles_keys_r) is {len(self.candles_keys_r)}")
        update_status(f"min candles_keys_r is {min(self.candles_keys_r)}")
        update_status(f"max candles_keys_r is {max(self.candles_keys_r)}")
        
        self.sma_mark = int(os.getenv('SMA_mark', '1'))
        self.sma_ma1 = int(os.getenv('SMA_ma1', '2'))
        self.sma_ma2 = int(os.getenv('SMA_ma2', '5'))
        self.sma_ma3 = int(os.getenv('SMA_ma3', '8'))
        self.sma_ma4 = int(os.getenv('SMA_ma4', '16'))
        self.sma_ma5 = int(os.getenv('SMA_ma5', '32'))
        self.sma_ma6 = int(os.getenv('SMA_ma6', '64'))
        self.sma_ma7 = int(os.getenv('SMA_ma7', '128'))
        self.sma_ma8 = int(os.getenv('SMA_ma8', '256'))
        self.sma_ma9 = int(os.getenv('SMA_ma9', '512'))
        
obj=BackT()            

try:
    update_status("Начинаем выполнение основного кода...")
    obj.write_read()
    update_status("Код выполнен успешно!", 100)
    
    # Даём время фоновой задаче завершиться
    time.sleep(3)
    
except KeyboardInterrupt as e:
    update_status("Бот остановлен вручную!")
    logger.debug("Бот остановлен вручную!")
except OkxAPIException as e:
    update_status(f"Ошибка API: {e}")
    logger.debug(str(e))
except Exception as e:
    update_status(f"Ошибка: {e}")
    logger.error(str(e))
    # Логируем ошибку в файл
    try:
        with open("/storage/emulated/0/error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"Ошибка в основном коде: {e}\n")
    except:
        pass
finally:
    # Останавливаем фоновую задачу
    global is_running
    is_running = False
    update_status("Скрипт завершён")