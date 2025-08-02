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

# Функция, которая будет работать в фоне
def background_logger():
    counter = 0
    while True:
        counter += 1
        try:
            with open("/storage/emulated/0/status.txt", "w", encoding="utf-8") as f:
                f.write(f"Скрипт работает... Счётчик: {counter}\n")
                f.write(f"Время: {time.strftime('%H:%M:%S')}\n")
                f.write("Если счётчик растёт - скрипт работает!")
        except Exception as e:
            # Логируем ошибки в фоновой задаче
            try:
                with open("/storage/emulated/0/error_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"Ошибка в фоновой задаче: {e}\n")
            except:
                pass
        time.sleep(1)

# Запускаем фоновую задачу
print("Запускаем фоновую задачу...")
thread = threading.Thread(target=background_logger, daemon=True)
thread.start()

# ВАЖНО: Даём время фоновой задаче запуститься
time.sleep(2)
print("Фоновая задача запущена!")

class BackT: 
    
    def __init__(self):
        super(BackT, self).__init__()
        
    def from_str_ms(self, time : str) -> datetime:
        return datetime.fromtimestamp(time+10800, tz=pytz.UTC)
        
    def write_read(self):
        print("*** OKX Candles History ***")
        candles = {}
        second_values = {}
        cl = MarketAPI(
            flag=os.getenv('IS_DEMO'),
            debug=False
            )

        print(self.from_str_ms(time.time()))
        for page in range(0, pages):
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
                print(f"Finished at page {page}")
                break

            # задержка чтобы не нарушать лимиты
            if page > 0 and page % 20 == 0:
                print(f"page {page} of {pages}, {int(page/pages*100)}% candles progress, candles safety delay...")
                sleep(1.1)
                
        with open('candles_10d.json', 'r') as file1: # 1d2806 2d2706 3d2606 6d1605  10d2106 30d2206
            self.candles_r=json.load(file1)
        with open('candles_keys_10d.json', 'r') as file1:
            self.candles_keys_r=json.load(file1)

        print('len(candles_keys_r) is', len(self.candles_keys_r))
        print('min candles_keys_r is', min(self.candles_keys_r))
        print('max candles_keys_r is', max(self.candles_keys_r))
        
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
    print("Начинаем выполнение основного кода...")
    obj.write_read()
    print("Код выполнен успешно!")
    
    # Даём время фоновой задаче завершиться
    time.sleep(3)
    
except KeyboardInterrupt as e:
    logger.debug("Бот остановлен вручную!")
except OkxAPIException as e:
    logger.debug(str(e))
except Exception as e:
    logger.error(str(e))
    # Логируем ошибку в файл
    try:
        with open("/storage/emulated/0/error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"Ошибка в основном коде: {e}\n")
    except:
        pass