import os
os.environ['MPLBACKEND'] = 'Agg'  # Принудительно используем неинтерактивный backend

# Теперь можно импортировать matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Дополнительно отключаем интерактивный режим
plt.ioff()  # Отключаем интерактивный режим