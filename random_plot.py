import matplotlib
matplotlib.use('Agg')  # Используем backend без GUI
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta

# Настройка стиля
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Генерация рандомных данных
np.random.seed(random.randint(1, 1000))

# Создание временных меток
start_date = datetime.now() - timedelta(days=30)
dates = [start_date + timedelta(days=i) for i in range(30)]

# Генерация различных типов рандомных данных
data_types = ['sine_wave', 'random_walk', 'exponential', 'polynomial']
selected_type = random.choice(data_types)

if selected_type == 'sine_wave':
    # Синусоидальная волна с шумом
    x = np.linspace(0, 4*np.pi, 30)
    y = np.sin(x) * np.random.uniform(10, 50) + np.random.normal(0, 5, 30)
    title = "Синусоидальная волна с шумом"
    color = 'blue'
    
elif selected_type == 'random_walk':
    # Случайное блуждание
    y = np.cumsum(np.random.randn(30))
    x = np.arange(30)
    title = "Случайное блуждание"
    color = 'red'
    
elif selected_type == 'exponential':
    # Экспоненциальный рост с шумом
    x = np.arange(30)
    y = np.exp(x/10) * (1 + np.random.normal(0, 0.1, 30))
    title = "Экспоненциальный рост с шумом"
    color = 'green'
    
else:  # polynomial
    # Полиномиальная функция с шумом
    x = np.linspace(-3, 3, 30)
    y = x**3 - 2*x**2 + x + np.random.normal(0, 2, 30)
    title = "Полиномиальная функция с шумом"
    color = 'purple'

# Создание графика
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Основной график
ax.plot(x, y, marker='o', linewidth=2, markersize=6, color=color, alpha=0.8)

# Добавление трендовой линии для некоторых типов
if selected_type in ['exponential', 'polynomial']:
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    ax.plot(x, p(x), '--', color='orange', linewidth=2, alpha=0.7, label='Тренд')
    ax.legend()

# Настройка осей и заголовков
ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('X', fontsize=14)
ax.set_ylabel('Y', fontsize=14)
ax.grid(True, alpha=0.3)

# Добавление случайных аннотаций
if random.choice([True, False]):
    max_idx = np.argmax(y)
    min_idx = np.argmin(y)
    
    ax.annotate(f'Максимум: {y[max_idx]:.2f}', 
                xy=(x[max_idx], y[max_idx]), 
                xytext=(10, 10), 
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.annotate(f'Минимум: {y[min_idx]:.2f}', 
                xy=(x[min_idx], y[min_idx]), 
                xytext=(10, -10), 
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Добавление статистики
stats_text = f'Среднее: {np.mean(y):.2f}\nСтд. откл.: {np.std(y):.2f}\nМедиана: {np.median(y):.2f}'
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Случайный выбор цветовой схемы
color_schemes = ['default', 'dark', 'pastel']
scheme = random.choice(color_schemes)

if scheme == 'dark':
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(colors='white')
elif scheme == 'pastel':
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

# Сохранение графика
plt.tight_layout()
plt.savefig('random_plot.png', dpi=300, bbox_inches='tight')

print(f"График типа '{selected_type}' создан и сохранен как 'random_plot.png'")
print(f"Цветовая схема: {scheme}")
print(f"Статистика данных:")
print(f"  - Количество точек: {len(y)}")
print(f"  - Минимум: {np.min(y):.2f}")
print(f"  - Максимум: {np.max(y):.2f}")
print(f"  - Среднее: {np.mean(y):.2f}")
print(f"  - Стандартное отклонение: {np.std(y):.2f}")