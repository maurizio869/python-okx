# В начале скрипта - НЕ импортируем matplotlib
# import matplotlib.pyplot as plt  # ЗАКОММЕНТИРОВАТЬ

# Ваш код без matplotlib
def your_main_code():
    # Весь ваш код здесь
    pass

# Импортируем matplotlib только когда нужно
def import_matplotlib_when_needed():
    import os
    os.environ['MPLBACKEND'] = 'Agg'
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    plt.ioff()
    return plt, patches

# Используйте так:
# plt, patches = import_matplotlib_when_needed()
# Только когда действительно нужно matplotlib