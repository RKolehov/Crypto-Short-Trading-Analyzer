import pandas as pd
from datetime import timedelta

# Параметри (змінювати тут)
file_path = 'M_USDT-202507.csv'  # Шлях до файлу
end_date_time = '2025-07-04 02:00:00'  # Кінцевий час (UTC)

# Читання CSV
df = pd.read_csv(file_path, header=None)
df.columns = ['timestamp', 'volume', 'open', 'high', 'low', 'close']

# Конвертація timestamp у datetime (UTC)
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

# Форматування datetime без секунд і без +00:00
df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M')

# Обчислення початку (18 годин до кінця)
end_time = pd.to_datetime(end_date_time, utc=True)
start_time = end_time - timedelta(hours=18)

# Фільтрація
df_filtered = df[(df['datetime'] >= start_time.strftime('%Y-%m-%d %H:%M')) & (df['datetime'] <= end_time.strftime('%Y-%m-%d %H:%M'))]

# Збереження
output_file = f'filtered_{file_path}'
df_filtered.to_csv(output_file, index=False, encoding='utf-8')
print(f'Збережено {len(df_filtered)} рядків у {output_file}') 