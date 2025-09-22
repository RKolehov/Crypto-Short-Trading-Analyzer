import pandas as pd
import os

# Параметри
folder_path = r'C:\Users\HOME\Desktop\Data Analytics\Qaunter_test'  # Шлях до директорії
output_file = 'combined_output.csv'  # Вихідний файл

# Список для зберігання даних
all_data = []

# Обробка файлів
for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        # Витягуємо назву монети (все до першого дефіса)
        coin_name = file.split('-')[0]

        # === АВТОВИЗНАЧЕННЯ РОЗДІЛЬНИКА ===
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if ";" in first_line and first_line.count(";") > first_line.count(","):
                sep = ";"
            else:
                sep = ","

        # Читання CSV з правильним роздільником
        df = pd.read_csv(file_path, header=0, sep=sep)

        # Додавання колонки coin
        df.insert(0, 'coin', coin_name)
        all_data.append(df)

# Об'єднання всіх даних
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    # Сортування за datetime
    if "datetime" in combined_df.columns:
        combined_df = combined_df.sort_values("datetime")
    # Збереження у CSV
    combined_df.to_csv(os.path.join(folder_path, output_file), index=False, encoding="utf-8")
    print(f'✅ Об\'єднано {len(all_data)} файлів, збережено {len(combined_df)} рядків у {output_file}')
else:
    print("⚠️ Не знайдено CSV файлів")
