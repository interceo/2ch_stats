from thread_loader import ThreadDataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta

# Создадим папку для сохранения графиков, если её нет
os.makedirs('stats', exist_ok=True)

# Настройки отображения
sns.set(style='darkgrid', font_scale=1.1)
plt.rcParams['figure.figsize'] = (14, 8)

# Установим минимальное количество постов для анализа
min_posts_count = 0

def load_data(max_threads: int = None):
    """Загрузка данных из JSON файлов тредов

    Args:
        max_threads: Максимальное количество тредов для загрузки. Если None, загружает все.
    """
    print("Загрузка данных из JSON файлов...")
    loader = ThreadDataLoader()
    df = loader.load_all_threads(max_threads=max_threads)

    # Фильтрация по минимальному количеству постов
    df = df[df['posts_count'] >= min_posts_count]
    print(f"После фильтрации осталось {len(df)} тредов (минимум {min_posts_count} постов)")

    return df

def analyze_temporal_patterns(df):
    """Анализ временных паттернов"""
    print("Анализ временных паттернов...")

    # Фильтруем данные с валидными временными метками
    time_df = df[df['created_dt'].notna()]

    if len(time_df) == 0:
        print("Нет данных о времени создания тредов")
        return

    # 1. Распределение по часам создания
    plt.figure(figsize=(14, 6))
    hour_counts = time_df['created_hour'].value_counts().sort_index()
    sns.barplot(x=hour_counts.index, y=hour_counts.values, palette='viridis')
    plt.title('Распределение создания тредов по часам суток', fontsize=16)
    plt.xlabel('Час суток', fontsize=14)
    plt.ylabel('Количество тредов', fontsize=14)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('stats/threads_per_hour.png', dpi=300)
    plt.close()

    # 2. Распределение по дням недели
    if 'created_day_name' in time_df.columns:
        plt.figure(figsize=(12, 6))
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_names_ru = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
        day_counts = time_df['created_day_name'].value_counts().reindex(day_order)

        plt.figure(figsize=(12, 6))
        bars = sns.barplot(x=range(len(day_counts)), y=day_counts.values, palette='Set1')
        plt.title('Распределение создания тредов по дням недели', fontsize=16)
        plt.xlabel('День недели', fontsize=14)
        plt.ylabel('Количество тредов', fontsize=14)
        plt.xticks(range(len(day_names_ru)), day_names_ru, rotation=45)

        # Добавляем числа над столбцами
        for i, v in enumerate(day_counts.values):
            if not pd.isna(v):
                plt.text(i, v + 0.1, str(int(v)), ha='center', va='bottom')

        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig('stats/threads_per_day.png', dpi=300)
        plt.close()

    # 3. Тепловая карта активности (часы × дни недели)
    if 'created_hour' in time_df.columns and 'created_day' in time_df.columns:
        plt.figure(figsize=(20, 10))

        # Создаем таблицу сопряженности для тепловой карты
        heatmap_data = pd.crosstab(time_df['created_day'], time_df['created_hour'])

        # Убеждаемся, что у нас есть все часы (0-23) и дни (0-6)
        all_hours = range(24)
        all_days = range(7)

        # Переиндексируем данные, чтобы включить все часы и дни
        heatmap_data = heatmap_data.reindex(index=all_days, columns=all_hours, fill_value=0)

        # Переименовываем индексы для русских названий дней
        day_names_mapping = {0: 'Понедельник', 1: 'Вторник', 2: 'Среда', 3: 'Четверг',
                           4: 'Пятница', 5: 'Суббота', 6: 'Воскресенье'}
        heatmap_data.index = heatmap_data.index.map(day_names_mapping)

        # Создаем более красивую тепловую карту
        sns.heatmap(heatmap_data,
                   cmap='RdYlBu_r',
                   annot=True,
                   fmt='d',
                   cbar_kws={'label': 'Количество тредов'},
                   linewidths=0.5,
                   square=False,
                   annot_kws={'size': 8})

        plt.title('Тепловая карта активности создания тредов по часам и дням недели', fontsize=18)
        plt.xlabel('Час суток', fontsize=16)
        plt.ylabel('День недели', fontsize=16)

        # Устанавливаем метки для часов
        plt.xticks(range(24), [f'{h:02d}:00' for h in range(24)], rotation=45)

        plt.tight_layout()
        plt.savefig('stats/activity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Средняя популярность тредов по дням недели
    if 'created_day_name' in time_df.columns:
        plt.figure(figsize=(12, 6))

        # Группируем по дням недели и считаем среднюю популярность
        day_popularity = time_df.groupby('created_day_name').agg({
            'posts_count': 'mean',
            'unique_posters': 'mean'
        }).reindex(day_order)

        # Создаем график с двумя осями
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

        x_pos = range(len(day_names_ru))
        bars1 = ax1.bar([x - 0.2 for x in x_pos], day_popularity['posts_count'].values,
                       width=0.4, alpha=0.7, color='steelblue', label='Среднее количество постов')
        bars2 = ax2.bar([x + 0.2 for x in x_pos], day_popularity['unique_posters'].values,
                       width=0.4, alpha=0.7, color='coral', label='Среднее количество пользователей')

        ax1.set_xlabel('День недели', fontsize=14)
        ax1.set_ylabel('Среднее количество постов', fontsize=14, color='steelblue')
        ax2.set_ylabel('Среднее количество пользователей', fontsize=14, color='coral')

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(day_names_ru, rotation=45)

        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='coral')

        plt.title('Средняя популярность тредов по дням недели', fontsize=16)

        # Добавляем легенду
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('stats/day_popularity.png', dpi=300)
        plt.close()

    # 5. Динамика активности в течение недели (линейный график)
    if 'created_day' in time_df.columns:
        plt.figure(figsize=(12, 6))

        # Считаем активность по дням недели
        day_activity = time_df.groupby('created_day').size()

        # Создаем плавную линию
        x_smooth = range(7)
        y_smooth = [day_activity.get(i, 0) for i in range(7)]

        plt.plot(x_smooth, y_smooth, marker='o', linewidth=3, markersize=8,
                color='darkgreen', markerfacecolor='lightgreen', markeredgewidth=2)

        # Заполняем область под кривой
        plt.fill_between(x_smooth, y_smooth, alpha=0.3, color='lightgreen')

        plt.title('Динамика создания тредов в течение недели', fontsize=16)
        plt.xlabel('День недели', fontsize=14)
        plt.ylabel('Количество тредов', fontsize=14)
        plt.xticks(range(7), day_names_ru, rotation=45)

        # Добавляем значения на точки
        for i, v in enumerate(y_smooth):
            plt.annotate(str(v), (i, v), textcoords="offset points",
                        xytext=(0,10), ha='center')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('stats/weekly_trend.png', dpi=300)
        plt.close()

def create_correlation_matrix(df):
    """Создание корреляционной матрицы"""
    print("Создание корреляционной матрицы...")

    # Выбираем численные признаки для корреляции
    numeric_features = [
        'posts_count', 'unique_posters', 'total_files', 'op_comment_length',
        'op_files_count', 'op_reply_ratio', 'question_marks', 'title_len',
        'avg_post_length', 'files_per_post', 'unique_posters_ratio'
    ]

    # Добавляем длительность треда, если есть данные
    if 'thread_duration_hours' in df.columns and df['thread_duration_hours'].notna().any():
        numeric_features.append('thread_duration_hours')

    # Фильтруем только существующие колонки
    available_features = [f for f in numeric_features if f in df.columns]

    if len(available_features) < 2:
        print("Недостаточно численных признаков для корреляционного анализа")
        return

    corr_matrix = df[available_features].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f',
                linewidths=0.5, square=True)
    plt.title('Корреляционная матрица признаков тредов', fontsize=16)
    plt.tight_layout()
    plt.savefig('stats/correlation_matrix.png', dpi=300)
    plt.close()

def generate_summary_statistics(df):
    """Генерация сводной статистики"""
    print("Генерация сводной статистики...")

    # Основная статистика
    stats = {
        'Всего тредов': len(df),
        'Средняя длина заголовка': df['title_len'].mean(),
        'Среднее количество постов': df['posts_count'].mean(),
        'Среднее количество уникальных пользователей': df['unique_posters'].mean(),
        'Процент тредов с файлами': (df['has_files'].sum() / len(df)) * 100,
        'Процент тредов с активным ОП': (df['op_reply_ratio'] > 0).sum() / len(df) * 100,
        'Средняя длина комментария ОП': df['op_comment_length'].mean(),
        'Среднее количество файлов на пост': df['files_per_post'].mean()
    }

    if 'thread_duration_hours' in df.columns and df['thread_duration_hours'].notna().any():
        valid_duration = df[df['thread_duration_hours'] > 0]['thread_duration_hours']
        if len(valid_duration) > 0:
            stats['Средняя длительность жизни треда (часы)'] = valid_duration.mean()

    # Сохраняем статистику в файл
    with open('stats/summary_statistics.txt', 'w', encoding='utf-8') as f:
        f.write("СВОДНАЯ СТАТИСТИКА АНАЛИЗА ТРЕДОВ\n")
        f.write("=" * 50 + "\n\n")

        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.2f}\n")
            else:
                f.write(f"{key}: {value}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("Топ-5 самых популярных тредов:\n")
        top_threads = df.nlargest(5, 'posts_count')[['thread_num', 'title', 'posts_count', 'unique_posters']]
        for _, thread in top_threads.iterrows():
            title = thread['title'][:60] + '...' if len(thread['title']) > 60 else thread['title']
            f.write(f"[{thread['thread_num']}] {title} ({thread['posts_count']} постов, {thread['unique_posters']} пользователей)\n")

    # Выводим основную статистику в консоль
    print("\nОСНОВНАЯ СТАТИСТИКА:")
    print("-" * 50)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

def main():
    """Основная функция для выполнения всего анализа"""
    try:
        # Загружаем данные
        df = load_data()

        if len(df) == 0:
            print("Нет данных для анализа")
            return

        analyze_temporal_patterns(df)
        create_correlation_matrix(df)
        generate_summary_statistics(df)

        print("\nАнализ завершен! Все графики и статистика сохранены в папке 'stats'")

    except Exception as e:
        print(f"Ошибка при выполнении анализа: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
