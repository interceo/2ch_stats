import const
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import re
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import string

# Создаем папку для сохранения результатов
os.makedirs('stats', exist_ok=True)

# Настройки визуализации
sns.set(style='darkgrid', font_scale=1.1)
plt.rcParams['figure.figsize'] = (14, 8)

# Отключаем использование NLTK, будем использовать регулярные выражения
print("Используем регулярные выражения для токенизации...")

def load_data():
    """Загрузка и предобработка данных из CSV файла"""
    print("Загрузка данных...")
    df = pd.read_csv(const.CSV_FILE)
    df.drop_duplicates(subset=['datetime', 'thread_num'], inplace=True)

    # Преобразование дат и времени
    try:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df['created_dt'] = pd.to_datetime(df['created_dt'], errors='coerce')
        df['last_seen'] = pd.to_datetime(df['last_seen'], errors='coerce')

        # Отбрасываем строки с некорректными датами
        df = df.dropna(subset=['datetime', 'created_dt', 'last_seen'])
        print(f"Даты успешно преобразованы, осталось {len(df)} записей")
    except Exception as e:
        print(f"Ошибка при обработке дат: {e}")

    # Преобразование числовых данных
    df['op_reply_ratio'] = pd.to_numeric(df['op_reply_ratio'], errors='coerce')
    df = df.dropna(subset=['op_reply_ratio'])

    # Создаем дополнительные признаки
    df['thread_lifetime_hours'] = (df['last_seen'] - df['created_dt']).dt.total_seconds() / 3600
    df['op_replied'] = np.where(df['op_reply_ratio'] > 0, 1, 0)
    df['created_hour'] = df['created_dt'].dt.hour
    df['created_minute'] = df['created_dt'].dt.minute
    df['created_dayofweek'] = df['created_dt'].dt.dayofweek

    # Преобразуем thread_num в строку для удобства работы
    df['thread_num'] = df['thread_num'].astype(str)

    print(f"Данные загружены, {len(df)} тредов для анализа")
    return df

def load_thread_json(thread_num):
    """Загрузка JSON-данных треда по его номеру"""
    filepath = f'threads/{thread_num}.json'
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Ошибка при чтении JSON-файла {filepath}: {e}")
            return None
    return None

def extract_linguistic_features(text):
    """Извлечение лингвистических признаков из текста (без использования NLTK)"""
    if not text or pd.isna(text):
        return {
            'text_length': 0,
            'avg_word_length': 0,
            'lexical_diversity': 0,
            'punctuation_ratio': 0,
            'uppercase_ratio': 0,
            'special_chars_ratio': 0
        }

    # Очистка текста от HTML-тегов
    clean_text = re.sub(r'<.*?>', '', text)

    # Токенизация с помощью регулярных выражений (простое разделение по пробелам и пунктуации)
    words_raw = re.findall(r'\b\w+\b', clean_text.lower())
    words = [word for word in words_raw if word]

    if not words:
        return {
            'text_length': len(clean_text),
            'avg_word_length': 0,
            'lexical_diversity': 0,
            'punctuation_ratio': sum(1 for c in clean_text if c in string.punctuation) / max(1, len(clean_text)),
            'uppercase_ratio': sum(1 for c in clean_text if c.isupper()) / max(1, len(clean_text)),
            'special_chars_ratio': sum(1 for c in clean_text if not c.isalnum() and c not in string.punctuation) / max(1, len(clean_text))
        }

    # Базовые метрики
    text_length = len(clean_text)
    avg_word_length = sum(len(word) for word in words) / max(1, len(words))

    # Лексическое разнообразие (уникальные слова / общее количество слов)
    lexical_diversity = len(set(words)) / max(1, len(words))

    # Доля знаков пунктуации
    punctuation_ratio = sum(1 for c in clean_text if c in string.punctuation) / max(1, len(clean_text))

    # Доля заглавных букв
    uppercase_ratio = sum(1 for c in clean_text if c.isupper()) / max(1, len(clean_text))

    # Доля специальных символов (не буквы, не цифры, не пунктуация)
    special_chars_ratio = sum(1 for c in clean_text if not c.isalnum() and c not in string.punctuation) / max(1, len(clean_text))

    return {
        'text_length': text_length,
        'avg_word_length': avg_word_length,
        'lexical_diversity': lexical_diversity,
        'punctuation_ratio': punctuation_ratio,
        'uppercase_ratio': uppercase_ratio,
        'special_chars_ratio': special_chars_ratio
    }

def analyze_thread_pattern(df):
    """Анализ паттернов создания тредов (регулярность, временные интервалы)"""
    # Группировка по часу и минуте создания
    time_pattern = df.groupby(['created_hour', 'created_minute']).size().reset_index(name='count')

    # Находим часы и минуты с аномально высокой частотой тредов (потенциальные боты)
    threshold = time_pattern['count'].mean() + 2 * time_pattern['count'].std()
    suspicious_times = time_pattern[time_pattern['count'] > threshold]

    # Находим треды, созданные в подозрительное время
    suspicious_threads = []
    for _, row in suspicious_times.iterrows():
        hour, minute = row['created_hour'], row['created_minute']
        threads = df[(df['created_hour'] == hour) & (df['created_minute'] == minute)]
        suspicious_threads.append(threads)

    if suspicious_threads:
        suspicious_df = pd.concat(suspicious_threads)
        return suspicious_df['thread_num'].tolist()
    return []

def detect_bots_by_text(df, top_n=50):
    """Обнаружение ботов на основе текстового анализа"""
    print("Анализ текстов заголовков тредов...")

    # Извлечение лингвистических признаков из заголовков
    features = []
    for _, row in df.iterrows():
        thread_features = extract_linguistic_features(row['title'])
        features.append(thread_features)

    # Создание DataFrame с признаками
    features_df = pd.DataFrame(features)
    df_combined = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

    # Применение алгоритма Isolation Forest для обнаружения аномалий
    feature_columns = ['text_length', 'avg_word_length', 'lexical_diversity',
                       'punctuation_ratio', 'uppercase_ratio', 'special_chars_ratio']

    # Удаляем строки с NaN значениями в выбранных признаках
    df_clean = df_combined.dropna(subset=feature_columns)

    # Если данных недостаточно, возвращаем пустой список
    if len(df_clean) < 10:
        print("Недостаточно данных для анализа")
        return []

    # Нормализация данных
    X = df_clean[feature_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Применение Isolation Forest
    clf = IsolationForest(contamination=0.1, random_state=42)
    df_clean['anomaly'] = clf.fit_predict(X_scaled)

    # Получение и сортировка аномалий по уровню отклонения
    df_clean['anomaly_score'] = clf.decision_function(X_scaled)
    df_anomalies = df_clean[df_clean['anomaly'] == -1].sort_values('anomaly_score')

    # Визуализация результатов (2D PCA для наглядности)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(12, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_clean['anomaly'], cmap='viridis', alpha=0.7)
    plt.colorbar(label='Аномалия (-1: аномалия, 1: норма)')
    plt.title('Обнаружение аномалий в лингвистических признаках заголовков', fontsize=16)
    plt.xlabel('Главная компонента 1', fontsize=14)
    plt.ylabel('Главная компонента 2', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('stats/text_anomalies_pca.png', dpi=300)
    plt.close()

    # Возвращаем top_n аномальных тредов
    return df_anomalies.head(top_n)['thread_num'].tolist()

def analyze_op_behavior(df, top_n=50):
    """Анализ поведения ОП для выявления ботов"""
    print("Анализ поведения ОП...")

    # Признаки для анализа
    behavior_features = ['op_reply_ratio', 'op_post_count', 'thread_lifetime_hours',
                         'posts_count', 'question_marks', 'anon_triggered']

    # Удаляем строки с NaN значениями
    df_clean = df.dropna(subset=behavior_features)

    # Если данных недостаточно, возвращаем пустой список
    if len(df_clean) < 10:
        print("Недостаточно данных для анализа поведения ОП")
        return []

    # Нормализация данных
    X = df_clean[behavior_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Применение Isolation Forest
    clf = IsolationForest(contamination=0.1, random_state=42)
    df_clean['behavior_anomaly'] = clf.fit_predict(X_scaled)

    # Получение и сортировка аномалий по уровню отклонения
    df_clean['behavior_score'] = clf.decision_function(X_scaled)
    df_anomalies = df_clean[df_clean['behavior_anomaly'] == -1].sort_values('behavior_score')

    # Визуализация аномалий в поведении ОП
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='op_reply_ratio',
        y='posts_count',
        hue='behavior_anomaly',
        size='op_post_count',
        palette={-1: 'red', 1: 'blue'},
        data=df_clean,
        alpha=0.7
    )
    plt.title('Аномалии в поведении ОП', fontsize=16)
    plt.xlabel('Доля ответов ОП', fontsize=14)
    plt.ylabel('Количество постов в треде', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('stats/op_behavior_anomalies.png', dpi=300)
    plt.close()

    return df_anomalies.head(top_n)['thread_num'].tolist()

def analyze_time_patterns(df):
    """Анализ временных паттернов создания тредов"""
    print("Анализ временных паттернов создания тредов...")

    # Группируем по часу создания
    hour_counts = df.groupby('created_hour').size()

    plt.figure(figsize=(14, 6))
    sns.barplot(x=hour_counts.index, y=hour_counts.values, palette='viridis')
    plt.title('Распределение создания тредов по часам суток', fontsize=16)
    plt.xlabel('Час суток', fontsize=14)
    plt.ylabel('Количество тредов', fontsize=14)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('stats/threads_per_hour_bots.png', dpi=300)
    plt.close()

    # Тепловая карта по часам и дням недели
    heatmap_data = pd.crosstab(df['created_dayofweek'], df['created_hour'])

    plt.figure(figsize=(16, 8))
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d',
                cbar_kws={'label': 'Количество тредов'})
    plt.title('Тепловая карта создания тредов по дням недели и часам', fontsize=16)
    plt.xlabel('Час суток', fontsize=14)
    plt.ylabel('День недели (0: Понедельник, 6: Воскресенье)', fontsize=14)
    plt.tight_layout()
    plt.savefig('stats/threads_heatmap_bots.png', dpi=300)
    plt.close()

    # Возвращаем потенциальные боты на основе регулярности создания
    return analyze_thread_pattern(df)

def check_thread_content(thread_num):
    """Подробный анализ содержимого треда"""
    thread_data = load_thread_json(thread_num)
    if not thread_data:
        return None

    thread_info = {}

    # Основная информация о треде
    thread_info['title'] = thread_data.get('title', '')
    thread_info['posts_count'] = thread_data.get('posts_count', 0)
    thread_info['unique_posters'] = thread_data.get('unique_posters', 0)

    # Анализ первого поста (ОП-пост)
    op_post = None
    if 'threads' in thread_data and thread_data['threads'] and 'posts' in thread_data['threads'][0]:
        posts = thread_data['threads'][0]['posts']
        if posts:
            op_post = posts[0]

    if op_post:
        # Текст ОП-поста
        thread_info['op_comment'] = op_post.get('comment', '')

        # Лингвистические признаки ОП-поста
        linguistic_features = extract_linguistic_features(thread_info['op_comment'])
        thread_info.update(linguistic_features)

        # Количество файлов в ОП-посте
        thread_info['op_files_count'] = len(op_post.get('files', [])) if op_post.get('files') else 0

        # Информация о времени создания
        thread_info['timestamp'] = op_post.get('timestamp', 0)
        thread_info['date'] = op_post.get('date', '')

    return thread_info

def identify_bot_threads(df, threshold=0.6):
    """Идентификация тредов, созданных ботами или нейросетями"""
    print("Выявление тредов, созданных ботами или нейросетями...")

    # Получаем подозрительные треды из каждого метода
    text_anomalies = detect_bots_by_text(df)
    behavior_anomalies = analyze_op_behavior(df)
    time_pattern_anomalies = analyze_time_patterns(df)

    # Объединяем результаты и подсчитываем, сколько раз каждый тред попал в подозрительные
    all_suspicious = text_anomalies + behavior_anomalies + time_pattern_anomalies
    thread_counts = Counter(all_suspicious)

    # Треды, которые попали в несколько категорий, с большей вероятностью боты
    multi_anomaly_threads = [thread for thread, count in thread_counts.items() if count >= 2]

    # Для отчета возьмем также и треды, которые попали только в одну категорию
    single_anomaly_threads = [thread for thread, count in thread_counts.items() if count == 1]

    # Более подробный анализ многоаномальных тредов
    detailed_results = []
    for thread_num in multi_anomaly_threads[:20]:  # Ограничиваем 20 тредами для детального анализа
        thread_info = check_thread_content(thread_num)
        if thread_info:
            thread_info['thread_num'] = thread_num
            thread_info['anomaly_sources'] = []
            if thread_num in text_anomalies:
                thread_info['anomaly_sources'].append('text')
            if thread_num in behavior_anomalies:
                thread_info['anomaly_sources'].append('behavior')
            if thread_num in time_pattern_anomalies:
                thread_info['anomaly_sources'].append('time_pattern')
            detailed_results.append(thread_info)

    # Создаем DataFrame для детальных результатов
    if detailed_results:
        detailed_df = pd.DataFrame(detailed_results)

        # Сохраняем информацию о подозрительных тредах
        detailed_df.to_csv('stats/suspicious_threads.csv', index=False)

        # Визуализация количества аномалий по типам
        anomaly_counts = Counter()
        for thread_info in detailed_results:
            for source in thread_info['anomaly_sources']:
                anomaly_counts[source] += 1

        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(anomaly_counts.keys()), y=list(anomaly_counts.values()), palette='Set2')
        plt.title('Распределение аномалий по типам', fontsize=16)
        plt.xlabel('Тип аномалии', fontsize=14)
        plt.ylabel('Количество тредов', fontsize=14)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig('stats/anomaly_types.png', dpi=300)
        plt.close()

    # Вывод результатов
    print(f"\nНайдено {len(multi_anomaly_threads)} тредов с множественными аномалиями (высокая вероятность ботов)")
    print(f"Найдено {len(single_anomaly_threads)} тредов с одиночными аномалиями (средняя вероятность ботов)")

    if detailed_results:
        print("\nТоп-5 подозрительных тредов:")
        for i, thread_info in enumerate(detailed_results[:5], 1):
            sources = ', '.join(thread_info['anomaly_sources'])
            title = thread_info.get('title', 'Нет заголовка')
            if len(title) > 70:
                title = title[:67] + '...'
            print(f"{i}. Тред {thread_info['thread_num']} - {title}")
            print(f"   Признаки бота: {sources}")

    return multi_anomaly_threads, single_anomaly_threads

if __name__ == "__main__":
    # Загружаем данные
    df = load_data()

    # Выявляем треды, созданные ботами
    high_prob_bots, med_prob_bots = identify_bot_threads(df)

    print("\nАнализ завершен! Результаты сохранены в папку 'stats'")
