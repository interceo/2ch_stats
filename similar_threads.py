import const
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import re
import string
import os
from collections import Counter, defaultdict

# Создаем папку для сохранения результатов
os.makedirs('simular', exist_ok=True)

# Настройки визуализации
sns.set(style='darkgrid', font_scale=1.1)
plt.rcParams['figure.figsize'] = (14, 8)

# Предопределенные шаблоны для поиска
PATTERNS = {
    'оценка': r'(оцени|оцените|как вам|рейт|rate|что думаете|как тебе|как по-вашему)',
    'мнение': r'(мнение|как считае(шь|те)|как дума(ешь|ете)|как полага(ешь|ете))',
    'вопрос': r'(\?|что ты|что вы|почему|зачем|когда|где|куда|откуда|как)',
    'предложение': r'(предлагаю|давайте|а не|может|как насчет|как считаете)',
    'хвастовство': r'(смотрите|глядите|зацените|показываю|демонстрирую|хвастаюсь)',
    'тред': r'(тред|thread|webm|шебм|mp4)',
    'просьба': r'(помогите|прошу|требуется|нужна? помощь|подскажите|посоветуйте)',
    'представление': r'(представь|представьте|вообрази|вообразите)'
}

def load_data():
    """Загрузка и предобработка данных из CSV файла"""
    print("Загрузка данных...")
    df = pd.read_csv(const.CSV_FILE)

    # Убираем дубликаты
    df.drop_duplicates(subset=['thread_num'], inplace=True)

    # Фильтруем записи с пустыми заголовками
    df = df[df['title'].notna() & (df['title'] != '')]

    # Приводим к строковому типу
    df['title'] = df['title'].astype(str)

    print(f"Загружено {len(df)} тредов для анализа")
    return df

def preprocess_text(text):
    """Предобработка текста для анализа"""
    # Приведение к нижнему регистру
    text = text.lower()

    # Удаление HTML-тегов
    text = re.sub(r'<.*?>', '', text)

    # Удаление URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def pattern_based_grouping(df):
    """Группировка тредов на основе предопределенных шаблонов"""
    print("Выполняется группировка по шаблонам...")

    # Создаем словарь для хранения результатов
    pattern_groups = defaultdict(list)

    for _, row in df.iterrows():
        title = preprocess_text(row['title'])
        thread_num = row['thread_num']

        # Определяем, к какому шаблону относится заголовок
        matched = False
        for pattern_name, pattern in PATTERNS.items():
            if re.search(pattern, title):
                pattern_groups[pattern_name].append((thread_num, title))
                matched = True

        # Если ни один шаблон не подошел, добавляем в категорию "другое"
        if not matched:
            pattern_groups['другое'].append((thread_num, title))

    # Визуализация результатов
    pattern_counts = {k: len(v) for k, v in pattern_groups.items()}
    plt.figure(figsize=(12, 6))
    bars = plt.bar(pattern_counts.keys(), pattern_counts.values(), color=sns.color_palette('pastel'))

    # Добавляем числа над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height}', ha='center', va='bottom')

    plt.title('Распределение тредов по шаблонам', fontsize=16)
    plt.xlabel('Тип шаблона', fontsize=14)
    plt.ylabel('Количество тредов', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('stats/pattern_distribution.png', dpi=300)
    plt.close()

    # Создаем DataFrame с результатами
    result_data = []
    for pattern_name, threads in pattern_groups.items():
        for thread_num, title in threads:
            result_data.append({
                'thread_num': thread_num,
                'title': title,
                'pattern': pattern_name
            })

    pattern_df = pd.DataFrame(result_data)

    # Сохраняем результаты в CSV
    pattern_df.to_csv('stats/pattern_groups.csv', index=False)

    # Выводим примеры для каждой группы
    print("\nПримеры тредов для каждой группы шаблонов:")
    for pattern_name, threads in pattern_groups.items():
        if threads:
            print(f"\n{pattern_name.upper()} ({len(threads)} тредов):")
            for i, (thread_num, title) in enumerate(threads[:3]):
                if len(title) > 70:
                    title = title[:67] + '...'
                print(f"{i+1}. [{thread_num}] {title}")

    return pattern_groups, pattern_df

def tfidf_cluster_analysis(df, n_clusters=10):
    """Кластеризация тредов на основе TF-IDF векторизации"""
    print(f"Выполняется кластеризация на {n_clusters} групп...")

    # Предобработка заголовков
    titles = df['title'].apply(preprocess_text).tolist()

    # Векторизация текста с помощью TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=['и', 'в', 'на', 'с', 'по', 'за', 'из', 'к', 'у', 'от', 'для'],
        ngram_range=(1, 2)
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(titles)

    # Уменьшение размерности для визуализации
    svd = TruncatedSVD(n_components=2)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X_2d = lsa.fit_transform(tfidf_matrix)

    # Кластеризация
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(tfidf_matrix)

    # Визуализация кластеров
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df['cluster'], cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Кластер')
    plt.title('Кластеризация тредов на основе заголовков', fontsize=16)
    plt.xlabel('Компонента 1', fontsize=14)
    plt.ylabel('Компонента 2', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('stats/tfidf_clusters.png', dpi=300)
    plt.close()

    # Анализ кластеров
    cluster_groups = defaultdict(list)
    for i, row in df.iterrows():
        cluster_num = row['cluster']
        thread_num = row['thread_num']
        title = preprocess_text(row['title'])
        cluster_groups[cluster_num].append((thread_num, title))

    # Находим наиболее типичные заголовки для каждого кластера
    feature_names = tfidf_vectorizer.get_feature_names_out()
    cluster_keywords = {}

    for cluster_num, cluster_center in enumerate(kmeans.cluster_centers_):
        # Топ 10 слов для каждого кластера
        top_keywords_idx = cluster_center.argsort()[-10:]
        top_keywords = [feature_names[idx] for idx in top_keywords_idx]
        cluster_keywords[cluster_num] = top_keywords

    # Сохраняем результаты кластеризации
    cluster_data = []
    for cluster_num, threads in cluster_groups.items():
        keywords = ' '.join(cluster_keywords[cluster_num])
        for thread_num, title in threads:
            cluster_data.append({
                'thread_num': thread_num,
                'title': title,
                'cluster': cluster_num,
                'keywords': keywords
            })

    cluster_df = pd.DataFrame(cluster_data)
    cluster_df.to_csv('stats/cluster_groups.csv', index=False)

    # Выводим примеры для каждого кластера
    print("\nПримеры тредов для каждого кластера:")
    for cluster_num, threads in cluster_groups.items():
        if threads:
            keywords = ', '.join(cluster_keywords[cluster_num][-5:])  # Берем 5 самых значимых слов
            print(f"\nКЛАСТЕР {cluster_num} ({len(threads)} тредов):")
            print(f"Ключевые слова: {keywords}")
            for i, (thread_num, title) in enumerate(threads[:3]):
                if len(title) > 70:
                    title = title[:67] + '...'
                print(f"{i+1}. [{thread_num}] {title}")

    return cluster_groups, cluster_df

def find_semantic_duplicates(df, threshold=0.7):
    """Поиск семантических дубликатов среди заголовков"""
    print("Поиск семантически похожих тредов...")

    # Предобработка заголовков
    titles = df['title'].apply(preprocess_text).tolist()

    # Векторизация текста с помощью TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(titles)

    # Преобразуем в матрицу схожести
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Находим пары схожих тредов
    duplicate_pairs = []
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix)):
            if similarity_matrix[i, j] > threshold:
                duplicate_pairs.append((
                    df.iloc[i]['thread_num'],
                    df.iloc[i]['title'],
                    df.iloc[j]['thread_num'],
                    df.iloc[j]['title'],
                    similarity_matrix[i, j]
                ))

    # Сортируем по уровню схожести (от большего к меньшему)
    duplicate_pairs.sort(key=lambda x: x[4], reverse=True)

    # Сохраняем результаты
    if duplicate_pairs:
        duplicate_data = []
        for thread1_num, thread1_title, thread2_num, thread2_title, similarity in duplicate_pairs:
            duplicate_data.append({
                'thread1_num': thread1_num,
                'thread1_title': thread1_title,
                'thread2_num': thread2_num,
                'thread2_title': thread2_title,
                'similarity': similarity
            })

        duplicate_df = pd.DataFrame(duplicate_data)
        duplicate_df.to_csv('stats/similar_thread_pairs.csv', index=False)

        # Выводим примеры схожих пар
        print(f"\nНайдено {len(duplicate_pairs)} пар семантически похожих тредов (порог схожести: {threshold}):")
        for i, (thread1_num, thread1_title, thread2_num, thread2_title, similarity) in enumerate(duplicate_pairs[:10]):
            if len(thread1_title) > 50:
                thread1_title = thread1_title[:47] + '...'
            if len(thread2_title) > 50:
                thread2_title = thread2_title[:47] + '...'
            print(f"{i+1}. Схожесть: {similarity:.2f}")
            print(f"   [{thread1_num}] {thread1_title}")
            print(f"   [{thread2_num}] {thread2_title}")
    else:
        print("Семантические дубликаты не найдены.")

    return duplicate_pairs

def analyze_short_threads(df, max_length=50):
    """Анализ коротких заголовков тредов"""
    print(f"Анализ коротких заголовков (до {max_length} символов)...")

    # Фильтруем короткие заголовки
    short_titles_df = df[df['title'].apply(lambda x: len(str(x)) <= max_length)]

    if short_titles_df.empty:
        print("Короткие заголовки не найдены.")
        return []

    # Группировка коротких заголовков по шаблонам
    pattern_groups, _ = pattern_based_grouping(short_titles_df)

    # Кластеризация коротких заголовков
    n_clusters = min(5, len(short_titles_df))
    if n_clusters > 1:
        cluster_groups, _ = tfidf_cluster_analysis(short_titles_df, n_clusters=n_clusters)

    # Визуализация распределения длин коротких заголовков
    plt.figure(figsize=(12, 6))
    sns.histplot(short_titles_df['title'].apply(len), bins=max_length, kde=True)
    plt.title('Распределение длин коротких заголовков', fontsize=16)
    plt.xlabel('Длина заголовка (символы)', fontsize=14)
    plt.ylabel('Количество тредов', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('stats/short_titles_distribution.png', dpi=300)
    plt.close()

    # Создаем облако частых фраз
    from wordcloud import WordCloud

    all_short_titles = ' '.join(short_titles_df['title'].apply(preprocess_text))

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100
    ).generate(all_short_titles)

    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Облако слов из коротких заголовков', fontsize=16)
    plt.tight_layout()
    plt.savefig('stats/short_titles_wordcloud.png', dpi=300)
    plt.close()

    # Выводим список коротких заголовков
    print(f"\nНайдено {len(short_titles_df)} коротких заголовков (до {max_length} символов):")
    for i, (_, row) in enumerate(short_titles_df.head(20).iterrows()):
        print(f"{i+1}. [{row['thread_num']}] {row['title']}")

    return short_titles_df

def main():
    # Загружаем данные
    df = load_data()

    # Анализ на основе шаблонов
    print("\n--- АНАЛИЗ НА ОСНОВЕ ШАБЛОНОВ ---")
    pattern_groups, pattern_df = pattern_based_grouping(df)

    # Кластеризация
    print("\n--- КЛАСТЕРНЫЙ АНАЛИЗ ---")
    cluster_groups, cluster_df = tfidf_cluster_analysis(df)

    # Поиск семантических дубликатов
    print("\n--- ПОИСК СЕМАНТИЧЕСКИ ПОХОЖИХ ТРЕДОВ ---")
    duplicate_pairs = find_semantic_duplicates(df, threshold=0.75)

    # Анализ коротких заголовков
    print("\n--- АНАЛИЗ КОРОТКИХ ЗАГОЛОВКОВ ---")
    short_titles_df = analyze_short_threads(df, max_length=50)

    print("\nАнализ завершен! Результаты сохранены в папке 'stats'")

if __name__ == "__main__":
    main()
