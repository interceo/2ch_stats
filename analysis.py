import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from datetime import datetime

import const

sns.set(style='whitegrid', font_scale=1.1)

# === Загрузка данных ===
df = pd.read_csv(const.CSV_FILE, parse_dates=['datetime'])
df.drop_duplicates(subset=['datetime', 'thread_num'], inplace=True)

# === Предобработка ===
df['hour'] = df['datetime'].dt.hour
df['date'] = df['datetime'].dt.date
df['title_clean'] = df['title'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
df['op_reply_ratio'] = pd.to_numeric(df['op_reply_ratio'], errors='coerce')
df = df.dropna(subset=['op_reply_ratio'])
df = df[df['posts_count'] >= 10]

def cluster_titles(titles, num_clusters=10):
    # Удалим строки, где нет текста
    titles = titles.dropna()
    titles = titles[titles.str.strip().astype(bool)]

    num_titles = len(titles)
    if num_titles == 0:
        raise ValueError("Нет валидных заголовков для кластеризации.")

    min_df_value = 2 if num_titles > 20 else 1

    vectorizer = TfidfVectorizer(
        token_pattern=r'\b\w{3,}\b',
        lowercase=True,
        max_df=0.95,
        min_df=min_df_value
    )
    X = vectorizer.fit_transform(titles)
    model = KMeans(n_clusters=min(num_clusters, num_titles), random_state=42, n_init='auto')
    labels = model.fit_predict(X)

    return labels, model, vectorizer


valid_titles = df['title_clean'].dropna()
valid_titles = valid_titles[valid_titles.str.strip().astype(bool)]

cluster_labels, kmeans_model, tfidf_vectorizer = cluster_titles(valid_titles)

df.loc[valid_titles.index, 'cluster'] = cluster_labels


# === Топовые кластеры ===
top_clusters = df['cluster'].value_counts().head(5)
print("\n📌 Топ-5 кластеров (ботоподобные шаблоны):")
for cluster_id in top_clusters.index:
    print(f"\nCluster {cluster_id} (size: {top_clusters[cluster_id]})")
    sample_titles = df[df['cluster'] == cluster_id]['title'].value_counts().head(5)
    print(sample_titles.to_string())

# === Визуализация: распределение кластеров по времени ===
cluster_hour = df.groupby(['hour', 'cluster']).size().unstack(fill_value=0)


df['op_reply_percent'] = df['op_reply_ratio'] * 100


sns.set(style="whitegrid")

plt.figure(figsize=(12,6))

sns.kdeplot(df['op_reply_percent'], fill=True, color='teal', bw_adjust=0.5)

plt.title('Гладкое распределение процентов ответов ОП в тредах (KDE)')
plt.xlabel('Процент ответов от ОП (%)')
plt.ylabel('Плотность вероятности')

plt.xlim(0, 100)
plt.grid(True)
plt.tight_layout()

plt.show()