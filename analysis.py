import const
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style='darkgrid', font_scale=1.1)

min_posts_count = 100

df = pd.read_csv(const.CSV_FILE, parse_dates=['datetime'])
df.drop_duplicates(subset=['datetime', 'thread_num'], inplace=True)

df['op_reply_ratio'] = pd.to_numeric(df['op_reply_ratio'], errors='coerce')
df = df.dropna(subset=['op_reply_ratio'])
df = df[df['posts_count'] >= min_posts_count]

df['op_reply_percent'] = np.ceil(df['op_reply_ratio'] * 100)
df['op_replied'] = np.where(df['op_reply_ratio'] > 0, 1, 0)

fifteen_min_op_reply = df.resample('15T', on='datetime')['op_replied'].mean()

zero_percent_count = df[df['op_reply_percent'] == 0].shape[0]

bins = [0, 0.1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
labels = ['0%', '1-10%', '11-20%', '21-30%', '31-40%', '41-50%', 
          '51-60%', '61-70%', '71-80%', '81-90%', '91-100%']
df['op_reply_category'] = pd.cut(df['op_reply_percent'], bins=bins, labels=labels)


category_counts = df['op_reply_category'].value_counts().sort_index()
category_counts['0%'] = zero_percent_count

total_threads = category_counts.sum()
category_ratios = category_counts / total_threads


plt.figure(figsize=(14, 6))
sns.barplot(x=category_ratios.index, y=category_ratios.values, palette='pastel')

plt.title('Доля ответов ОП-а в своих тредах')
plt.xlabel('Процент ответов от ОП-а к общему колву постов')
plt.ylabel('Доля от общего числа постов')
plt.xticks(rotation=45)

plt.gca().set_xticklabels([f'{label}' for label in category_ratios.index])

plt.grid(True)
plt.tight_layout()

plt.show()
plt.savefig('stats2.png')



plt.figure(figsize=(14, 6))

fifteen_min_op_reply.plot(marker='o', linestyle='-')

plt.title('Процент тредов с ответами от ОП по неделям')
plt.xlabel('Неделя')
plt.ylabel('Процент тредов с ответами ОП')
plt.xticks(rotation=45)

plt.grid(True)
plt.tight_layout()

plt.show()
plt.savefig('weekly_op_reply.png')

# Преобразуем created_dt в datetime, если он еще не в формате datetime
df['created_dt'] = pd.to_datetime(df['created_dt'], errors='coerce')
df = df.dropna(subset=['created_dt'])

# Округляем время создания до часа
df['created_hour'] = df['created_dt'].dt.floor('H')

# Группировка по часу — сколько тредов создано в каждый час
threads_per_hour = df.groupby('created_hour').size()

# Построение графика
plt.figure(figsize=(14, 6))
threads_per_hour.plot(kind='line', marker='o', color='teal')

plt.title('Скорость создания тредов (в час)')
plt.xlabel('Дата и час')
plt.ylabel('Количество новых тредов')
plt.xticks(rotation=45)

plt.grid(True)
plt.tight_layout()

plt.show()
plt.savefig('thread_creation_rate.png')
