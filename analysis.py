import const
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style='darkgrid', font_scale=1.1)

min_posts_count = 10

df = pd.read_csv(const.CSV_FILE, parse_dates=['datetime'])
df.drop_duplicates(subset=['datetime', 'thread_num'], inplace=True)

df['op_reply_ratio'] = pd.to_numeric(df['op_reply_ratio'], errors='coerce')
df = df.dropna(subset=['op_reply_ratio'])
df = df[df['posts_count'] >= min_posts_count]

df['op_reply_percent'] = np.ceil(df['op_reply_ratio'] * 100)

zero_percent_count = df[df['op_reply_percent'] == 0].shape[0]
nonzero_percent_count = df[df['op_reply_percent'] > 0][df['op_reply_percent'] <= 10]

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
