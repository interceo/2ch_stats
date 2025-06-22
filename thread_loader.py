import os
import json
import sys
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import re
from bs4 import BeautifulSoup

# Добавляем путь к модулю dvach
sys.path.append('./dvach')
from dvach import Thread, Post, Post_file

class ThreadDataLoader:
    """Класс для загрузки и обработки данных тредов из JSON файлов"""

    def __init__(self, threads_dir: str = 'threads'):
        self.threads_dir = threads_dir
        self.threads_data = []

    def load_all_threads(self, max_threads: int = None) -> pd.DataFrame:
        """Загружает треды из папки и возвращает DataFrame с аналитическими данными

        Args:
            max_threads: Максимальное количество тредов для загрузки. Если None, загружает все.
        """
        print(f"Загрузка тредов из папки {self.threads_dir}...")

        if not os.path.exists(self.threads_dir):
            raise FileNotFoundError(f"Папка {self.threads_dir} не найдена")

        json_files = [f for f in os.listdir(self.threads_dir) if f.endswith('.json')]
        print(f"Найдено {len(json_files)} JSON файлов")

        # Ограничиваем количество файлов, если задан лимит
        if max_threads is not None and max_threads > 0:
            json_files = json_files[:max_threads]
            print(f"Ограничение: будет обработано только {len(json_files)} файлов")

        threads_data = []
        failed_count = 0

        for i, json_file in enumerate(json_files, 1):
            try:
                if max_threads and i % 10 == 0:  # Показываем прогресс каждые 10 файлов
                    print(f"Обработано {i}/{len(json_files)} файлов...")

                file_path = os.path.join(self.threads_dir, json_file)
                thread_data = self._process_thread_file(file_path)
                if thread_data:
                    threads_data.append(thread_data)
            except Exception as e:
                print(f"Ошибка при обработке файла {json_file}: {e}")
                failed_count += 1
                continue

        print(f"Успешно обработано {len(threads_data)} тредов, ошибок: {failed_count}")

        if not threads_data:
            raise ValueError("Не удалось загрузить ни одного треда")

        df = pd.DataFrame(threads_data)
        self._add_derived_metrics(df)

        return df

    def _process_thread_file(self, file_path: str) -> Dict[str, Any]:
        """Обрабатывает один JSON файл треда и извлекает аналитические данные используя API dvach"""
        # Извлекаем номер треда из имени файла
        thread_num = os.path.basename(file_path).replace('.json', '')

        try:
            # Читаем JSON содержимое файла
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Извлекаем базовую информацию
            title = json_data.get('title', '')
            posts_count = json_data.get('posts_count', 0)
            unique_posters = json_data.get('unique_posters', 0)
            files_count = json_data.get('files_count', 0)

            # Создаем объект Thread и правильно инициализируем posts
            thread = Thread('b')
            thread.posts = []  # Инициализируем пустой список постов для этого экземпляра
            thread.num = thread_num

            # Используем API dvach для парсинга постов
            with open(file_path, 'r', encoding='utf-8') as f:
                json_content = f.read()
            thread.get_posts(json_content)

        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")
            return None

        if not thread.posts:
            return None

        # Анализируем ОП-пост используя API dvach
        op_post = thread.get_op_post
        op_comment = op_post.comment  # уже очищен от HTML в классе Post
        op_comment_length = len(op_comment)
        op_files_count = len(op_post.files) if op_post.files else 0

        # Временные данные из ОП-поста - используем timestamp из JSON данных
        op_timestamp = None
        if 'threads' in json_data and json_data['threads'] and 'posts' in json_data['threads'][0]:
            op_post_raw = json_data['threads'][0]['posts'][0]
            op_timestamp = op_post_raw.get('timestamp')

        op_date = None
        if op_timestamp:
            try:
                op_date = datetime.fromtimestamp(op_timestamp)
            except (ValueError, OSError):
                op_date = None

        # Подсчитываем ответы ОП
        op_replies_count = sum(1 for post in thread.posts[1:] if post.num == op_post.num)
        op_reply_ratio = op_replies_count / max(1, posts_count - 1) if posts_count > 1 else 0

        # Анализ всех постов
        total_files = sum(len(post.files) if post.files else 0 for post in thread.posts)
        avg_post_length = sum(len(post.comment) for post in thread.posts) / max(1, len(thread.posts))

        # Подсчитываем вопросительные знаки в ОП
        question_marks = op_comment.count('?')

        # Анализ типов файлов
        file_types = []
        for post in thread.posts:
            if post.files:
                for file in post.files:
                    if hasattr(file, 'name') and '.' in file.name:
                        ext = file.name.split('.')[-1].lower()
                        file_types.append(ext)

        # Временной анализ активности треда - используем timestamp из JSON
        post_timestamps = []
        if 'threads' in json_data and json_data['threads'] and 'posts' in json_data['threads'][0]:
            for post_raw in json_data['threads'][0]['posts']:
                timestamp = post_raw.get('timestamp')
                if timestamp:
                    try:
                        post_timestamps.append(datetime.fromtimestamp(timestamp))
                    except (ValueError, OSError):
                        continue

        thread_duration_hours = 0
        if len(post_timestamps) > 1:
            thread_duration_hours = (max(post_timestamps) - min(post_timestamps)).total_seconds() / 3600

        # Формируем результирующий словарь
        result = {
            'thread_num': thread_num,
            'title': title,
            'title_len': len(title),
            'posts_count': posts_count,
            'unique_posters': unique_posters,
            'files_count': files_count,
            'total_files': total_files,

            # ОП-пост данные
            'op_comment': op_comment,
            'op_comment_length': op_comment_length,
            'op_files_count': op_files_count,
            'op_reply_ratio': op_reply_ratio,
            'op_replies_count': op_replies_count,
            'question_marks': question_marks,

            # Временные данные
            'created_dt': op_date,
            'thread_duration_hours': thread_duration_hours,

            # Аналитические метрики
            'avg_post_length': avg_post_length,
            'files_per_post': total_files / max(1, posts_count),
            'unique_posters_ratio': unique_posters / max(1, posts_count),

            # Типы файлов (топ-3)
            'top_file_types': self._get_top_file_types(file_types, 3),

            # Файл откуда загружен
            'source_file': os.path.basename(file_path)
        }

        return result

    def _clean_html_text(self, text: str) -> str:
        """Очищает текст от HTML тегов"""
        if not text:
            return ""

        # Удаляем HTML теги
        soup = BeautifulSoup(text, 'html.parser')
        clean_text = soup.get_text()

        # Удаляем лишние пробелы
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text

    def _get_top_file_types(self, file_types: List[str], top_n: int = 3) -> str:
        """Возвращает строку с топ-N типами файлов"""
        if not file_types:
            return ""

        from collections import Counter
        top_types = Counter(file_types).most_common(top_n)
        return ', '.join([f"{ext}({count})" for ext, count in top_types])

    def _add_derived_metrics(self, df: pd.DataFrame):
        """Добавляет производные метрики к DataFrame"""

        # Категории по длине заголовка
        df['title_category'] = pd.cut(df['title_len'],
                                    bins=[0, 50, 150, 500, float('inf')],
                                    labels=['Короткий', 'Средний', 'Длинный', 'Очень длинный'])

        # Категории по количеству постов
        df['posts_category'] = pd.cut(df['posts_count'],
                                    bins=[0, 10, 50, 200, float('inf')],
                                    labels=['Мало постов', 'Средне постов', 'Много постов', 'Очень много постов'])

        # Категории по активности ОП
        df['op_activity'] = df['op_reply_ratio'].apply(
            lambda x: 'Неактивный' if x == 0 else
                     'Низкая активность' if x < 0.1 else
                     'Средняя активность' if x < 0.3 else
                     'Высокая активность'
        )

        # Наличие файлов в треде
        df['has_files'] = df['total_files'] > 0
        df['has_op_files'] = df['op_files_count'] > 0

        # Категории по времени жизни треда
        df['duration_category'] = pd.cut(df['thread_duration_hours'],
                                       bins=[0, 1, 12, 72, float('inf')],
                                       labels=['Короткая жизнь', 'Средняя жизнь', 'Долгая жизнь', 'Очень долгая жизнь'])

        # Час и день недели создания (если доступно)
        if df['created_dt'].notna().any():
            df['created_hour'] = df['created_dt'].dt.hour
            df['created_day'] = df['created_dt'].dt.dayofweek
            df['created_day_name'] = df['created_dt'].dt.day_name()

        print(f"Добавлены производные метрики для {len(df)} тредов")

def test_loader():
    """Тестовая функция для проверки загрузчика"""
    loader = ThreadDataLoader()
    try:
        df = loader.load_all_threads()
        print(f"\nУспешно загружено {len(df)} тредов")
        print("\nПервые 5 записей:")
        print(df[['thread_num', 'title', 'posts_count', 'unique_posters', 'op_comment_length']].head())
        print(f"\nОсновная статистика:")
        print(f"Средняя длина заголовка: {df['title_len'].mean():.1f}")
        print(f"Среднее количество постов: {df['posts_count'].mean():.1f}")
        print(f"Среднее количество уникальных пользователей: {df['unique_posters'].mean():.1f}")
        return df
    except Exception as e:
        print(f"Ошибка при тестировании: {e}")
        return None

if __name__ == "__main__":
    test_loader()
