import csv
import hashlib
import time
import datetime
import difflib
import json
from collections import defaultdict, Counter
import os

import const

from dvach import dvach


def normalize(text):
    return text.lower().strip()

def get_board_threads(board_name='b'):
    board = dvach.Board.from_json(dvach.Board.json_download(board_name))
    board.update_threads()
    board.sort_threads_by_posts()
    return board.threads

def extract_features(title):
    triggers = ['анон', 'пикрил', 'согласны', 'ноу дискас']
    title_lower = title.lower()
    return {
        'title_len': len(title),
        'question_marks': title.count('?'),
        'anon_triggered': int(any(t in title_lower for t in triggers))
    }

def snapshot_board(board='b'):
    threads = get_board_threads(board)
    now = datetime.datetime.now()
    snapshot_time = now.strftime('%Y-%m-%d %H:%M')

    updated_data = {}

    for [_, thread] in threads.items():
        try:
            thread.update_posts()
        except Exception:
            continue

        op_post = thread.get_op_post.comment
        subject = thread.subject or ""
        thread_num = str(thread.num)
        posts_count = thread.posts_count
        posts = thread.posts
        created_dt = datetime.datetime.fromtimestamp(thread.timestamp).strftime('%Y-%m-%d %H:%M:%S')

        op_post_count = sum(post.op for post in posts)
        op_reply_ratio = round(op_post_count / posts_count, 4) if posts_count > 0 else 0

        title = normalize(op_post)
        features = extract_features(title)

        updated_data[thread_num] = {
            'datetime': snapshot_time,
            'created_dt': created_dt,
            'last_seen': snapshot_time,
            'thread_num': thread_num,
            'title': title,
            'subtitle': subject,
            'posts_count': posts_count,
            'op_post_count': op_post_count,
            'op_reply_ratio': op_reply_ratio,
            **features  # 15 минут в секундах
        }

        print(f'save thread {thread_num}, op_reply_ratio: {op_reply_ratio}')

        json_dir = 'threads'
        os.makedirs(json_dir, exist_ok=True)
        thread_path = os.path.join(json_dir, f"{thread_num}.json")
        with open(thread_path, 'w', encoding='utf-8') as f:
            json.dump(json.loads(thread.json_download()), f, ensure_ascii=False, indent=2)

    existing_data = {}
    if os.path.exists(const.CSV_FILE):
        with open(const.CSV_FILE, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_data[row['thread_num']] = row

    existing_data.update(updated_data)

    with open(const.CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=const.FIELDNAMES)
        writer.writeheader()
        for row in existing_data.values():
            writer.writerow(row)

    print(f"[{snapshot_time}] Snapshot complete: {len(updated_data)} threads updated")


if __name__ == '__main__':
    while True:
        try:
            snapshot_board('b')
        except Exception as e:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: {e}")
        time.sleep(15 * 60)
