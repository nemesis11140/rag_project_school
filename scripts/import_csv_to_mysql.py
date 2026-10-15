"""把 data/高频问答对.csv 导入 MySQL（批量单事务）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import DATA_DIR
from src.mysql_qa import MySQLManager


def main():
    print("=" * 60)
    print("导入高频问答 CSV 到 MySQL")
    print("=" * 60)

    csv_path = os.path.join(DATA_DIR, "高频问答对.csv")
    if not os.path.exists(csv_path):
        print(f"错误：CSV 文件不存在: {csv_path}")
        return 1

    print(f"\n[Step 1] 读取 CSV: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"  读取 {len(df)} 条记录，列名: {list(df.columns)}")

    if '问题' not in df.columns or '答案' not in df.columns:
        print("  错误：CSV 必须包含 '问题' 和 '答案' 两列")
        return 1

    print("\n[Step 2] 创建数据库和表...")
    mysql_manager = MySQLManager()
    mysql_manager.create_table_if_not_exists()

    print("\n[Step 3] 清空旧数据并批量导入...")
    mysql_manager.truncate_qa_pairs()

    pairs, seen = [], set()
    for _, row in df.iterrows():
        question = str(row['问题']).strip()
        answer = str(row['答案']).strip()
        if not question or not answer or question.lower() == 'nan':
            continue
        if question in seen:
            print(f"  [跳过] 重复问题: {question}")
            continue
        seen.add(question)
        pairs.append((question, answer))

    inserted = mysql_manager.insert_qa_pairs(pairs)
    print(f"  成功写入 {inserted} 条，跳过 {len(df) - len(pairs)} 条")

    print("\n[Step 4] 验证...")
    all_qa = mysql_manager.get_all_qa_pairs()
    print(f"  MySQL 中共有 {len(all_qa)} 条问答数据")

    print("\n" + "=" * 60)
    print("导入完成！请接着运行: python scripts/init_redis.py")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
