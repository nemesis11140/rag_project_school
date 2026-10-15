"""从 MySQL 重建 Redis 缓存（问题列表 + 分词结果 + 归一化问题索引）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.text_utils import normalize
from src.mysql_qa import BM25Retriever, MySQLManager, RedisCache


def main():
    print("=" * 60)
    print("初始化 Redis 缓存")
    print("=" * 60)

    mysql_manager = MySQLManager()
    redis_cache = RedisCache()
    retriever = BM25Retriever()

    print("\n[Step 1] 从 MySQL 加载问答数据...")
    qa_pairs = mysql_manager.get_all_qa_pairs()
    print(f"  共加载 {len(qa_pairs)} 条问答数据")
    if not qa_pairs:
        print("  警告：MySQL 中没有数据，请先运行 scripts/import_csv_to_mysql.py")
        return 1

    print("\n[Step 2] 归一化 + 分词...")
    questions = [pair["question"] for pair in qa_pairs]
    ids = [pair["id"] for pair in qa_pairs]
    tokenized = [retriever.tokenize(q) for q in questions]

    entries = []
    seen = {}
    for idx, question in enumerate(questions):
        normalized = normalize(question)
        if not normalized:
            continue
        if normalized in seen:
            print(f"  [警告] 归一化后重复：'{question}' 与 '{questions[seen[normalized]]}'")
            continue
        seen[normalized] = idx
        entries.append({
            "normalized": normalized,
            "question": question,
            "id": ids[idx]
        })
    print(f"  归一化后得到 {len(entries)} 个唯一问题")

    print("\n[Step 3] 写入 Redis...")
    redis_cache.set_questions_cache(questions, tokenized, ids)
    redis_cache.set_qa_index(entries)

    # 阈值/归一化策略变化后，旧的答案缓存可能是在更松的判定下写进去的，
    # 会在精确匹配之前先命中，导致绕过新阈值。重建时一并清掉。
    cleared = redis_cache.clear_cache()
    print(f"  已清空旧答案缓存: {cleared}")

    print("\n[Step 4] 验证缓存...")
    stats = redis_cache.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Redis 缓存初始化完成！")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
