"""离线知识库构建：文档读取 -> 预处理 -> 递归切割 + 父子分块 -> bge-m3 向量化 -> Milvus。

用法：
    python scripts/build_knowledge_base.py              # 处理 data/ 下全部 PDF
    python scripts/build_knowledge_base.py 2025学生手册.pdf

注意：**会删除并重建集合**，已有索引会被覆盖。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHUNKING_CONFIG, MILVUS_CONFIG
from src.knowledge_base import KnowledgeBase


def main():
    file_names = sys.argv[1:] or None

    print("=" * 70)
    print("离线知识库构建")
    print("=" * 70)
    print(f"  Milvus:   {MILVUS_CONFIG['host']}:{MILVUS_CONFIG['port']}"
          f" / {MILVUS_CONFIG['database_name']}.{MILVUS_CONFIG['collection_name']}")
    print(f"  父块:     {CHUNKING_CONFIG['parent_chunk_size']} 字"
          f" (overlap {CHUNKING_CONFIG['parent_chunk_overlap']})")
    print(f"  子块:     {CHUNKING_CONFIG['child_chunk_size']} 字"
          f" (overlap {CHUNKING_CONFIG['child_chunk_overlap']})")
    print()

    knowledge_base = KnowledgeBase()

    documents = file_names or knowledge_base.get_available_documents()
    print(f"待处理文档: {documents}\n")

    result = knowledge_base.build_knowledge_base(documents)

    print("\n" + "=" * 70)
    print("构建结果")
    print("=" * 70)
    for item in result["documents"]:
        print(f"  {item['document_name']}")
        print(f"    字符数: {item['total_chars']}  页数: {item['total_pages']}")
        print(f"    父块: {item['parent_chunks']}  子块: {item['total_chunks']}")
    print(f"\n  合计写入: {result['inserted']} 条")
    print(f"  集合行数: {result['stats']['row_count']}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
