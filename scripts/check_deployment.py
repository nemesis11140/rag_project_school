"""启动前检查知识库数据和模型挂载，不加载大模型或创建数据库。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import EMBEDDING_CONFIG, RERANKER_CONFIG
from src.common.startup import StartupIssue, check_model_files
from src.knowledge_base.vector_store import VectorStoreManager


def main():
    try:
        VectorStoreManager().check_ready()
        check_model_files(EMBEDDING_CONFIG["model_path"], "向量")
        check_model_files(RERANKER_CONFIG["model_path"], "精排")
    except StartupIssue as issue:
        print(f"部署检查失败 [{issue.code}]：{issue.message}\n{issue.hint}", flush=True)
        return 1
    print("部署检查通过：知识库存在且非空，向量与精排模型文件可见。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
