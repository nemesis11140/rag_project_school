"""向后兼容的转发模块。

单例实现已下沉到 src/common/embedding.py（用于打破 knowledge_base 与
rag_qa 之间的循环导入），这里仅做转发，避免旧引用失效。
"""
from ..common.embedding import EmbeddingModelSingleton, embedding_model_singleton

__all__ = ["EmbeddingModelSingleton", "embedding_model_singleton"]
