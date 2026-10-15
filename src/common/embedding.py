"""bge-m3 向量化模型单例。

放在 common 层是为了打破循环依赖：
knowledge_base 与 rag_qa 都依赖它，而它不依赖任何业务包。
"""
import os
import threading

from config import EMBEDDING_CONFIG


class EmbeddingModelSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    def get_model(self, model_path: str = None, device: str = None):
        """获取（必要时加载）bge-m3 模型。多次调用只加载一次。"""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            from sentence_transformers import SentenceTransformer

            if model_path is None:
                model_path = EMBEDDING_CONFIG.get("model_path")
            if not model_path or not os.path.exists(model_path):
                model_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "model", "bge-m3"
                )
            if device is None:
                device = EMBEDDING_CONFIG.get("device", "cpu")

            print(f"[EmbeddingModel] Loading embedding model from {model_path} on {device}")
            self._model = SentenceTransformer(model_path, device=device)
            print(
                "[EmbeddingModel] Model loaded. Dimension: "
                f"{self._model.get_sentence_embedding_dimension()}"
            )
            return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def clear_model(self):
        with self._lock:
            self._model = None
            print("[EmbeddingModel] Model cleared")


embedding_model_singleton = EmbeddingModelSingleton()
