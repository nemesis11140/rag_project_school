import os
from typing import List, Optional, Dict, Any
from config import DATA_DIR
from .document_loader import DocumentLoader
from .text_chunker import ParentChildChunker
from .vector_store import VectorStoreManager


class KnowledgeBase:
    """离线知识库：文档读取 -> 预处理 -> 递归切割 + 父子分块 -> bge-m3 向量化 -> Milvus。"""

    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.chunker = ParentChildChunker()
        self.document_loader = DocumentLoader()
        self._model_loaded = False

    def _ensure_model_loaded(self):
        if not self._model_loaded:
            self.vector_store.load_embedding_model()
            self._model_loaded = True

    def parse_document(self, file_name: str) -> Dict[str, Any]:
        """只做读取 -> 预处理 -> 切割，不写库。"""
        file_path = os.path.join(DATA_DIR, file_name)
        print(f"\n[KnowledgeBase] Parsing document: {file_name}")

        result = self.document_loader.load_document(file_path)
        if isinstance(result, tuple):
            text, page_data = result
        else:
            text, page_data = result, []
        print(f"[KnowledgeBase] Loaded {len(text)} characters from {len(page_data)} pages")

        chunks = self.chunker.split_text(text, document_name=file_name)
        parent_count = len({c["parent_index"] for c in chunks})
        print(f"[KnowledgeBase] Created {parent_count} parent / {len(chunks)} child chunks")

        return {
            "document_name": file_name,
            "total_chars": len(text),
            "total_pages": len(page_data),
            "parent_chunks": parent_count,
            "total_chunks": len(chunks),
            "chunks": chunks
        }

    def build_knowledge_base(self, file_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """把一份或多份文档构建进同一个集合。

        注意：一个集合只创建一次，避免处理多份文档时后一份把前一份清空。
        """
        file_names = file_names or self.get_available_documents()
        if not file_names:
            raise ValueError(f"{DATA_DIR} 下没有找到可处理的 PDF 文档")

        self._ensure_model_loaded()
        self.vector_store.create_collection(drop_existing=True)

        all_chunks: List[Dict[str, Any]] = []
        per_document = []

        for file_name in file_names:
            parsed = self.parse_document(file_name)
            all_chunks.extend(parsed.pop("chunks"))
            per_document.append(parsed)

        inserted = self.vector_store.insert_chunks(all_chunks)
        stats = self.vector_store.get_collection_stats()

        return {
            "documents": per_document,
            "total_chunks": len(all_chunks),
            "inserted": inserted,
            "stats": stats
        }

    def process_document(self, file_name: str, create_collection: bool = True) -> dict:
        """单文档增量写入（不删除已有集合）。"""
        self._ensure_model_loaded()
        parsed = self.parse_document(file_name)
        chunks = parsed.pop("chunks")

        if create_collection and not self.vector_store.has_collection():
            self.vector_store.create_collection(drop_existing=True)

        inserted = self.vector_store.insert_chunks(chunks)
        parsed["inserted"] = inserted
        parsed["stats"] = self.vector_store.get_collection_stats()
        return parsed

    def search(self, query: str, top_k: int = 5,
               document_name: Optional[str] = None) -> List[dict]:
        self._ensure_model_loaded()
        return self.vector_store.search(
            query=query, top_k=top_k, filter_document=document_name
        )

    def get_stats(self) -> dict:
        return self.vector_store.get_collection_stats()

    def has_collection(self) -> bool:
        return self.vector_store.has_collection()

    def get_available_documents(self) -> List[str]:
        documents = []
        if os.path.exists(DATA_DIR):
            for filename in sorted(os.listdir(DATA_DIR)):
                if filename.lower().endswith('.pdf'):
                    documents.append(filename)
        return documents
