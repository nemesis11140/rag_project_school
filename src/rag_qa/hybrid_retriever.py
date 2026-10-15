"""混合检索 + 双路召回。

语义路：bge-m3 向量 -> Milvus 向量检索
关键词路：jieba 分词 -> 内存 BM25

两路结果用 RRF（Reciprocal Rank Fusion）融合。不用加权求和的原因：
BM25 分数无界且依赖语料规模，向量余弦相似度取值区间也不同，
直接加权需要拍一个魔法系数，而 RRF 只用排名，天然免疫量纲问题。
"""
from typing import Dict, Any, List, Optional
import threading

import numpy as np
from rank_bm25 import BM25Okapi

from config import SYSTEM_CONFIG
from ..common.text_utils import tokenize
from ..common.startup import StartupIssue
from ..knowledge_base.vector_store import VectorStoreManager

# RRF 平滑常数，经验值 60（与原始 RRF 论文一致）
RRF_K = 60


class HybridRetriever:
    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.vector_store = vector_store or VectorStoreManager()
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[Dict[str, Any]] = []
        self._initialized = False
        self._init_lock = threading.Lock()
        self.recall_multiplier = SYSTEM_CONFIG.get("recall_multiplier", 3)

    # ---------- 初始化 ----------

    def _ensure_initialized(self):
        with self._init_lock:
            if not self._initialized:
                self._initialize()

    def _initialize(self):
        if self._initialized:
            return

        print("[HybridRetriever] Initializing...")
        self.vector_store.check_ready()
        self.vector_store.load_embedding_model()

        if not self.vector_store.has_collection():
            raise StartupIssue("knowledge_collection_missing", "milvus", "知识库集合不存在",
                               "请先恢复 Milvus 备份，再重试预热。")

        self._check_schema()
        self.vector_store._ensure_collection()
        self.vector_store.load()
        self._load_corpus()
        if not self.corpus:
            raise StartupIssue("knowledge_collection_empty", "milvus", "知识库中没有可读取的文档",
                               "请检查向量备份恢复是否完成。")
        self._build_bm25_index()

        self._initialized = True
        print(
            f"[HybridRetriever] Ready. dense={self.dense_available} "
            f"bm25={self.bm25 is not None} corpus={len(self.corpus)}"
        )

    def _check_schema(self):
        """启动自检：字段缺失会导致 dense 检索静默失效，这里直接报错。"""
        fields = self.vector_store.get_schema_fields()
        if not fields:
            return
        required = {"child_text", "parent_text", "document_name",
                    "parent_index", "child_index", "vector"}
        missing = required - set(fields)
        if missing:
            raise RuntimeError(
                f"Milvus 集合字段与代码不一致，缺少 {sorted(missing)}（现有 {sorted(fields)}）。"
                "请删除集合后重新运行 scripts/build_knowledge_base.py 重建索引。"
            )

    @property
    def dense_available(self) -> bool:
        return self.vector_store.embedding_model is not None

    def _load_corpus(self):
        rows = self.vector_store.query_all()
        self.corpus = rows
        print(f"[HybridRetriever] Loaded {len(rows)} chunks for BM25")

    def _build_bm25_index(self):
        if not self.corpus:
            self.bm25 = None
            return
        tokenized = [tokenize(row.get("child_text", "")) for row in self.corpus]
        if not any(tokenized):
            self.bm25 = None
            return
        self.bm25 = BM25Okapi(tokenized)
        print(f"[HybridRetriever] BM25 index built with {len(tokenized)} chunks")

    # ---------- 单路召回 ----------

    def _dense_search(self, query: str, top_k: int,
                      filter_document: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            return self.vector_store.search(
                query=query, top_k=top_k, filter_document=filter_document
            )
        except Exception as e:
            # 不再静默吞掉：dense 路失效是核心链路事故，必须可见
            print(f"[HybridRetriever] Dense search FAILED: {e}")
            return []

    def _sparse_search(self, query: str, top_k: int,
                       filter_document: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        # 只保留有分的候选，避免 0 分文档靠排名挤进融合结果
        candidate_idx = np.argsort(scores)[::-1]
        results = []
        for idx in candidate_idx:
            if len(results) >= top_k:
                break
            score = float(scores[idx])
            if score <= 0:
                break
            if idx >= len(self.corpus):
                continue
            row = self.corpus[idx]
            if filter_document and row.get("document_name") != filter_document:
                continue
            results.append({
                "child_text": row.get("child_text", ""),
                "parent_text": row.get("parent_text", ""),
                "document_name": row.get("document_name", ""),
                "parent_index": row.get("parent_index", 0),
                "child_index": row.get("child_index", 0),
                "sparse_score": score,
                "source": "sparse",
            })
        return results

    # ---------- 融合 ----------

    @staticmethod
    def _doc_key(doc: Dict[str, Any]):
        return (doc.get("document_name", ""),
                doc.get("parent_index", 0),
                doc.get("child_index", 0))

    def _rrf_fuse(self, ranked_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """RRF：score(d) = Σ 1 / (k + rank_i(d))，rank 从 1 开始。"""
        best: Dict[Any, Dict[str, Any]] = {}
        rrf_scores: Dict[Any, float] = {}

        for docs in ranked_lists:
            for rank, doc in enumerate(docs, start=1):
                key = self._doc_key(doc)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
                entry = best.get(key)
                if entry is None:
                    best[key] = dict(doc)
                else:
                    # 合并两路信息
                    entry["dense_score"] = max(
                        entry.get("dense_score", 0.0), doc.get("dense_score", 0.0)
                    )
                    entry["sparse_score"] = max(
                        entry.get("sparse_score", 0.0), doc.get("sparse_score", 0.0)
                    )
                    if entry.get("source") != doc.get("source"):
                        entry["source"] = "hybrid"
                    if not entry.get("parent_text"):
                        entry["parent_text"] = doc.get("parent_text", "")

        fused = []
        for key, doc in best.items():
            doc.setdefault("dense_score", 0.0)
            doc.setdefault("sparse_score", 0.0)
            doc["rrf_score"] = rrf_scores[key]
            doc["final_score"] = rrf_scores[key]
            fused.append(doc)

        fused.sort(key=lambda d: d["rrf_score"], reverse=True)
        return fused

    @staticmethod
    def _deduplicate_by_parent(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """同一父块只保留排名最高的子块，避免同一段内容占满上下文。"""
        seen = set()
        deduped = []
        for doc in results:
            key = (doc.get("document_name", ""), doc.get("parent_index", 0))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)
        return deduped

    # ---------- 对外接口 ----------

    def retrieve(self, queries: List[str], top_k: int = 5,
                 filter_document: Optional[str] = None) -> List[Dict[str, Any]]:
        self._ensure_initialized()

        queries = [q for q in (queries or []) if q and q.strip()]
        if not queries:
            return []

        recall_k = max(top_k * self.recall_multiplier, top_k)
        print(f"[HybridRetriever] Retrieving for {len(queries)} queries, recall_k={recall_k}")

        ranked_lists: List[List[Dict[str, Any]]] = []
        for query in queries:
            dense = self._dense_search(query, recall_k, filter_document)
            sparse = self._sparse_search(query, recall_k, filter_document)
            print(f"[HybridRetriever]   q='{query[:30]}' dense={len(dense)} sparse={len(sparse)}")
            if dense:
                ranked_lists.append(dense)
            if sparse:
                ranked_lists.append(sparse)

        if not ranked_lists:
            print("[HybridRetriever] No candidates from either channel")
            return []

        fused = self._rrf_fuse(ranked_lists)
        deduped = self._deduplicate_by_parent(fused)
        print(f"[HybridRetriever] fused={len(fused)} -> deduped={len(deduped)}")
        return deduped[:top_k]

    def retrieve_single(self, query: str, top_k: int = 5,
                        filter_document: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.retrieve([query], top_k, filter_document)
