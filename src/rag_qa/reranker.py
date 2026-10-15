"""精排：bge-reranker（CrossEncoder），具体用哪个由 config 的 reranker.model_path 决定。

要点：
  - 所有 (query, doc) 对一次性批量前向，不再逐条循环调用模型
  - 显式指定 device，走 inference_mode
  - 精排单元是子块（child_text）——它才是召回单元；父块只用于喂给 LLM
  - 模型不可用时退化为词面重叠 + RRF 分的启发式排序，服务不中断
"""
import os
import threading
from typing import Any, Dict, List, Optional

from config import RERANKER_CONFIG
from ..common.text_utils import keyword_overlap


class Reranker:
    def __init__(self):
        self.model_path = RERANKER_CONFIG["model_path"]
        self.device = RERANKER_CONFIG.get("device", "cpu")
        self.batch_size = RERANKER_CONFIG.get("batch_size", 16)
        self.max_length = RERANKER_CONFIG.get("max_length", 512)
        self.max_docs = RERANKER_CONFIG.get("max_docs", 8)
        self.model = None
        self.tokenizer = None
        self.torch = None
        self._initialized = False
        self.last_inference_error = None
        self._load_lock = threading.Lock()

    # ---------- 初始化 ----------

    def _ensure_initialized(self):
        # 无条件拿锁，不能用"锁外看一眼 _initialized 就返回"的双重检查：
        # _initialized 在真正加载之前就置位了，后到者会看到它已是 True
        # 却 model=None，于是静默退回启发式排序——不报错，但排序质量悄悄变差。
        with self._load_lock:
            if not self._initialized:
                self._initialized = True
                self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[Reranker] 模型目录不存在: {self.model_path}，使用启发式排序")
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            print(f"[Reranker] Loading {self.model_path} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            self.torch = torch
            print("[Reranker] Reranker model loaded")
        except Exception as e:
            print(f"[Reranker] 加载失败，使用启发式排序: {e}")
            self.model = None

    @property
    def available(self) -> bool:
        self._ensure_initialized()
        return self.model is not None

    @property
    def is_loaded(self) -> bool:
        """只读状态，不触发加载（供 /api/health 用）。"""
        return self.model is not None

    def warmup(self):
        """加载模型并跑一次真实形状的批。

        模型加载 ≠ 就绪：torch 首次前向还要初始化 OMP/MKL 线程池，
        实测精排首次 6.5s、第二次 0.8s，差的那几秒只有真跑一次才能消掉。
        """
        # 修复文件挂载等问题后允许重试，失败不应永久锁定为启发式模式。
        with self._load_lock:
            if self.model is None:
                self._initialized = False
        self._ensure_initialized()
        if self.model is None:
            return
        # 条数与文本长度都按线上形状来：max_docs 条、子块量级（约 300 字）
        dummy = [{"child_text": "预热文本" * 75} for _ in range(self.max_docs)]
        self.rerank("预热", dummy)
        if self.last_inference_error:
            from ..common.startup import StartupIssue
            raise StartupIssue("reranker_inference_failed", "reranker", "精排模型预热推理失败",
                               "检查运行依赖和服务器可用内存，详细错误见应用日志。")

    # ---------- 排序 ----------

    def rerank(self, query: str, documents: List[Dict[str, Any]],
               top_k: int = 5) -> List[Dict[str, Any]]:
        if not documents:
            return []

        self._ensure_initialized()

        if self.model is not None:
            scored = self._score_with_model(query, documents)
        else:
            scored = self._score_with_heuristics(query, documents)

        scored.sort(key=lambda d: d.get("rerank_score", 0.0), reverse=True)
        return scored[:top_k]

    def _score_with_model(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.last_inference_error = None
        texts = [self._doc_text(doc) for doc in documents]
        scores: List[float] = []

        try:
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start:start + self.batch_size]
                inputs = self.tokenizer(
                    [query] * len(batch_texts),
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with self.torch.inference_mode():
                    logits = self.model(**inputs).logits
                if logits.shape[-1] == 1:
                    batch_scores = self.torch.sigmoid(logits).squeeze(-1)
                else:
                    batch_scores = self.torch.softmax(logits, dim=-1)[:, 1]
                scores.extend(batch_scores.float().cpu().tolist())
        except Exception as e:
            self.last_inference_error = type(e).__name__
            print(f"[Reranker] 模型推理失败，退回启发式排序: {e}")
            return self._score_with_heuristics(query, documents)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
        return list(documents)

    @staticmethod
    def _doc_text(doc: Dict[str, Any]) -> str:
        """精排用子块文本；子块过短时补一点父块上下文。"""
        child = (doc.get("child_text") or "").strip()
        if len(child) >= 30:
            return child
        parent = (doc.get("parent_text") or "").strip()
        return f"{parent[:400]}{child}" if parent else child

    @staticmethod
    def _score_with_heuristics(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for doc in documents:
            overlap = keyword_overlap(query, doc.get("child_text", ""), drop_question_words=False)
            rrf = doc.get("rrf_score", 0.0)
            # RRF 分量级约 0~0.05，归一化到 0~1 后再与词面重叠加权
            normalized_rrf = min(rrf / 0.05, 1.0)
            doc["rerank_score"] = 0.7 * overlap + 0.3 * normalized_rrf
        return list(documents)
