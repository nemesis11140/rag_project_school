"""RAG 检索问答主链路。

    意图识别 -> (闲聊) 兜底回答
             -> (咨询) 检索策略选择 + query 改写
                     -> 向量化 + 混合检索双路召回
                     -> bge-reranker 精排
                     -> 父块 + 原 query 构建上下文 -> LLM 生成
"""
import time
from typing import Any, Dict, List, Optional

from config import SYSTEM_CONFIG, EMBEDDING_CONFIG, RERANKER_CONFIG
from ..common.embedding import embedding_model_singleton
from ..common.intent_features import INTENT_CONSULT
from ..common.startup import StartupIssue, check_model_files
from ..knowledge_base import KnowledgeBase
from .answer_generator import AnswerGenerator
from .hybrid_retriever import HybridRetriever
from .intent_classifier import IntentClassifier
from .query_rewriter import QueryRewriter
from .reranker import Reranker


class RAGQASystem:
    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        print("[RAGQA] Initializing RAG QA System...")

        # 与 KnowledgeBase 共用同一个向量库句柄和 embedding 单例，
        # 避免重复加载 bge-m3（CPU 上约 2GB 内存）
        self.knowledge_base = knowledge_base or KnowledgeBase()

        self.intent_classifier = IntentClassifier()
        self.query_rewriter = QueryRewriter()
        self.hybrid_retriever = HybridRetriever(vector_store=self.knowledge_base.vector_store)
        self.reranker = Reranker()
        self.answer_generator = AnswerGenerator()

        print("[RAGQA] RAG QA System initialized")

    def warmup(self) -> Dict[str, float]:
        """把懒加载的模型全部提前加载并各跑一次前向。幂等，可重复调用。

        链路上共有三处懒加载，漏掉任何一处首个请求就要等它：
          1. 检索：bge-m3（2.2G）+ Milvus + BM25 索引
          2. 意图：bert_intent（391M）
          3. 精排：bge-reranker-base（1.1G）
        """
        print("[RAGQA] Warming up...")
        timings: Dict[str, float] = {}

        # 先检查数据状态和模型挂载，避免把数据库缺失显示为模型加载失败。
        self.knowledge_base.vector_store.check_ready()
        check_model_files(EMBEDDING_CONFIG["model_path"], "向量")
        check_model_files(RERANKER_CONFIG["model_path"], "精排")
        t0 = time.time()
        try:
            self.hybrid_retriever._ensure_initialized()
        except StartupIssue:
            raise
        except Exception as error:
            raise StartupIssue("retrieval_initialization_failed", "retrieval", "检索组件初始化失败",
                               "查看应用日志中的具体异常，检查模型文件、内存及知识库结构。") from error
        timings["retrieval_ms"] = round((time.time() - t0) * 1000, 1)

        t0 = time.time()
        # classify() 内部会触发 _load()，不用碰私有方法
        self.intent_classifier.classify("奖学金怎么申请")
        timings["intent_ms"] = round((time.time() - t0) * 1000, 1)

        t0 = time.time()
        self.reranker.warmup()
        if not self.reranker.is_loaded:
            raise StartupIssue("reranker_load_failed", "reranker", "精排模型加载失败",
                               "检查模型文件、运行依赖和可用内存后重试预热。")
        timings["rerank_ms"] = round((time.time() - t0) * 1000, 1)

        timings["total_ms"] = round(sum(
            v for k, v in timings.items() if k != "total_ms"
        ), 1)
        print(f"[RAGQA] Warmup done in {timings['total_ms']}ms")
        return timings

    def get_model_status(self) -> Dict[str, Any]:
        """只读的模型就绪状态，不会触发任何加载。"""
        return {
            "embedding": embedding_model_singleton.is_loaded,
            "intent": self.intent_classifier.backend,
            "reranker": self.reranker.is_loaded,
        }

    def process_query(self, query: str, top_k: Optional[int] = None,
                      document: Optional[str] = None,
                      chat_history: Optional[List[Dict]] = None,
                      chat_count: int = 0,
                      intent: Optional[str] = None) -> Dict[str, Any]:
        if top_k is None:
            top_k = SYSTEM_CONFIG["default_top_k"]

        started = time.time()
        timings: Dict[str, float] = {}

        print(f"\n[RAGQA] query={query}")

        # 追问经常省略主题，例如上一轮问“转专业怎么申请”，这一轮只问
        # “那需要什么材料”。分类、检索和精排要使用补全后的问题；最终生成
        # 仍保留用户原话，并把历史消息一并交给 LLM。
        contextual_query = self.query_rewriter.contextualize(query, chat_history)
        if contextual_query != query:
            print(f"[RAGQA] contextual_query={contextual_query}")

        # 1. 意图识别
        t0 = time.time()
        if intent is None:
            intent = self.intent_classifier.classify(contextual_query)
        timings["intent_ms"] = round((time.time() - t0) * 1000, 1)
        print(f"[RAGQA] intent={intent} ({timings['intent_ms']}ms)")

        # 2. 闲聊 -> 兜底回答
        if intent != INTENT_CONSULT:
            answer = self.answer_generator.generate_general(
                query, chat_history=chat_history, chat_count=chat_count
            )
            timings["total_ms"] = round((time.time() - started) * 1000, 1)
            return {
                "query": query,
                "intent": intent,
                "answer": answer,
                "source": "chitchat_fallback",
                "timings": timings
            }

        # 3. 检索策略选择 + query 改写
        t0 = time.time()
        rewrite_result = self.query_rewriter.rewrite(contextual_query)
        rewritten_queries = rewrite_result["queries"]
        timings["rewrite_ms"] = round((time.time() - t0) * 1000, 1)
        print(f"[RAGQA] strategy={rewrite_result['strategy']} queries={rewritten_queries}")

        # 4. 混合检索 + 双路召回
        t0 = time.time()
        retrieved = self.hybrid_retriever.retrieve(
            rewritten_queries, top_k=top_k * 2, filter_document=document
        )
        timings["retrieval_ms"] = round((time.time() - t0) * 1000, 1)
        print(f"[RAGQA] retrieved={len(retrieved)} ({timings['retrieval_ms']}ms)")

        # 5. 精排。CPU 上单条前向约 0.2s（base）到 0.6s（large），是整条链路的耗时大头，
        #    所以先截断候选数再精排（有 GPU 时可放宽 max_docs）
        t0 = time.time()
        reranked = self.reranker.rerank(
            contextual_query, retrieved[:self.reranker.max_docs], top_k=top_k
        )
        timings["rerank_ms"] = round((time.time() - t0) * 1000, 1)
        print(f"[RAGQA] reranked={len(reranked)} ({timings['rerank_ms']}ms)")

        # 6. 生成
        t0 = time.time()
        answer = self.answer_generator.generate_rag(
            query, reranked, chat_history=chat_history, chat_count=chat_count
        )
        timings["generate_ms"] = round((time.time() - t0) * 1000, 1)
        timings["total_ms"] = round((time.time() - started) * 1000, 1)
        print(f"[RAGQA] done in {timings['total_ms']}ms")

        return {
            "query": query,
            "contextual_query": contextual_query,
            "intent": intent,
            "answer": answer,
            "source": "rag",
            "strategy": rewrite_result["strategy"],
            "rewritten_queries": rewritten_queries,
            "retrieved_docs": reranked,
            "timings": timings
        }

    def add_documents(self, file_name: str) -> Dict[str, Any]:
        return self.knowledge_base.process_document(file_name)
