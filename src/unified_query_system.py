"""统一入口：先识别意图，校园咨询优先匹配高频问答，未命中则转 RAG。"""
import time
from typing import Any, Dict, List, Optional

from .common.intent_features import INTENT_CONSULT
from .knowledge_base import KnowledgeBase
from .mysql_qa import QAProcessor
from .rag_qa import RAGQASystem


class UnifiedQuerySystem:
    def __init__(self):
        print("[UnifiedQuery] Initializing...")

        # 只创建一份 KnowledgeBase，向量库句柄和 bge-m3 单例在
        # QAProcessor / RAGQASystem / KnowledgeBase 之间共享
        self.knowledge_base = KnowledgeBase()
        self.mysql_qa_processor = QAProcessor()
        self.rag_system = RAGQASystem(knowledge_base=self.knowledge_base)

        print("[UnifiedQuery] Initialized")

    def query(self, user_query: str, top_k: int = 5, document: Optional[str] = None,
              chat_history: Optional[List[Dict]] = None,
              chat_count: int = 0) -> Dict[str, Any]:
        print(f"\n[UnifiedQuery] query={user_query}")
        started = time.perf_counter()

        # 1. 先识别意图，避免闲聊被“用”等通用词误匹配到校园卡问答，
        # 也不能从 Redis 直接返回之前误匹配留下的答案。
        contextual_query = self.rag_system.query_rewriter.contextualize(
            user_query, chat_history
        )
        intent_started = time.perf_counter()
        intent = self.rag_system.intent_classifier.classify(contextual_query)
        intent_ms = round((time.perf_counter() - intent_started) * 1000, 1)

        # 2. 仅校园独立问题走高频问答。省略主题的追问结合上下文走 RAG。
        is_follow_up = self.rag_system.query_rewriter.is_follow_up(
            user_query, chat_history
        )
        if intent == INTENT_CONSULT and not is_follow_up:
            mysql_result = self.mysql_qa_processor.query(user_query)
            if mysql_result:
                return mysql_result

        # 3. 闲聊或未命中高频库 -> RAG 系统对应分支，复用已识别的意图。
        rag_result = self.rag_system.process_query(
            user_query,
            top_k=top_k,
            document=document,
            chat_history=chat_history,
            chat_count=chat_count,
            intent=intent
        )
        # 分类已移到统一入口，计时也包含这一阶段及高频问答的尝试。
        rag_result.setdefault("timings", {}).update({
            "intent_ms": intent_ms,
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
        })

        return {
            "query": user_query,
            "contextual_query": rag_result.get("contextual_query", user_query),
            "source": rag_result.get("source", "rag"),
            "answer": rag_result["answer"],
            "intent": rag_result.get("intent"),
            "strategy": rag_result.get("strategy"),
            "retrieved_docs": rag_result.get("retrieved_docs", []),
            "timings": rag_result.get("timings", {}),
            "method": "rag_pipeline" if rag_result.get("source") == "rag" else "chitchat_fallback"
        }

    def warmup(self) -> Dict[str, Any]:
        """提前加载全部懒加载的模型。幂等，重复调用只是再跑一次前向。

        高频问答那一路的数据（Redis + MySQL + BM25）在 __init__ 里就加载完了，
        所以这里只需要管 RAG 的三个模型。
        """
        timings = self.rag_system.warmup()
        return {"timings": timings, "models": self.get_model_status()}

    def get_model_status(self) -> Dict[str, Any]:
        """只读的模型就绪状态，不触发加载。"""
        return self.rag_system.get_model_status()

    def add_document(self, file_name: str) -> Dict[str, Any]:
        return self.knowledge_base.process_document(file_name)

    def build_knowledge_base(self) -> Dict[str, Any]:
        return self.knowledge_base.build_knowledge_base()

    def refresh_cache(self):
        self.mysql_qa_processor.refresh_cache()

    def get_stats(self):
        return {
            "mysql_qa": self.mysql_qa_processor.get_stats(),
            "knowledge_base": self.knowledge_base.get_stats()
        }
