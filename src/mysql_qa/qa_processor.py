"""高频问答对处理流程。

    原文精确匹配 / 已验证的原始问题映射 -> MySQL 取最新答案
    其他问题 -> BM25 召回候选 -> 语义等价校验 -> MySQL 取答案
    不确定、服务不可用或无等价问题 -> 返回 None，由上层转 RAG

BM25 分数和关键词覆盖率只用于筛选候选，不代表问题等价。
归一化仅用于召回；不能用“挂失 -> 补办”等替换后的文字跳过语义校验。
"""
import hashlib
import json
from typing import Any, Dict, Optional

from config import SYSTEM_CONFIG
from ..common.text_utils import normalize
from .bm25_retriever import BM25Retriever
from .mysql_manager import MySQLManager
from .redis_cache import RedisCache
from .semantic_matcher import SemanticQAMatcher, question_key


class QAProcessor:
    def __init__(self, load_cache: bool = True):
        self.mysql_manager = MySQLManager()
        self.redis_cache = RedisCache()
        self.bm25_retriever = BM25Retriever()
        self.bm25_threshold = SYSTEM_CONFIG["bm25_threshold"]
        self.keyword_overlap_threshold = SYSTEM_CONFIG["keyword_overlap_threshold"]
        self.semantic_matcher = SemanticQAMatcher()
        self.candidate_count = SYSTEM_CONFIG["faq_candidate_count"]
        self._loaded = False
        if load_cache:
            self._load_data()

    # ---------- 数据加载 ----------

    def _load_data(self):
        print("[QAProcessor] Loading high-frequency QA data...")

        # 缓存格式过期（或索引为空）时必须回 MySQL 重建，
        # 否则会拿旧格式的缓存跑，丢失 id 与归一化索引
        cache_usable = (
            self.redis_cache.is_cache_current()
            and self.redis_cache.get_cache_stats().get("indexed_questions_count", 0) > 0
        )

        questions = self.redis_cache.get_original_questions() if cache_usable else []
        tokenized = self.redis_cache.get_tokenized_questions() if cache_usable else []
        ids = self.redis_cache.get_question_ids() if cache_usable else []

        if not questions or not tokenized:
            print("[QAProcessor] Cache miss/outdated, loading from MySQL...")
            qa_pairs = self.mysql_manager.get_all_qa_pairs()
            if not qa_pairs:
                print("[QAProcessor] WARNING: MySQL 中没有高频问答数据")
                self._loaded = True
                return
            questions = [p["question"] for p in qa_pairs]
            ids = [p["id"] for p in qa_pairs]
            tokenized = [self.bm25_retriever.tokenize(q) for q in questions]
            self._write_cache(questions, tokenized, ids)
        else:
            print(f"[QAProcessor] Loaded {len(questions)} questions from Redis cache")

        if not ids or len(ids) != len(questions):
            ids = [None] * len(questions)

        self.bm25_retriever.load_questions(questions, tokenized, ids)
        self._loaded = True

    def _write_cache(self, questions, tokenized, ids):
        self.redis_cache.set_questions_cache(questions, tokenized, ids)
        entries = []
        for idx, question in enumerate(questions):
            normalized = normalize(question)
            if not normalized:
                continue
            entries.append({
                "normalized": normalized,
                "question": question,
                "id": ids[idx] if idx < len(ids) else None
            })
        self.redis_cache.set_qa_index(entries)
        print(f"[QAProcessor] Wrote {len(entries)} normalized questions to Redis index")

    def refresh_cache(self):
        print("[QAProcessor] Refreshing cache from MySQL...")
        self.redis_cache.clear_cache(clear_index=True)
        self._load_data()

    # ---------- 查询 ----------

    def query(self, user_query: str) -> Optional[Dict[str, Any]]:
        if not user_query or not user_query.strip():
            return None

        self._ensure_loaded()

        raw_query = user_query.strip()
        questions = self.bm25_retriever.questions
        ids = self.bm25_retriever.ids
        entries = [{"question": question, "id": ids[index]} for index, question in enumerate(questions)]

        # 只允许不改变语义的原文精确匹配；不用归一化索引，也不用旧答案缓存。
        exact = [item for item in entries if question_key(item["question"]) == question_key(raw_query)]
        if len(exact) == 1:
            return self._result(user_query, exact[0], "mysql", "exact_match")
        if len(exact) > 1:
            return None  # 重复问题可能有不同答案，不能自行选择。

        # 问答库、语义模型或校验规则变更后，旧映射自动失效。
        scope = hashlib.sha256(json.dumps(
            [self.semantic_matcher.policy_key, entries], sort_keys=True, ensure_ascii=False
        ).encode()).hexdigest()
        cached = self.redis_cache.get_verified_match(raw_query, scope)
        if isinstance(cached, dict):
            current = [item for item in entries
                       if item["question"] == cached.get("question") and item["id"] == cached.get("id")]
            if len(current) == 1 and cached.get("query") == raw_query and cached.get("version") == 1:
                result = self._result(user_query, current[0], "redis", "semantic_cache_hit")
                if result:
                    return result

        # BM25 只召回候选，不能直接据此返回答案。
        candidates = self.bm25_retriever.search_topk(user_query, top_k=self.candidate_count)
        if not candidates:
            print("[QAProcessor] BM25 无候选，转 RAG")
            return None

        candidates = [item for item in candidates
                      if item["score"] >= self.bm25_threshold
                      and item["keyword_overlap"] >= self.keyword_overlap_threshold]
        matched = self.semantic_matcher.select_match(raw_query, candidates)
        if matched is None:
            print("[QAProcessor] 无确认等价的高频问题，转 RAG")
            return None

        if sum(question_key(item["question"]) == question_key(matched["question"])
               for item in entries) != 1:
            return None
        result = self._result(user_query, matched, "mysql", "semantic_verified")
        if result:
            self.redis_cache.set_verified_match(raw_query, scope, matched["question"], matched.get("id"))
        return result

    def _result(self, user_query, matched, source, method) -> Optional[Dict[str, Any]]:
        answer = self._fetch_answer(matched.get("id"), matched["question"])
        if not answer:
            return None
        result = {
            "query": user_query,
            "source": source,
            "answer": answer,
            "question": matched["question"],
            "method": method,
        }
        for field in ("score", "keyword_overlap", "semantic_confidence"):
            if field in matched:
                result[field] = matched[field]
        return result

    def _ensure_loaded(self):
        if not self._loaded:
            self._load_data()

    def _fetch_answer(self, qa_id, question) -> Optional[str]:
        if qa_id is not None:
            answer = self.mysql_manager.get_answer_by_id(qa_id)
            if answer:
                return answer
        if question:
            return self.mysql_manager.get_answer(question)
        return None

    def get_stats(self):
        return {
            "bm25_threshold": self.bm25_threshold,
            "keyword_overlap_threshold": self.keyword_overlap_threshold,
            "faq_candidate_count": self.candidate_count,
            "faq_semantic_min_confidence": self.semantic_matcher.min_confidence,
            "indexed_questions": len(self.bm25_retriever.questions),
            **self.redis_cache.get_cache_stats()
        }
