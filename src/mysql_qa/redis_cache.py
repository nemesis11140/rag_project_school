"""Redis 热缓存。

职责：
  1. qa_index   —— 归一化问题 -> QA id 的倒排索引（"Redis 是否已有完全匹配的问题"）
  2. qa_match:v1 —— 原始问题 + 问答库/校验策略指纹 -> 已验证的问题映射（TTL）
  3. 启动时缓存全量问题 + 分词结果，避免每次重建 BM25 都回查 MySQL

归一化索引和旧 qa_answer 接口保留兼容；新问答链路只使用已验证的问题映射，
避免“挂失/补办”等归一化碰撞和旧错误答案绕过语义校验。
"""
import json
import hashlib
from typing import Any, Dict, List, Optional

import redis

from config import REDIS_CONFIG

# 缓存结构版本。改动问题/分词缓存的存储格式时必须 +1，
# 否则旧格式会被当成有效缓存读入，导致 id 缺失、索引为空等静默降级。
CACHE_VERSION = 2


class RedisCache:
    def __init__(self):
        self.client = redis.Redis(
            host=REDIS_CONFIG["host"],
            port=REDIS_CONFIG["port"],
            password=REDIS_CONFIG["password"] or None,
            db=REDIS_CONFIG["db"],
            decode_responses=True,
            socket_connect_timeout=5
        )
        self.qa_original_key = REDIS_CONFIG["qa_original_key"]
        self.qa_tokenized_key = REDIS_CONFIG["qa_tokenized_key"]
        self.index_key = REDIS_CONFIG.get("qa_index_key", "qa_index")
        self.answer_prefix = "qa_answer:"
        self.verified_match_prefix = "qa_match:v1:"

    # ---------- 问题索引 ----------

    def set_qa_index(self, entries: List[Dict[str, Any]]) -> bool:
        """写入 归一化问题 -> {id, question} 索引。"""
        try:
            pipe = self.client.pipeline()
            pipe.delete(self.index_key)
            if entries:
                mapping = {
                    e["normalized"]: json.dumps(
                        {"id": e["id"], "question": e["question"]}, ensure_ascii=False
                    )
                    for e in entries
                }
                pipe.hset(self.index_key, mapping=mapping)
            pipe.execute()
            return True
        except Exception as e:
            print(f"[RedisCache] Error setting qa index: {e}")
            return False

    def get_qa_index_entry(self, normalized_question: str) -> Optional[Dict[str, Any]]:
        try:
            raw = self.client.hget(self.index_key, normalized_question)
            return json.loads(raw) if raw else None
        except Exception as e:
            print(f"[RedisCache] Error reading qa index: {e}")
            return None

    # ---------- 问题 / 分词缓存 ----------

    def get_original_questions(self) -> List[str]:
        try:
            data = self.client.get(self.qa_original_key)
            if not data:
                return []
            payload = json.loads(data)
            # 兼容旧格式（纯 list）与新格式（{"ids": [...], "questions": [...]}）
            if isinstance(payload, dict):
                return payload.get("questions", [])
            return payload
        except Exception as e:
            print(f"[RedisCache] Error getting original questions: {e}")
            return []

    def get_tokenized_questions(self) -> List[List[str]]:
        try:
            data = self.client.get(self.qa_tokenized_key)
            return json.loads(data) if data else []
        except Exception as e:
            print(f"[RedisCache] Error getting tokenized questions: {e}")
            return []

    def set_questions_cache(self, questions: List[str],
                            tokenized_questions: List[List[str]],
                            ids: List[Any]) -> bool:
        try:
            pipe = self.client.pipeline()
            pipe.set(self.qa_original_key, json.dumps(
                {"version": CACHE_VERSION, "ids": ids, "questions": questions},
                ensure_ascii=False))
            pipe.set(self.qa_tokenized_key,
                     json.dumps(tokenized_questions, ensure_ascii=False))
            pipe.execute()
            return True
        except Exception as e:
            print(f"[RedisCache] Error setting questions cache: {e}")
            return False

    def get_question_ids(self) -> List[Any]:
        try:
            data = self.client.get(self.qa_original_key)
            if not data:
                return []
            payload = json.loads(data)
            if isinstance(payload, dict):
                return payload.get("ids", [])
            return list(range(len(payload)))
        except Exception as e:
            print(f"[RedisCache] Error getting question ids: {e}")
            return []

    def is_cache_current(self) -> bool:
        """缓存格式是否为当前版本。旧格式必须重建，否则会静默降级。"""
        try:
            data = self.client.get(self.qa_original_key)
            if not data:
                return False
            payload = json.loads(data)
            return isinstance(payload, dict) and payload.get("version") == CACHE_VERSION
        except Exception:
            return False

    # ---------- 答案缓存 ----------

    def get_cached_answer(self, normalized_question: str) -> Optional[str]:
        try:
            return self.client.get(f"{self.answer_prefix}{normalized_question}")
        except Exception as e:
            print(f"[RedisCache] Error getting cached answer: {e}")
            return None

    def set_cached_answer(self, normalized_question: str, answer: str, ttl: int = 86400) -> bool:
        try:
            self.client.set(f"{self.answer_prefix}{normalized_question}", answer, ex=ttl)
            return True
        except Exception as e:
            print(f"[RedisCache] Error setting cached answer: {e}")
            return False

    # ---------- 维护 ----------

    def get_verified_match(self, query: str, scope: str) -> Optional[Dict[str, Any]]:
        """新缓存仅记录已确认等价的原始问题映射；旧答案字符串不作为证据。"""
        try:
            digest = hashlib.sha256(json.dumps([scope, query], ensure_ascii=False).encode()).hexdigest()
            raw = self.client.get(f"{self.verified_match_prefix}{digest}")
            item = json.loads(raw) if raw else None
            if isinstance(item, dict) and item.get("query") == query and item.get("version") == 1:
                return item
        except Exception as error:
            print(f"[RedisCache] Error reading verified match: {type(error).__name__}")
        return None

    def set_verified_match(self, query: str, scope: str, question: str, qa_id: Any,
                           ttl: int = 86400) -> bool:
        try:
            digest = hashlib.sha256(json.dumps([scope, query], ensure_ascii=False).encode()).hexdigest()
            payload = {"version": 1, "query": query, "question": question, "id": qa_id}
            self.client.set(f"{self.verified_match_prefix}{digest}",
                            json.dumps(payload, ensure_ascii=False), ex=ttl)
            return True
        except Exception as error:
            print(f"[RedisCache] Error writing verified match: {type(error).__name__}")
            return False

    def _scan_keys(self, pattern: str) -> List[str]:
        """用 SCAN 迭代替代 KEYS，避免阻塞 Redis。"""
        keys = []
        cursor = 0
        while True:
            cursor, batch = self.client.scan(cursor=cursor, match=pattern, count=500)
            keys.extend(batch)
            if cursor == 0:
                break
        return keys

    def clear_cache(self, clear_index: bool = False) -> bool:
        try:
            keys = self._scan_keys(f"{self.answer_prefix}*")
            keys += self._scan_keys(f"{self.verified_match_prefix}*")
            if keys:
                self.client.unlink(*keys)
            if clear_index:
                self.client.delete(self.index_key)
                self.client.delete(self.qa_original_key)
                self.client.delete(self.qa_tokenized_key)
            return True
        except Exception as e:
            print(f"[RedisCache] Error clearing cache: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        try:
            original = self.get_original_questions()
            return {
                "indexed_questions_count": self.client.hlen(self.index_key),
                "cached_questions_count": len(original),
                "cached_answers_count": len(self._scan_keys(f"{self.answer_prefix}*")),
                "verified_matches_count": len(self._scan_keys(f"{self.verified_match_prefix}*")),
            }
        except Exception as e:
            print(f"[RedisCache] Error getting cache stats: {e}")
            return {}
