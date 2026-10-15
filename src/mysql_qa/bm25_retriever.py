"""高频问答对的 BM25 检索器。

分词/停用词统一走 src/common/text_utils，与 RAG 检索侧保持一致。
"""
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from ..common.text_utils import char_ngram_similarity, keyword_overlap, normalize, tokenize

# BM25 打平时的字面相似度权重。取值远小于典型分差（约 1.2），
# 只影响"分数几乎相同"的候选，不会翻掉 BM25 的真实排序。
CHAR_TIEBREAK_WEIGHT = 0.5


class BM25Retriever:
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.questions: List[str] = []
        self.normalized_questions: List[str] = []
        self.ids: List[Any] = []
        self.tokenized_questions: List[List[str]] = []

    def load_questions(self, questions: List[str], tokenized_questions: List[List[str]] = None,
                       ids: List[Any] = None):
        self.questions = questions or []
        self.ids = ids or list(range(len(self.questions)))
        self.normalized_questions = [normalize(q) for q in self.questions]

        if tokenized_questions:
            self.tokenized_questions = tokenized_questions
        else:
            self.tokenized_questions = [self.tokenize(q) for q in self.questions]

        if self.tokenized_questions:
            self.bm25 = BM25Okapi(self.tokenized_questions)
            print(f"[BM25Retriever] BM25 model built with {len(self.tokenized_questions)} questions")
        else:
            self.bm25 = None
            print("[BM25Retriever] No questions loaded, BM25 model not available")

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """对"问题"分词：先归一化，再去掉疑问词，让「如何X」和「怎么X」落到同一表示。"""
        return tokenize(text, drop_question_words=True)

    def calculate_keyword_overlap(self, query: str, matched_question: str) -> float:
        return keyword_overlap(query, matched_question, drop_question_words=True)

    def search(self, query: str) -> Optional[Dict[str, Any]]:
        """返回 top1，兼容旧接口。"""
        results = self.search_topk(query, top_k=1)
        return results[0] if results else None

    def search_topk(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """返回按分数降序的候选，附带分数与关键词覆盖率。"""
        if not self.bm25 or not self.questions:
            return []

        normalized_query = normalize(query)
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        results = []
        for idx in range(len(self.questions)):
            score = float(scores[idx])
            if score <= 0:
                continue
            question = self.questions[idx]
            results.append({
                "score": score,
                "rank_score": score + CHAR_TIEBREAK_WEIGHT * char_ngram_similarity(
                    normalized_query, self.normalized_questions[idx]
                ),
                "question": question,
                "id": self.ids[idx] if idx < len(self.ids) else None,
                "query": query,
                "keyword_overlap": self.calculate_keyword_overlap(query, question),
            })

        results.sort(key=lambda r: r["rank_score"], reverse=True)
        return results[:top_k]

    def search_with_overlap(self, query: str, keyword_overlap_threshold: float = 0.4,
                            top_k: int = 3) -> Optional[Dict[str, Any]]:
        for candidate in self.search_topk(query, top_k=top_k):
            if candidate["keyword_overlap"] >= keyword_overlap_threshold:
                return candidate
        return None
