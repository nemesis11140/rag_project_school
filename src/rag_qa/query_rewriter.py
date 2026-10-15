"""检索策略选择 + query 改写。

四种策略：
    direct      原问题      —— 永远保留，保证召回不劣化
    hypothetical 假设问题    —— HyDE 思路，改写成"文档式"表述，提升语义召回
    sub_query   子问题      —— 多子句问题拆开，避免只召回其中一半
    backtrack   回溯问题    —— step-back 思路，退一步问更上位的规定，补背景知识

原实现无条件把四类策略的约 10 条 query 全跑一遍，每条都要一次向量检索 + 一次
BM25，既慢又引入噪声（模板句会稀释语义相似度）。这里改为按 query 特征选 1 种，
加上原问题共 2 条。
"""
import re
from typing import Dict, List, Optional

from config import SYSTEM_CONFIG
from ..common.assistant_intent import get_assistant_reply

STRATEGY_DIRECT = "direct"
STRATEGY_HYPOTHETICAL = "hypothetical"
STRATEGY_SUB_QUERY = "sub_query"
STRATEGY_BACKTRACK = "backtrack"

ALL_STRATEGIES = [STRATEGY_DIRECT, STRATEGY_HYPOTHETICAL, STRATEGY_SUB_QUERY, STRATEGY_BACKTRACK]

# 多子句信号
_MULTI_CLAUSE_RE = re.compile(r"[、；;]|并且|以及|同时|还有|分别")
# 条件 / 许可类问法
_CONDITION_RE = re.compile(r"能不能|可不可以|能否|是否可以|需要什么|什么条件|可以不|是否要")
# 疑问句式
_QUESTION_RE = re.compile(r"吗|呢|怎么|如何|什么|哪些|多少|几点|多久")
# 追问通常省略上一轮已经说过的主题。只有出现明确承接词，或问题只剩下
# "材料/条件/流程"这类泛指信息时，才拼接历史，避免把一个全新问题误接到旧话题。
_FOLLOW_UP_RE = re.compile(
    r"^(那|那么|这个|这种|这项|它|其|上述|前面|刚才|然后|还要|还有|另外|具体)"
)
_VAGUE_FOLLOW_UP_RE = re.compile(
    r"^(需要|要|有|会|能|可以|应该|怎么办|怎么处理).*(材料|条件|流程|手续|影响|时间|地点|费用|多久|多少|吗|呢)"
)


class QueryRewriter:
    def __init__(self, max_queries: Optional[int] = None):
        self.max_queries = max_queries or SYSTEM_CONFIG.get("rewrite_max_queries", 2)

    # ---------- 多轮上下文 ----------

    @staticmethod
    def is_follow_up(query: str, chat_history: Optional[List[Dict]] = None) -> bool:
        """判断当前问题是否省略了上一轮主题。"""
        if not chat_history:
            return False

        query = (query or "").strip()
        if not query:
            return False

        # “你能做什么呢”是在询问助手，不承接上一轮的校园卡等主题。
        if get_assistant_reply(query) is not None:
            return False

        has_previous_user_message = any(
            msg.get("role") == "user" and str(msg.get("content", "")).strip()
            for msg in chat_history
        )
        if not has_previous_user_message:
            return False

        return bool(
            _FOLLOW_UP_RE.search(query)
            or _VAGUE_FOLLOW_UP_RE.search(query)
            or (len(query) <= 10 and query.rstrip("？?").endswith("呢"))
        )

    @classmethod
    def contextualize(cls, query: str,
                      chat_history: Optional[List[Dict]] = None) -> str:
        """为省略主题的追问补上最近两条用户问题，供分类和检索使用。"""
        query = (query or "").strip()
        if not cls.is_follow_up(query, chat_history):
            return query

        previous_questions = [
            str(msg.get("content", "")).strip()
            for msg in (chat_history or [])
            if msg.get("role") == "user" and str(msg.get("content", "")).strip()
        ][-2:]
        context = "；".join(previous_questions)
        if len(context) > 200:
            context = context[-200:]
        return f"{context}。用户继续追问：{query}"

    # ---------- 策略选择 ----------

    def select_strategy(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return STRATEGY_DIRECT

        # 多子句 -> 拆子问题
        if _MULTI_CLAUSE_RE.search(query) or query.count("？") + query.count("?") > 1:
            return STRATEGY_SUB_QUERY

        # 条件/许可类 -> 退一步问规定
        if _CONDITION_RE.search(query):
            return STRATEGY_BACKTRACK

        # 很短或没有明确疑问词 -> 假设一个文档式表述来兜住语义
        if len(query) <= 8 or not _QUESTION_RE.search(query):
            return STRATEGY_HYPOTHETICAL

        return STRATEGY_DIRECT

    # ---------- 改写 ----------

    def rewrite(self, query: str, strategy: Optional[str] = None,
                max_queries: Optional[int] = None) -> Dict[str, object]:
        """返回 {"strategy": ..., "queries": [...]}，原问题永远排第一位。"""
        query = (query or "").strip()
        if not query:
            return {"strategy": STRATEGY_DIRECT, "queries": []}

        strategy = strategy or self.select_strategy(query)
        limit = max_queries or self.max_queries

        queries = [query]
        if strategy == STRATEGY_HYPOTHETICAL:
            queries.extend(self._hypothetical(query))
        elif strategy == STRATEGY_SUB_QUERY:
            queries.extend(self._sub_queries(query))
        elif strategy == STRATEGY_BACKTRACK:
            queries.extend(self._backtrack(query))

        deduped = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))
        return {"strategy": strategy, "queries": deduped[:limit]}

    @staticmethod
    def _hypothetical(query: str) -> List[str]:
        """HyDE-lite：改写成手册/规定正文的口吻，靠近文档向量空间。"""
        return [
            f"根据上海交通大学学生手册和教职工手册的规定，{query}的相关条款与办理流程如下。",
        ]

    @staticmethod
    def _sub_queries(query: str) -> List[str]:
        """按标点/连接词把多子句问题拆成独立子问题。"""
        parts = re.split(r"[、；;]|并且|以及|同时|还有|分别", query)
        sub_queries = []
        for part in parts:
            part = part.strip(" ，,。？?！!")
            if len(part) >= 4:
                sub_queries.append(f"{part}是怎么规定的？")
        return sub_queries[:2]

    @staticmethod
    def _backtrack(query: str) -> List[str]:
        """step-back：退一步问更上位的规定，用于补背景知识。"""
        cleaned = re.sub(_CONDITION_RE, "", query).strip(" ，,。？?！!")
        if len(cleaned) < 3:
            return []
        return [f"关于{cleaned}，学校有哪些相关规定和办理条件？"]
