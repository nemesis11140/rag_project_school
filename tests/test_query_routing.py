"""无需数据库和模型文件的问答路由回归：python -m unittest discover -s tests -p test_query_routing.py。"""
import csv
import unittest
from pathlib import Path
from unittest.mock import Mock

from config import SYSTEM_CONFIG
from src.common.assistant_intent import ASSISTANT_INTRO, get_assistant_reply
from src.common.intent_features import INTENT_CHITCHAT, INTENT_CONSULT
from src.mysql_qa.bm25_retriever import BM25Retriever
from src.mysql_qa.qa_processor import QAProcessor
from src.rag_qa.answer_generator import AnswerGenerator
from src.rag_qa.intent_classifier import IntentClassifier
from src.rag_qa.query_rewriter import QueryRewriter
from src.rag_qa.rag_qa_system import RAGQASystem
from src.unified_query_system import UnifiedQuerySystem


class QueryRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "data" / "高频问答对.csv"
        if not path.is_file():
            raise unittest.SkipTest("请先提供本地 data/高频问答对.csv，再运行 FAQ 数据回归")
        if not path.is_file():
            raise unittest.SkipTest("请先提供本地 data/高频问答对.csv，再运行 FAQ 数据回归")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            cls.pairs = list(csv.DictReader(stream))
        cls.answers = {i: row["答案"] for i, row in enumerate(cls.pairs)}
        cls.retriever = BM25Retriever()
        cls.retriever.load_questions([row["问题"] for row in cls.pairs])

    def setUp(self):
        # 使用实际路由、规则分类、分词和 BM25，只替换数据库、检索模型和 LLM。
        qa = QAProcessor.__new__(QAProcessor)
        qa._loaded = True
        qa.bm25_retriever = self.retriever
        qa.bm25_threshold = SYSTEM_CONFIG["bm25_threshold"]
        qa.keyword_overlap_threshold = SYSTEM_CONFIG["keyword_overlap_threshold"]
        qa.candidate_count = SYSTEM_CONFIG["faq_candidate_count"]
        qa.semantic_matcher = Mock(policy_key="test-policy", min_confidence=0.9)
        qa.semantic_matcher.select_match.side_effect = lambda query, candidates: (
            {**candidates[0], "semantic_confidence": 0.99} if candidates else None
        )
        qa.redis_cache = Mock()
        qa.redis_cache.get_verified_match.return_value = None
        qa.redis_cache.get_cached_answer.return_value = None
        qa.redis_cache.get_qa_index_entry.return_value = None
        qa.mysql_manager = Mock()
        qa.mysql_manager.get_answer_by_id.side_effect = self.answers.get
        qa.query = Mock(wraps=qa.query)

        rag = RAGQASystem.__new__(RAGQASystem)
        classifier = IntentClassifier()
        classifier.preferred = "rules"
        classifier.classify = Mock(wraps=classifier.classify)
        rag.intent_classifier = classifier
        rag.query_rewriter = QueryRewriter()
        rag.hybrid_retriever = Mock()
        rag.hybrid_retriever.retrieve.return_value = [{"parent_text": "办理规定"}]
        rag.reranker = Mock(max_docs=5)
        rag.reranker.rerank.return_value = [{"parent_text": "办理规定"}]
        rag.answer_generator = AnswerGenerator()
        rag.answer_generator.api_url = "https://example.invalid"
        rag.answer_generator.api_key = "test-only"
        rag.answer_generator._call_llm = Mock(return_value="模型回答")

        self.system = UnifiedQuerySystem.__new__(UnifiedQuerySystem)
        self.system.mysql_qa_processor = qa
        self.system.rag_system = rag
        self.qa, self.rag = qa, rag

    def test_reproduce_generic_word_false_match(self):
        top = self.retriever.search_topk("你有什么用", top_k=1)[0]
        self.assertEqual(top["question"], "校园卡有什么用？")
        self.assertGreaterEqual(top["score"], self.qa.bm25_threshold)
        self.assertEqual(top["keyword_overlap"], 1)

    def test_assistant_phrasings(self):
        for query in (
            "你有什么用", "你能做什么？", "请问，你能帮我做什么？", "你是谁？",
            "介绍一下你自己", "你的功能是什么", "你有啥功能", "你是干嘛的",
            "校园问答助手有什么用？", "你能回答哪些校园问题？", "你好！",
        ):
            with self.subTest(query=query):
                # “干嘛”与“干什么”都是常见的口语问法。
                self.assertEqual(get_assistant_reply(query), ASSISTANT_INTRO)

    def test_domain_questions_are_not_assistant_introductions(self):
        for query in (
            "校园卡有什么用？", "你知道校园卡有什么用吗？", "你能帮我申请奖学金吗？",
            "请问奖学金怎么申请", "你好，学生证丢了怎么办？", "那有什么用？",
            "你有什么用，校园卡怎么补办？",
        ):
            with self.subTest(query=query):
                self.assertIsNone(get_assistant_reply(query))

    def test_assistant_query_bypasses_wrong_cached_answer(self):
        self.qa.redis_cache.get_cached_answer.return_value = "校园卡可用于食堂消费。"
        result = self.system.query("你有什么用")
        self.assertEqual(result["answer"], ASSISTANT_INTRO)
        self.assertEqual(result["intent"], INTENT_CHITCHAT)
        self.assertEqual(result["source"], "chitchat_fallback")
        self.qa.query.assert_not_called()
        self.qa.redis_cache.get_cached_answer.assert_not_called()
        self.rag.hybrid_retriever.retrieve.assert_not_called()
        self.rag.answer_generator._call_llm.assert_not_called()
        self.rag.intent_classifier.classify.assert_called_once_with("你有什么用")

    def test_assistant_query_does_not_inherit_previous_topic(self):
        history = [
            {"role": "user", "content": "校园卡有什么用？"},
            {"role": "assistant", "content": "校园卡可用于食堂消费。"},
        ]
        result = self.system.query("你能做什么呢？", chat_history=history, chat_count=1)
        self.assertEqual(result["answer"], ASSISTANT_INTRO)
        self.rag.intent_classifier.classify.assert_called_once_with("你能做什么呢？")
        self.qa.query.assert_not_called()
        self.rag.hybrid_retriever.retrieve.assert_not_called()

    def test_direct_rag_entry_handles_assistant_query(self):
        result = self.rag.process_query("你有什么用")
        self.assertEqual(result["answer"], ASSISTANT_INTRO)
        self.rag.hybrid_retriever.retrieve.assert_not_called()
        self.rag.answer_generator._call_llm.assert_not_called()

    def test_assistant_intro_works_without_llm(self):
        self.rag.answer_generator.api_key = ""
        self.assertEqual(self.system.query("你能做什么？")["answer"], ASSISTANT_INTRO)

    def test_assistant_intent_does_not_require_model(self):
        self.rag.intent_classifier._load = Mock(side_effect=AssertionError("不应加载模型"))
        self.assertEqual(self.system.query("校园问答助手有什么用？")["answer"], ASSISTANT_INTRO)

    def test_general_question_bypasses_faq(self):
        result = self.system.query("今天天气怎么样？")
        self.assertEqual(result["intent"], INTENT_CHITCHAT)
        self.qa.query.assert_not_called()
        self.rag.hybrid_retriever.retrieve.assert_not_called()
        self.rag.answer_generator._call_llm.assert_called_once()

    def test_campus_faq_still_matches(self):
        cases = {
            "校园卡有什么用？": "校园卡有什么用？",
            "如何申请奖学金": "怎么申请奖学金？",
            "学费多少钱": "学费要交多少啊？",
            "我弄丢了学生证怎么办": "学生证丢了怎么办？",
            "宿舍门禁是几点": "宿舍门禁时间是几点？",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                result = self.system.query(query)
                self.assertEqual(result["question"], expected)
                self.assertEqual(result["answer"], next(
                    row["答案"] for row in self.pairs if row["问题"] == expected
                ))
        self.rag.hybrid_retriever.retrieve.assert_not_called()

    def test_explicit_card_question_keeps_campus_intent(self):
        result = self.system.query("你知道校园卡有什么用吗？")
        self.assertEqual(result["intent"], INTENT_CONSULT)
        self.assertEqual(result["source"], "rag")
        self.assertNotEqual(result["answer"], ASSISTANT_INTRO)
        self.qa.query.assert_called_once()
        self.rag.hybrid_retriever.retrieve.assert_called_once()

    def test_campus_follow_up_uses_history_and_classifies_once(self):
        history = [{"role": "user", "content": "奖学金怎么申请？"}]
        result = self.system.query("那需要什么材料？", chat_history=history)
        self.assertEqual(result["intent"], INTENT_CONSULT)
        self.assertIn("奖学金", result["contextual_query"])
        self.qa.query.assert_not_called()
        self.rag.hybrid_retriever.retrieve.assert_called_once()
        self.rag.intent_classifier.classify.assert_called_once()


if __name__ == "__main__":
    unittest.main()
