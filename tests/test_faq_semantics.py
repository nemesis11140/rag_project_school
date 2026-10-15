"""FAQ 语义闸门、严格解析与缓存隔离的回归；不访问外部服务或数据库。"""
import json
import unittest
from unittest.mock import Mock, patch

import requests

import test_query_routing as routing_tests
from src.mysql_qa.redis_cache import RedisCache
from src.mysql_qa.semantic_matcher import SemanticQAMatcher, question_key, has_negated_condition


class FAQSemanticGateTests(unittest.TestCase):
    setUpClass = classmethod(routing_tests.QueryRoutingTests.setUpClass.__func__)
    setUp = routing_tests.QueryRoutingTests.setUp

    def test_lexical_high_score_cannot_return_answer_without_semantic_approval(self):
        self.qa.semantic_matcher.select_match.side_effect = None
        self.qa.semantic_matcher.select_match.return_value = None
        for query in ("你有什么用", "校园卡怎么挂失", "不申请奖学金会怎么样"):
            with self.subTest(query=query):
                self.assertIsNone(self.qa.query(query))
        self.qa.mysql_manager.get_answer_by_id.assert_not_called()
        self.qa.redis_cache.set_verified_match.assert_not_called()

    def test_normalized_exact_index_cannot_bypass_semantics(self):
        self.qa.redis_cache.get_qa_index_entry.return_value = {
            "question": "校园卡怎么补办？", "id": 128,
        }
        self.qa.redis_cache.get_cached_answer.return_value = "旧错误答案"
        self.qa.semantic_matcher.select_match.side_effect = None
        self.qa.semantic_matcher.select_match.return_value = None
        self.assertIsNone(self.qa.query("校园卡怎么挂失？"))
        self.qa.redis_cache.get_qa_index_entry.assert_not_called()
        self.qa.redis_cache.get_cached_answer.assert_not_called()
        self.qa.semantic_matcher.select_match.assert_called_once()
        self.assertEqual(self.qa.semantic_matcher.select_match.call_args.args[0], "校园卡怎么挂失？")

    def test_all_200_original_questions_keep_exact_answers_without_llm(self):
        self.qa.semantic_matcher.select_match.side_effect = AssertionError("原文无需语义服务")
        for row in self.pairs:
            with self.subTest(query=row["问题"]):
                result = self.qa.query(row["问题"])
                self.assertEqual(result["answer"], row["答案"])
                self.assertEqual(result["method"], "exact_match")

    def test_semantic_selection_can_accept_second_candidate(self):
        candidates = [
            {"question": "校园卡有什么用？", "id": 132, "score": 8, "keyword_overlap": 1},
            {"question": "怎么申请奖学金？", "id": 90, "score": 7, "keyword_overlap": 1},
        ]
        self.qa.bm25_retriever = Mock(questions=self.retriever.questions, ids=self.retriever.ids)
        self.qa.bm25_retriever.search_topk.return_value = candidates
        self.qa.semantic_matcher.select_match.side_effect = lambda query, items: {
            **items[1], "semantic_confidence": 0.99,
        }
        result = self.qa.query("奖学金如何办理申请？")
        self.assertEqual(result["question"], "怎么申请奖学金？")
        self.assertEqual(result["method"], "semantic_verified")
        self.qa.redis_cache.set_verified_match.assert_called_once()

    def test_verified_cache_reuses_mapping_but_fetches_current_answer(self):
        index = next(i for i, row in enumerate(self.pairs) if row["问题"] == "怎么申请奖学金？")
        self.qa.redis_cache.get_verified_match.return_value = {
            "version": 1, "query": "如何申请奖学金", "question": "怎么申请奖学金？", "id": index,
        }
        self.qa.mysql_manager.get_answer_by_id.side_effect = None
        self.qa.mysql_manager.get_answer_by_id.return_value = "数据库最新答案"
        result = self.qa.query("如何申请奖学金")
        self.assertEqual(result["answer"], "数据库最新答案")
        self.assertEqual(result["method"], "semantic_cache_hit")
        self.qa.semantic_matcher.select_match.assert_not_called()

    def test_cache_for_other_query_or_missing_question_is_ignored(self):
        for cached in (
            {"version": 1, "query": "另一问题", "question": "怎么申请奖学金？", "id": 90},
            {"version": 1, "query": "如何申请奖学金", "question": "已删除的问题", "id": 90},
            {"version": 0, "query": "如何申请奖学金", "question": "怎么申请奖学金？", "id": 90},
        ):
            with self.subTest(cached=cached):
                self.qa.redis_cache.get_verified_match.return_value = cached
                self.qa.semantic_matcher.select_match.side_effect = None
                self.qa.semantic_matcher.select_match.return_value = None
                self.assertIsNone(self.qa.query("如何申请奖学金"))

    def test_cache_scope_changes_with_corpus_and_semantic_policy(self):
        self.qa.query("如何申请奖学金")
        first_scope = self.qa.redis_cache.get_verified_match.call_args.args[1]
        self.qa.semantic_matcher.policy_key = "new-policy"
        self.qa.query("如何申请奖学金")
        self.assertNotEqual(first_scope, self.qa.redis_cache.get_verified_match.call_args.args[1])


class SemanticResponseTests(unittest.TestCase):
    def setUp(self):
        self.matcher = SemanticQAMatcher()
        self.matcher.api_url, self.matcher.api_key = "https://example.invalid", "test-only"
        self.candidates = [{"question": "校园卡怎么补办？", "id": 1}]

    def response(self, content, finish="stop", status=200):
        response = Mock(status_code=status)
        response.json.return_value = {"choices": [{
            "finish_reason": finish, "message": {"content": content},
        }]}
        return response

    def test_valid_semantic_verdict_and_original_query_payload(self):
        with patch("src.mysql_qa.semantic_matcher.requests.post", return_value=self.response(
            '{"match_index": 0, "confidence": 0.99}'
        )) as post:
            result = self.matcher.select_match("校园卡怎么挂失？", self.candidates)
        self.assertEqual(result["semantic_confidence"], 0.99)
        payload = json.loads(post.call_args.kwargs["json"]["messages"][1]["content"])
        self.assertEqual(payload["user_query"], "校园卡怎么挂失？")
        self.assertEqual(payload["candidates"][0]["question"], "校园卡怎么补办？")

    def test_rejection_ambiguous_low_confidence_and_invalid_schema(self):
        for content in (
            '{"match_index": -1, "confidence": 1}', '{"match_index": 0, "confidence": 0.8}',
            '{"match_index": 999, "confidence": 1}', '{"match_index": true, "confidence": 1}',
            '{"match_index": 0, "confidence": true}', '{"match_index": 0, "confidence": NaN}',
            '{"match_index": 0, "confidence": 2}', '{"match_index": "0", "confidence": 1}',
            '{"match_index": 0}', '[]', 'null', '可以匹配', '```json\n{}\n```',
        ):
            with self.subTest(content=content), patch(
                "src.mysql_qa.semantic_matcher.requests.post", return_value=self.response(content)
            ):
                self.assertIsNone(self.matcher.select_match("测试问题", self.candidates))

    def test_timeout_http_error_and_truncated_response_fail_closed(self):
        with patch("src.mysql_qa.semantic_matcher.requests.post", side_effect=requests.Timeout):
            self.assertIsNone(self.matcher.select_match("测试", self.candidates))
        for response in (self.response("{}", status=500), self.response(
            '{"match_index": 0, "confidence": 1}', finish="length"
        )):
            with patch("src.mysql_qa.semantic_matcher.requests.post", return_value=response):
                self.assertIsNone(self.matcher.select_match("测试", self.candidates))

    def test_unconfigured_model_rejects_non_exact_match(self):
        self.matcher.api_key = ""
        with patch("src.mysql_qa.semantic_matcher.requests.post") as post:
            self.assertIsNone(self.matcher.select_match("测试", self.candidates))
        post.assert_not_called()

    def test_model_cannot_approve_changed_negation(self):
        for query, question in (
            ("请病假不提供医院证明可以吗", "请病假要医院证明吗"),
            ("必修课选上了怎么办", "必修课没选上怎么办"),
            ("不申请奖学金会怎么样", "怎么申请奖学金"),
        ):
            with self.subTest(query=query), patch(
                "src.mysql_qa.semantic_matcher.requests.post", return_value=self.response(
                    '{"match_index": 0, "confidence": 1}'
                )
            ):
                self.assertIsNone(self.matcher.select_match(query, [{"question": question}]))

    def test_neutral_yes_no_forms_are_not_negated_conditions(self):
        for query in ("宿舍有没有空调", "请病假需不需要医院证明", "能不能申请缓考", "住宿费退不退"):
            self.assertFalse(has_negated_condition(query), query)
        self.assertTrue(has_negated_condition("请病假不提供医院证明可以吗"))

    def test_format_cleanup_preserves_negation_and_different_actions(self):
        self.assertEqual(question_key("校园卡有什么用？"), question_key("校园卡有什么用?"))
        self.assertNotEqual(question_key("校园卡怎么挂失"), question_key("校园卡怎么补办"))
        self.assertNotEqual(question_key("不申请奖学金"), question_key("申请奖学金"))


class VerifiedRedisCacheTests(unittest.TestCase):
    def test_new_cache_is_bound_to_raw_query_and_policy(self):
        cache = RedisCache.__new__(RedisCache)
        cache.verified_match_prefix = "qa_match:v1:"
        values = {}
        cache.client = Mock()
        cache.client.set.side_effect = lambda key, value, **kwargs: values.update({key: value})
        cache.client.get.side_effect = values.get
        self.assertTrue(cache.set_verified_match("校园卡怎么补办", "scope", "校园卡怎么补办？", 1))
        self.assertIsNotNone(cache.get_verified_match("校园卡怎么补办", "scope"))
        self.assertIsNone(cache.get_verified_match("校园卡怎么挂失", "scope"))
        self.assertIsNone(cache.get_verified_match("校园卡怎么补办", "new-scope"))


if __name__ == "__main__":
    unittest.main()
