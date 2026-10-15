"""部署回归：空表恢复、现有数据保护、知识库预检、故障分类和修复后重试。"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import main
from scripts.restore_components import mysql_plan, schema_signature
from src.common.startup import StartupIssue, check_model_files
from src.knowledge_base.vector_store import VectorStoreManager
from src.rag_qa.hybrid_retriever import HybridRetriever


class MySQLRestorePlanTests(unittest.TestCase):
    def setUp(self):
        self.expected = {name: {"schema": f"CREATE TABLE `{name}` (`id` int) ENGINE=InnoDB AUTO_INCREMENT=8",
                                "rows": [[1]]} for name in ("users", "user_sessions", "chat_conversations", "chat_messages", "high_frequency_qa")}

    def test_four_compatible_empty_tables_are_reused(self):
        existing = copy.deepcopy(self.expected)
        existing.pop("high_frequency_qa")
        for item in existing.values():
            item["rows"] = []
            item["schema"] = item["schema"].replace("AUTO_INCREMENT=8", "AUTO_INCREMENT=1")
        plan = mysql_plan(existing, self.expected)
        self.assertEqual(plan["create"], ["high_frequency_qa"])
        self.assertEqual(set(plan["insert"]), set(self.expected))

    def test_schema_mismatch_stops_before_writing(self):
        existing = {"users": {"schema": "CREATE TABLE users (other_column int)", "rows": []}}
        with self.assertRaisesRegex(ValueError, "不兼容"):
            mysql_plan(existing, self.expected)

    def test_redundant_column_charset_is_compatible_but_different_collation_is_not(self):
        base = "CREATE TABLE t (name varchar(20)) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"
        explicit = base.replace("varchar(20)", "varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        self.assertEqual(schema_signature(base), schema_signature(explicit))
        different = base.replace("varchar(20)", "varchar(20) COLLATE utf8mb4_bin")
        self.assertNotEqual(schema_signature(base), schema_signature(different))

    def test_known_legacy_auth_schema_can_be_rebuilt_only_when_whole_database_is_empty(self):
        expected = {"chat_conversations": {"schema": "CREATE TABLE chat_conversations (id int, owner_id int)",
                                            "columns": ["id", "owner_id"], "rows": [[1, None]]}}
        actual = {"chat_conversations": {"schema": "CREATE TABLE chat_conversations (id int)",
                                          "columns": ["id"], "rows": []}}
        self.assertEqual(mysql_plan(actual, expected)["recreate"], ["chat_conversations"])
        actual["users"] = {"schema": "CREATE TABLE users (id int)", "columns": ["id"], "rows": [[99]]}
        expected["users"] = copy.deepcopy(actual["users"])
        with self.assertRaises(ValueError):
            mysql_plan(actual, expected, resume=True)

    def test_existing_business_data_is_not_overwritten(self):
        actual = copy.deepcopy(self.expected)
        actual["users"]["rows"] = [[999]]
        with self.assertRaises(ValueError):
            mysql_plan(actual, self.expected, resume=True)
        self.assertTrue(mysql_plan(actual, self.expected, keep_existing=True)["skip"])

    def test_verified_resume_only_fills_missing_or_empty_tables(self):
        actual = copy.deepcopy(self.expected)
        actual["chat_messages"]["rows"] = []
        actual.pop("high_frequency_qa")
        plan = mysql_plan(actual, self.expected, resume=True)
        self.assertEqual(set(plan["insert"]), {"chat_messages", "high_frequency_qa"})


class StartupChecksTests(unittest.TestCase):
    def test_missing_database_has_specific_reason_and_is_not_created(self):
        store = VectorStoreManager()
        with patch("src.knowledge_base.vector_store.connections") as connections, patch(
            "src.knowledge_base.vector_store.db"
        ) as database:
            database.list_database.return_value = ["default"]
            with self.assertRaises(StartupIssue) as raised:
                store.check_ready()
            self.assertEqual(raised.exception.code, "knowledge_database_missing")
            database.create_database.assert_not_called()
            connections.disconnect.assert_called_once()

    def test_missing_collection_and_empty_collection_are_not_ready(self):
        for has_collection, expected_code in ((False, "knowledge_collection_missing"), (True, "knowledge_collection_empty")):
            with self.subTest(code=expected_code), patch("src.knowledge_base.vector_store.connections"), patch(
                "src.knowledge_base.vector_store.db"
            ) as database, patch("src.knowledge_base.vector_store.utility") as utility, patch(
                "src.knowledge_base.vector_store.Collection"
            ) as collection:
                store = VectorStoreManager()
                database.list_database.return_value = [store.db_name]
                utility.has_collection.return_value = has_collection
                collection.return_value.num_entities = 0
                with self.assertRaises(StartupIssue) as raised:
                    store.check_ready()
                self.assertEqual(raised.exception.code, expected_code)

    def test_failed_preflight_can_retry_without_restarting(self):
        store = Mock()
        store.check_ready.side_effect = [StartupIssue("knowledge_database_missing", "milvus", "缺少库", "恢复备份"), None]
        retriever = HybridRetriever(vector_store=store)
        retriever._check_schema = Mock()
        retriever._load_corpus = Mock(side_effect=lambda: setattr(retriever, "corpus", [{"child_text": "测试文档"}]))
        retriever._build_bm25_index = Mock()
        with self.assertRaises(StartupIssue):
            retriever._ensure_initialized()
        self.assertFalse(retriever._initialized)
        store.load_embedding_model.assert_not_called()
        retriever._ensure_initialized()
        self.assertTrue(retriever._initialized)

    def test_model_directory_must_contain_weights_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(StartupIssue) as raised:
                check_model_files(directory, "向量")
            self.assertEqual(raised.exception.code, "model_files_missing")
            Path(directory, "config.json").write_text("{}")
            Path(directory, "model.safetensors").write_bytes(b"test-file")
            check_model_files(directory, "向量")


class HealthReportingTests(unittest.TestCase):
    def setUp(self):
        self.original = copy.deepcopy(main._warmup)
        main._warmup.update(status="pending", problem=None, error=None)

    def tearDown(self):
        main._warmup.clear()
        main._warmup.update(self.original)

    def test_database_error_is_not_reported_as_model_error_and_retry_recovers(self):
        system = Mock()
        system.warmup.side_effect = [StartupIssue("knowledge_database_missing", "milvus", "知识库尚未恢复", "恢复备份"), {"timings": {}}]
        system.get_model_status.return_value = {"embedding": True, "intent": "bert", "reranker": True}
        with patch.object(main, "get_auth_manager"), patch.object(main, "get_conversation_store"), patch.object(
            main, "get_system", return_value=system
        ), patch.object(main, "unified_system", system):
            main.run_warmup()
            state = main.health_check()
            self.assertFalse(state["models_ready"])
            self.assertEqual(state["problem"]["code"], "knowledge_database_missing")
            main.run_warmup()
            self.assertTrue(main.health_check()["models_ready"])
            self.assertIsNone(main.health_check()["problem"])

    def test_unexpected_details_are_logged_but_not_exposed_to_browser(self):
        with patch.object(main, "get_auth_manager", side_effect=RuntimeError("private connection details")):
            main.run_warmup()
        self.assertEqual(main._warmup["problem"]["code"], "mysql_unavailable")
        self.assertNotIn("private connection details", str(main.health_check()))


if __name__ == "__main__":
    unittest.main()
