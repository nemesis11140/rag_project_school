"""真实数据库恢复演练；仅使用独立临时库与空 Redis DB15，需显式 --live。"""
import argparse
import copy
import hashlib
import json
import sys
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymysql
import redis
from pymilvus import Collection, connections, db, utility
from config import MYSQL_CONFIG, MILVUS_CONFIG, REDIS_CONFIG
from scripts.migrate_to_docker import decode, encode
from scripts import restore_components as restore_tool
from src.auth import auth_manager
from src.mysql_qa import conversation_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live:
        parser.error("真实恢复演练需要 --live")
    name = "release_verify_" + uuid.uuid4().hex[:12]
    source_dir = Path("/app/backups/rag-project-school-db-backup-20261012-151748")
    source_raw = (source_dir / "snapshot.json").read_bytes()
    assert hashlib.sha256(source_raw).hexdigest() == (source_dir / "snapshot.sha256").read_text().strip()
    snapshot = json.loads(source_raw, object_hook=decode)
    snapshot["database"] = name
    snapshot["milvus_database"] = name
    admin = pymysql.connect(**MYSQL_CONFIG, autocommit=True)
    cache = redis.Redis(**{key: value for key, value in REDIS_CONFIG.items()
                          if key in {"host", "port", "password"}}, db=15)
    assert cache.dbsize() == 0, "Redis DB15 非空，未执行演练"
    connections.connect(alias="release_audit", host=MILVUS_CONFIG["host"], port=MILVUS_CONFIG["port"], db_name="default")
    assert name not in db.list_database(using="release_audit")
    with admin.cursor() as cursor:
        cursor.execute("SHOW DATABASES LIKE %s", (name,))
        assert not cursor.fetchone()
        cursor.execute(f"CREATE DATABASE `{name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    restore_tool.DB_NAME = name
    restore_tool.MILVUS_CONFIG = {**MILVUS_CONFIG, "database_name": name}
    restore_tool.REDIS_CONFIG = {**REDIS_CONFIG, "db": 15}
    auth_manager.DB_NAME = name
    conversation_store.DB_NAME = name
    created_keys = {item["key"] for item in snapshot["redis"]} | {b"release_verify_marker"}

    def drop_vectors():
        if name in db.list_database(using="release_audit"):
            db.using_database(name, using="release_audit")
            for collection in utility.list_collections(using="release_audit"):
                Collection(collection, using="release_audit").drop()
            db.using_database("default", using="release_audit")
            db.drop_database(name, using="release_audit")

    try:
        # 由真正应用代码创建四张空表，包含与历史快照不同的新版会话结构。
        auth_manager.AuthManager().ensure_tables()
        conversation_store.ConversationStore().ensure_tables()
        with tempfile.TemporaryDirectory(prefix="release_restore_") as directory:
            raw = json.dumps(snapshot, ensure_ascii=False, default=encode).encode()
            Path(directory, "snapshot.json").write_bytes(raw)
            Path(directory, "snapshot.sha256").write_text(hashlib.sha256(raw).hexdigest(), encoding="ascii")
            options = SimpleNamespace(backup_dir=directory, use_configured_target=True,
                                      resume=False, keep_existing=False, components=["mysql", "redis", "milvus"])
            first = restore_tool.restore(options)
            assert first["mysql"]["status"] == first["milvus"]["status"] == "verified"
            assert first["milvus"]["rows"] == {"rag_knowledge_base": 676}
            assert first["mysql"]["rows"]["users"] == 1
            assert "chat_conversations" in first["mysql"]["recreated_empty_tables"]
            print("PASS: application-created four empty tables and legacy schema restored correctly", flush=True)
            options.resume = True
            with pymysql.connect(**MYSQL_CONFIG, database=name) as diagnostic:
                from scripts.migrate_to_docker import mysql_snapshot
                import difflib
                actual = mysql_snapshot(diagnostic)
                for table, item in snapshot["mysql"].items():
                    if restore_tool.schema_signature(actual[table]["schema"]) != restore_tool.schema_signature(item["schema"]):
                        print("SCHEMA DIAGNOSTIC", table, flush=True)
                        print("\n".join(difflib.unified_diff(item["schema"].splitlines(), actual[table]["schema"].splitlines())), flush=True)
            assert restore_tool.restore(options)["milvus"]["status"] == "verified"
            with admin.cursor() as cursor:
                cursor.execute(f"INSERT INTO `{name}`.users (username,password_hash) VALUES (%s,%s)",
                               ("release_verification_user", "test-hash-not-a-real-account"))
            cache.set("release_verify_marker", "preserve")
            drop_vectors()
            try:
                restore_tool.restore(options)
            except ValueError:
                pass
            else:
                raise AssertionError("changed MySQL data was not protected")
            assert name not in db.list_database(using="release_audit")
            options.keep_existing = True
            repaired = restore_tool.restore(options)
            assert repaired["mysql"]["status"] == repaired["redis"]["status"] == "existing_kept"
            assert repaired["milvus"]["rows"] == {"rag_knowledge_base": 676}
            with admin.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM `{name}`.users WHERE username=%s", ("release_verification_user",))
                assert cursor.fetchone()[0] == 1
            assert cache.get("release_verify_marker") == b"preserve"
            print("PASS: missing Milvus repaired while changed MySQL and Redis data were preserved", flush=True)
        return 0
    finally:
        assert name.startswith("release_verify_")
        drop_vectors()
        with admin.cursor() as cursor:
            cursor.execute(f"DROP DATABASE `{name}`")
        cache.delete(*created_keys)
        cache.close()
        admin.close()
        connections.disconnect("release_audit")
        print("Temporary rehearsal databases cleaned; real project data unchanged.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
