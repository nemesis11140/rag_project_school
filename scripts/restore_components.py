"""按组件规划恢复：兼容空表可复用；已有业务数据可显式保留；不同数据不覆盖。"""
import hashlib
import json
import re
from pathlib import Path

import pymysql
import redis
from pymilvus import Collection, CollectionSchema, connections, db, utility

from config import DB_NAME, MILVUS_CONFIG, MYSQL_CONFIG, REDIS_CONFIG
from scripts.migrate_to_docker import decode, fingerprint, mysql_snapshot, collection_rows, redis_content


def schema_signature(schema):
    schema = re.sub(r"\sAUTO_INCREMENT=\d+", "", schema, flags=re.I)
    # SHOW CREATE TABLE 在不同 MySQL 版本中可能显式输出冗余的列字符集。
    # 只消除与表默认值一致的声明；不同的字符集/排序规则仍会被识别为冲突。
    charset = re.search(r"DEFAULT CHARSET=(\w+)", schema, flags=re.I)
    collation = re.search(r"\bCOLLATE=(\w+)", schema, flags=re.I)
    if charset:
        schema = re.sub(r"\sCHARACTER SET " + re.escape(charset.group(1)) + r"\b", "", schema, flags=re.I)
    if collation:
        schema = re.sub(r"\sCOLLATE " + re.escape(collation.group(1)) + r"\b", "", schema, flags=re.I)
    return re.sub(r"\s+", " ", schema).strip().rstrip(";").lower()


def mysql_plan(actual, expected, resume=False, keep_existing=False):
    if keep_existing and any(item["rows"] for item in actual.values()):
        return {"skip": True, "create": [], "insert": [], "recreate": []}
    if set(actual) - set(expected):
        raise ValueError("MySQL 存在备份之外的表，拒绝自动修改。")
    create, insert, recreate = [], [], []
    all_empty = not any(item["rows"] for item in actual.values())
    auth_tables = {"users", "user_sessions", "chat_conversations", "chat_messages"}
    for name, item in expected.items():
        if name not in actual:
            create.append(name)
            insert.append(name)
            continue
        existing = actual[name]
        if schema_signature(existing["schema"]) != schema_signature(item["schema"]):
            # 旧匿名会话迁移后保留 owner_id/user_id nullable 等历史结构。
            # 仅整个目标库为空且字段属于备份内的已知认证表时，安全重建空表。
            known_empty_auth = all_empty and name in auth_tables and existing.get("columns") and (
                set(existing["columns"]) <= set(item.get("columns", []))
            )
            if not known_empty_auth:
                raise ValueError(f"MySQL 表 {name} 的结构与备份不兼容，未覆盖。")
            recreate.append(name)
        if not existing["rows"]:
            insert.append(name)
        elif not resume or fingerprint(existing["rows"]) != fingerprint(item["rows"]):
            raise ValueError(f"MySQL 表 {name} 已有数据；只允许 --resume 校验一致的数据，或 --keep-existing 保留现有库。")
    return {"skip": False, "create": create, "insert": insert, "recreate": recreate, "existing": list(actual)}


def restore(args):
    directory = Path(args.backup_dir or "")
    if not args.backup_dir:
        raise ValueError("恢复需要 --backup-dir")
    raw = (directory / "snapshot.json").read_bytes()
    checksum = (directory / "snapshot.sha256").read_text(encoding="ascii").strip()
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("备份 SHA-256 校验失败，未执行恢复。")
    snapshot = json.loads(raw, object_hook=decode)
    if snapshot.get("format") != 2 or snapshot["database"] != DB_NAME:
        raise ValueError("备份格式或 MySQL 库名与当前项目不一致。")
    selected = set(getattr(args, "components", None) or ["mysql", "redis", "milvus"])
    keep = getattr(args, "keep_existing", False)
    configs = {"mysql": MYSQL_CONFIG, "redis": REDIS_CONFIG, "milvus": MILVUS_CONFIG}
    targets = {}
    for service in selected:
        configured = configs[service]
        host = configured["host"] if args.use_configured_target else getattr(args, service + "_host")
        port = configured["port"] if args.use_configured_target else getattr(args, service + "_port")
        same_host = host == configured["host"] or {host, configured["host"]} <= {"localhost", "127.0.0.1", "::1"}
        if not args.use_configured_target and same_host and int(port) == int(configured["port"]):
            raise ValueError(f"{service} 目标与源服务一致，拒绝恢复。")
        targets[service] = (host, port)
    conn, cache, alias = None, None, "component_restore"
    mysql_state, redis_keys, milvus_state = None, set(), None
    report = {}
    try:
        # 所选组件全部完成只读预检后才开始写入。
        if "mysql" in selected:
            host, port = targets["mysql"]
            conn = pymysql.connect(**{**MYSQL_CONFIG, "host": host, "port": port}, database=DB_NAME)
            with conn.cursor() as cursor:
                cursor.execute("SET SESSION time_zone = %s", (snapshot["mysql_time_zone"],))
            mysql_state = mysql_plan(mysql_snapshot(conn), snapshot["mysql"], args.resume, keep)
        if "redis" in selected:
            host, port = targets["redis"]
            cache = redis.Redis(host=host, port=port, password=REDIS_CONFIG["password"] or None,
                                db=REDIS_CONFIG["db"], socket_connect_timeout=5)
            redis_keys = set(cache.scan_iter())
            if not (keep and redis_keys):
                expected_keys = {item["key"] for item in snapshot["redis"]}
                if redis_keys and not args.resume or redis_keys - expected_keys:
                    raise ValueError("Redis 目标存在其他内容，未覆盖；可选择 --keep-existing 保留该组件。")
                for item in snapshot["redis"]:
                    if item["key"] in redis_keys and fingerprint([redis_content(cache, item["key"])]) != fingerprint([item["content"]]):
                        raise ValueError("Redis 已有内容与备份不同，未覆盖。")
        if "milvus" in selected:
            if snapshot["milvus_database"] != MILVUS_CONFIG["database_name"]:
                raise ValueError("备份的 Milvus 库名与应用配置不一致，未执行恢复。")
            if MILVUS_CONFIG["collection_name"] not in snapshot["milvus"]:
                raise ValueError("备份不含应用需要的 Milvus 集合。")
            host, port = targets["milvus"]
            connections.connect(alias=alias, host=host, port=str(port), db_name="default", timeout=15)
            exists = snapshot["milvus_database"] in db.list_database(using=alias)
            current = set()
            if exists:
                db.using_database(snapshot["milvus_database"], using=alias)
                current = set(utility.list_collections(using=alias))
            populated = any(Collection(name, using=alias).num_entities for name in current)
            milvus_state = {"exists": exists, "current": current, "verified": set(), "skip": bool(keep and populated)}
            if not milvus_state["skip"]:
                if current - set(snapshot["milvus"]):
                    raise ValueError("Milvus 存在备份之外的集合，未覆盖。")
                for name, item in snapshot["milvus"].items():
                    expected_schema = CollectionSchema.construct_from_dict(item["schema"])
                    if not item.get("indexes"):
                        raise ValueError(f"Milvus 集合 {name} 缺少索引配置。")
                    if name not in current:
                        continue
                    collection = Collection(name, using=alias)
                    if collection.schema.to_dict() != expected_schema.to_dict():
                        raise ValueError(f"Milvus 集合 {name} 的结构与备份不同。")
                    if collection.num_entities:
                        if not args.resume or fingerprint(collection_rows(collection)) != fingerprint(item["rows"]):
                            raise ValueError(f"Milvus 集合 {name} 已有不同内容，未覆盖。")
                        milvus_state["verified"].add(name)

        if mysql_state is not None:
            if mysql_state["skip"]:
                report["mysql"] = {"status": "existing_kept"}
            else:
                with conn.cursor() as cursor:
                    cursor.execute("SET FOREIGN_KEY_CHECKS=0")
                    try:
                        if mysql_state["recreate"]:
                            for name in mysql_state["existing"]:
                                quoted = "`" + name.replace("`", "``") + "`"
                                cursor.execute(f"SELECT 1 FROM {quoted} LIMIT 1")
                                if cursor.fetchone():
                                    raise ValueError("预检后发现新增数据，停止空表结构重建。请先停止写入。")
                        for name, item in snapshot["mysql"].items():
                            if name in mysql_state["recreate"]:
                                quoted = "`" + name.replace("`", "``") + "`"
                                cursor.execute(f"DROP TABLE {quoted}")
                            if name in mysql_state["create"] or name in mysql_state["recreate"]:
                                cursor.execute(item["schema"])
                            if name not in mysql_state["insert"] or not item["rows"]:
                                continue
                            table = "`" + name.replace("`", "``") + "`"
                            columns = ",".join("`" + column.replace("`", "``") + "`" for column in item["columns"])
                            values = ",".join(["%s"] * len(item["columns"]))
                            cursor.executemany(f"INSERT INTO {table} ({columns}) VALUES ({values})", item["rows"])
                    finally:
                        cursor.execute("SET FOREIGN_KEY_CHECKS=1")
                conn.commit()
                actual = mysql_snapshot(conn)
                for name, item in snapshot["mysql"].items():
                    if fingerprint(actual[name]["rows"]) != fingerprint(item["rows"]):
                        raise RuntimeError(f"MySQL 表 {name} 恢复后内容校验失败。")
                report["mysql"] = {"status": "verified", "recreated_empty_tables": mysql_state["recreate"],
                                   "rows": {name: len(item["rows"]) for name, item in snapshot["mysql"].items()}}

        if milvus_state is not None:
            if milvus_state["skip"]:
                report["milvus"] = {"status": "existing_kept"}
            else:
                if not milvus_state["exists"]:
                    db.create_database(snapshot["milvus_database"], using=alias)
                    db.using_database(snapshot["milvus_database"], using=alias)
                for name, item in snapshot["milvus"].items():
                    if name in milvus_state["verified"]:
                        continue
                    collection = Collection(name, using=alias) if name in milvus_state["current"] else Collection(
                        name, using=alias, schema=CollectionSchema.construct_from_dict(item["schema"]))
                    if not collection.has_index():
                        for index in item["indexes"]:
                            collection.create_index(index["field"], index["params"])
                    for start in range(0, len(item["rows"]), 256):
                        collection.insert(item["rows"][start:start + 256])
                    collection.flush()
                    if fingerprint(collection_rows(collection)) != fingerprint(item["rows"]):
                        raise RuntimeError(f"Milvus 集合 {name} 恢复后内容校验失败。")
                report["milvus"] = {"status": "verified", "rows": {name: len(item["rows"]) for name, item in snapshot["milvus"].items()}}

        if cache is not None:
            if keep and redis_keys:
                report["redis"] = {"status": "existing_kept"}
            else:
                import time
                restored = 0
                for item in snapshot["redis"]:
                    expiry = item["expires_at_ms"]
                    ttl = 0 if expiry is None else expiry - int(time.time() * 1000)
                    if expiry is not None and ttl <= 0:
                        continue
                    if item["key"] not in redis_keys:
                        cache.restore(item["key"], ttl, item["data"])
                    if fingerprint([redis_content(cache, item["key"])]) != fingerprint([item["content"]]):
                        raise RuntimeError("Redis 恢复后内容校验失败。")
                    restored += 1
                report["redis"] = {"status": "verified", "keys": restored}
        print(json.dumps({"restore": report}, ensure_ascii=False, indent=2), flush=True)
        return report
    except Exception:
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if conn is not None:
            conn.close()
        if cache is not None:
            cache.close()
        if connections.has_connection(alias):
            connections.disconnect(alias)
