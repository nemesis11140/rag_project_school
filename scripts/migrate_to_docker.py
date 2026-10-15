"""备份现有项目数据，并恢复到空的 Docker 数据库；不删除或修改源数据。

python scripts/migrate_to_docker.py backup
python scripts/migrate_to_docker.py restore --backup-dir backups/20261011-190000
"""
import argparse
import base64
import hashlib
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymysql
import redis
import numpy as np
from pymilvus import Collection, CollectionSchema, connections, db, utility

from config import DB_NAME, MILVUS_CONFIG, MYSQL_CONFIG, REDIS_CONFIG


def encode(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime, date, Decimal)):
        return {"__backup_type__": type(value).__name__, "value": str(value)}
    if isinstance(value, bytes):
        return {"__backup_type__": "bytes", "value": base64.b64encode(value).decode()}
    raise TypeError(f"Unsupported backup value: {type(value).__name__}")


def decode(value):
    kind = value.get("__backup_type__")
    constructors = {"datetime": datetime.fromisoformat, "date": date.fromisoformat,
                    "Decimal": Decimal, "bytes": base64.b64decode}
    return constructors[kind](value["value"]) if kind in constructors else value


def fingerprint(rows):
    normalized = sorted(json.dumps(row, ensure_ascii=False, sort_keys=True, default=encode)
                        for row in rows)
    return hashlib.sha256("\n".join(normalized).encode()).hexdigest()


def mysql_snapshot(conn):
    tables = {}
    with conn.cursor() as cursor:
        cursor.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY")
        cursor.execute("SHOW FULL TABLES WHERE Table_type = 'BASE TABLE'")
        names = [row[0] for row in cursor.fetchall()]
        for name in names:
            quoted = "`" + name.replace("`", "``") + "`"
            cursor.execute(f"SHOW CREATE TABLE {quoted}")
            schema = cursor.fetchone()[1]
            cursor.execute(f"SELECT * FROM {quoted}")
            columns = [column[0] for column in cursor.description]
            rows = list(cursor.fetchall())
            tables[name] = {"schema": schema, "columns": columns, "rows": rows}
    conn.rollback()
    return tables


def collection_rows(collection):
    collection.load()
    fields = [field.name for field in collection.schema.fields]
    iterator = collection.query_iterator(batch_size=256, expr="", output_fields=fields)
    rows = []
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            rows.extend(batch)
    finally:
        iterator.close()
    if collection.schema.auto_id:
        primary = collection.schema.primary_field.name
        for row in rows:
            row.pop(primary, None)
    return rows


def source_redis():
    return redis.Redis(host=REDIS_CONFIG["host"], port=REDIS_CONFIG["port"],
                       password=REDIS_CONFIG["password"] or None, db=REDIS_CONFIG["db"],
                       socket_connect_timeout=5)


def redis_content(client, key):
    kind = client.type(key).decode()
    if kind == "string":
        items = client.get(key)
    elif kind == "hash":
        items = sorted(client.hgetall(key).items())
    elif kind == "list":
        items = client.lrange(key, 0, -1)
    elif kind == "set":
        items = sorted(client.smembers(key))
    elif kind == "zset":
        items = client.zrange(key, 0, -1, withscores=True)
    else:
        raise ValueError(f"Unsupported project cache type: {kind}")
    return {"type": kind, "items": items}


def backup(args):
    backup_time = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))
    directory = Path(args.backup_dir or f"backups/{backup_time:%Y%m%d-%H%M%S}")
    directory.mkdir(parents=True, exist_ok=False)
    conn = pymysql.connect(**MYSQL_CONFIG, database=DB_NAME)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT TIMESTAMPDIFF(SECOND, UTC_TIMESTAMP(), NOW())")
            offset = int(cursor.fetchone()[0])
            mysql_time_zone = f"{'+' if offset >= 0 else '-'}{abs(offset) // 3600:02d}:{abs(offset) % 3600 // 60:02d}"
        tables = mysql_snapshot(conn)
    finally:
        conn.close()
    connections.connect(alias="backup_source", host=MILVUS_CONFIG["host"],
                        port=MILVUS_CONFIG["port"], db_name=MILVUS_CONFIG["database_name"], timeout=15)
    collections = {}
    for name in utility.list_collections(using="backup_source"):
        collection = Collection(name, using="backup_source")
        rows = collection_rows(collection)
        if len(rows) != collection.num_entities:
            raise RuntimeError(f"Collection snapshot count mismatch: {name}")
        collections[name] = {"schema": collection.schema.to_dict(), "rows": rows,
                             "indexes": [{"field": index.field_name, "params": index.params}
                                         for index in collection.indexes]}
    client = source_redis()
    keys = {REDIS_CONFIG["qa_original_key"].encode(), REDIS_CONFIG["qa_tokenized_key"].encode(),
            REDIS_CONFIG.get("qa_index_key", "qa_index").encode()}
    keys.update(client.scan_iter(match="qa_answer:*"))
    keys.update(client.scan_iter(match="qa_match:v1:*"))
    entries = []
    for key in sorted(keys):
        pipe = client.pipeline(transaction=True)
        pipe.dump(key)
        pipe.pttl(key)
        data, ttl = pipe.execute()
        if data is not None and ttl != -2:
            entries.append({"key": key, "data": data,
                            "content": redis_content(client, key),
                            "expires_at_ms": None if ttl < 0 else int(time.time() * 1000) + ttl})
    snapshot = {"format": 2, "created_at": backup_time.isoformat(), "database": DB_NAME,
                "mysql_time_zone": mysql_time_zone,
                "mysql": tables, "milvus_database": MILVUS_CONFIG["database_name"],
                "milvus": collections, "redis": entries}
    path = directory / "snapshot.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False, default=encode), encoding="utf-8")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "snapshot.sha256").write_text(checksum + "\n", encoding="ascii")
    print("Backup:", directory.resolve())
    print("MySQL:", {name: len(item["rows"]) for name, item in tables.items()})
    print("Milvus:", {name: len(item["rows"]) for name, item in collections.items()})
    print("Redis project keys:", len(entries))


def restore(args):
    from scripts.restore_components import restore as restore_components
    return restore_components(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["backup", "restore"])
    parser.add_argument("--backup-dir")
    parser.add_argument("--resume", action="store_true", help="验证已有目标内容相同后恢复未完成部分，不覆盖数据")
    parser.add_argument("--components", nargs="+", choices=["mysql", "redis", "milvus"],
                        default=["mysql", "redis", "milvus"], help="仅恢复指定组件，不连接其他数据库")
    parser.add_argument("--keep-existing", action="store_true",
                        help="保留已有业务数据的整个组件；兼容空表和缺失组件仍可恢复")
    parser.add_argument("--use-configured-target", action="store_true", help="云端恢复使用环境变量配置的 Compose 数据库地址，仍拒绝覆盖已有数据")
    parser.add_argument("--mysql-host", default="127.0.0.1")
    parser.add_argument("--mysql-port", type=int, default=3307)
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6380)
    parser.add_argument("--milvus-host", default="127.0.0.1")
    parser.add_argument("--milvus-port", type=int, default=19531)
    args = parser.parse_args()
    (backup if args.action == "backup" else restore)(args)


if __name__ == "__main__":
    main()
