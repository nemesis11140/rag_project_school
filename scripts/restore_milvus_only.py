"""从项目快照单独恢复 Milvus。校验全部向量；不连接 MySQL/Redis，不覆盖不同内容。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from pymilvus import Collection, CollectionSchema, connections, db, utility

# 支持脚本挂载到 /app/restore_milvus_only.py 或在项目 scripts/ 下直接运行。
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR if (SCRIPT_DIR / "config.py").exists() else SCRIPT_DIR.parent))
from config import MILVUS_CONFIG

ALIAS = "milvus_backup_repair"


def fingerprint(rows):
    def encode(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    ordered = sorted(json.dumps(row, ensure_ascii=False, sort_keys=True, default=encode)
                     for row in rows)
    return hashlib.sha256("\n".join(ordered).encode()).hexdigest()


def read_snapshot(directory):
    directory = Path(directory)
    raw = (directory / "snapshot.json").read_bytes()
    checksum = (directory / "snapshot.sha256").read_text(encoding="ascii").strip()
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("备份校验失败，未执行恢复。请重新上传完整备份。")
    snapshot = json.loads(raw)
    if snapshot.get("format") != 2 or not snapshot.get("milvus"):
        raise ValueError("备份格式不正确或没有向量集合，未执行恢复。")
    # 在连接服务器前确认所有集合结构能解析，且备份包含必要的索引。
    for name, item in snapshot["milvus"].items():
        CollectionSchema.construct_from_dict(item["schema"])
        if not item.get("indexes"):
            raise ValueError(f"集合 {name} 缺少索引配置，未执行恢复。")
    return snapshot


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
        for row in rows:
            row.pop(collection.schema.primary_field.name, None)
    return rows


def restore(directory, config=None):
    config = config or MILVUS_CONFIG
    snapshot = read_snapshot(directory)
    target = config["database_name"]
    if snapshot["milvus_database"] != target:
        raise ValueError(f"备份库名 {snapshot['milvus_database']} 与应用库名 {target} 不一致，未执行恢复。")
    if config["collection_name"] not in snapshot["milvus"]:
        raise ValueError("备份缺少应用需要的集合，未执行恢复。")
    connections.connect(alias=ALIAS, host=config["host"], port=str(config["port"]),
                        db_name="default", timeout=15)
    verified = set()
    try:
        exists = target in db.list_database(using=ALIAS)
        if exists:
            db.using_database(target, using=ALIAS)
            current = set(utility.list_collections(using=ALIAS))
            if current - set(snapshot["milvus"]):
                raise ValueError("目标库含备份之外的集合，停止恢复，未覆盖已有内容。")
            # 先验证所有已有集合。仅允许完全一致的集合，方便安全重复执行。
            for name in current:
                collection = Collection(name, using=ALIAS)
                item = snapshot["milvus"][name]
                expected_schema = CollectionSchema.construct_from_dict(item["schema"]).to_dict()
                if collection.schema.to_dict() != expected_schema:
                    raise ValueError(f"目标集合 {name} 的结构与备份不同，未覆盖。")
                if fingerprint(collection_rows(collection)) != fingerprint(item["rows"]):
                    raise ValueError(f"目标集合 {name} 的数据与备份不同，未覆盖。")
                verified.add(name)
        else:
            db.create_database(target, using=ALIAS)
            db.using_database(target, using=ALIAS)

        report = {}
        for name, item in snapshot["milvus"].items():
            if name not in verified:
                collection = Collection(name, using=ALIAS,
                                        schema=CollectionSchema.construct_from_dict(item["schema"]))
                for index in item["indexes"]:
                    collection.create_index(index["field"], index["params"])
                for start in range(0, len(item["rows"]), 256):
                    collection.insert(item["rows"][start:start + 256])
                collection.flush()
                actual = collection_rows(collection)
                if fingerprint(actual) != fingerprint(item["rows"]):
                    raise RuntimeError(f"集合 {name} 写入后的内容校验失败，请保留日志。")
            report[name] = len(item["rows"])
            print(f"集合 {name}：{report[name]} 条，{'已有内容一致' if name in verified else '恢复并校验成功'}", flush=True)
        print(f"Milvus 恢复并校验完成，数据库：{target}。未连接 MySQL 或 Redis。", flush=True)
        return report
    finally:
        connections.disconnect(ALIAS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-dir", required=True)
    args = parser.parse_args()
    restore(args.backup_dir)


if __name__ == "__main__":
    main()
