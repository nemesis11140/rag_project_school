import os
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility, db
from typing import List, Dict, Any, Optional
from config import MILVUS_CONFIG
from ..common.embedding import embedding_model_singleton
from ..common.startup import StartupIssue

# Milvus 单次 insert 的数据量上限较保守，分批写入避免请求过大
INSERT_BATCH_SIZE = 512
# 向量化批量大小
ENCODE_BATCH_SIZE = 32


class VectorStoreManager:
    """Milvus 向量库封装（父子分块 schema）。

    唯一权威 schema：child_text / parent_text / document_name /
    parent_index / child_index / vector(dim)。检索侧（HybridRetriever）
    必须与之保持一致。
    """

    def __init__(self):
        self.db_name = MILVUS_CONFIG.get("database_name", "default")
        self.collection_name = MILVUS_CONFIG["collection_name"]
        self.host = MILVUS_CONFIG["host"]
        self.port = str(MILVUS_CONFIG["port"])
        self.conn_alias = "my_rag_vector_store"
        self.collection = None
        self.embedding_model = None
        self.dimension: Optional[int] = None

    # ---------- 连接 ----------

    def check_ready(self):
        """只读检查项目库和集合，不通过创建空库掩盖未恢复数据。"""
        probe = self.conn_alias + "_probe"
        try:
            connections.connect(alias=probe, host=self.host, port=self.port,
                                db_name="default", timeout=10)
            if self.db_name not in db.list_database(using=probe):
                raise StartupIssue("knowledge_database_missing", "milvus",
                                   f"知识库尚未恢复：缺少 Milvus 数据库 {self.db_name}",
                                   "使用部署脚本恢复备份；已恢复 MySQL 时可只补恢复 Milvus。")
            db.using_database(self.db_name, using=probe)
            if not utility.has_collection(self.collection_name, using=probe):
                raise StartupIssue("knowledge_collection_missing", "milvus",
                                   f"知识库尚未恢复：缺少集合 {self.collection_name}",
                                   "先恢复 Milvus 备份，或在首次初始化时构建知识库。")
            if Collection(self.collection_name, using=probe).num_entities == 0:
                raise StartupIssue("knowledge_collection_empty", "milvus",
                                   "知识库集合为空，尚无文档向量",
                                   "恢复向量备份，或在首次初始化时导入文档。")
        except StartupIssue:
            raise
        except Exception as error:
            raise StartupIssue("milvus_unavailable", "milvus", "无法连接或读取 Milvus 知识库",
                               "检查 Milvus、etcd、MinIO 的容器状态与连接配置。") from error
        finally:
            if connections.has_connection(probe):
                connections.disconnect(probe)

    def _connect(self):
        if not connections.has_connection(self.conn_alias):
            connections.connect(
                host=self.host,
                port=self.port,
                db_name=self.db_name,
                alias=self.conn_alias,
                timeout=10
            )
            print(f"[VectorStore] Connected to database: {self.db_name}")

    def _ensure_collection(self):
        if self.collection is None and utility.has_collection(self.collection_name, using=self.conn_alias):
            self.collection = Collection(self.collection_name, using=self.conn_alias)

    # ---------- 模型 ----------

    def load_embedding_model(self):
        self.embedding_model = embedding_model_singleton.get_model()
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        print(f"[VectorStore] Embedding dimension: {self.dimension}")
        return self.embedding_model

    def _ensure_model(self):
        if self.embedding_model is None:
            self.load_embedding_model()

    # ---------- 建库 ----------

    def create_collection(self, drop_existing: bool = True):
        self._connect()
        self._ensure_model()

        if drop_existing and utility.has_collection(self.collection_name, using=self.conn_alias):
            print(f"[VectorStore] Dropping existing collection: {self.collection_name}")
            Collection(self.collection_name, using=self.conn_alias).drop()

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="child_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="parent_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="document_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="parent_index", dtype=DataType.INT64),
            FieldSchema(name="child_index", dtype=DataType.INT64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Document chunks with parent-child relationship"
        )
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using=self.conn_alias
        )
        print(f"[VectorStore] Collection '{self.collection_name}' created in '{self.db_name}'")

        self._create_index()
        return self.collection

    def _create_index(self):
        if self.collection is None:
            return
        if self.collection.has_index():
            return
        # FLAT + COSINE：语料量级（万级 chunk 以下）下精度最高、无需调参
        self.collection.create_index(
            field_name="vector",
            index_params={"metric_type": "COSINE", "index_type": "FLAT", "params": {}}
        )
        print("[VectorStore] Vector index created")

    def load(self):
        if self.collection is None:
            return
        self.collection.load()
        utility.wait_for_loading_complete(
            self.collection_name, using=self.conn_alias, timeout=60
        )

    # ---------- 写入 ----------

    def insert_chunks(self, chunks: List[Dict[str, Any]]):
        self._connect()
        self._ensure_model()

        if not chunks:
            print("[VectorStore] No chunks to insert")
            return 0

        self._ensure_collection()
        if self.collection is None:
            raise RuntimeError(
                f"Collection '{self.collection_name}' not found in database '{self.db_name}'"
            )

        # 校验维度：避免出现"建库 384 / 检索 1024"这类静默失效
        collection_dim = None
        for field in self.collection.schema.fields:
            if field.name == "vector":
                collection_dim = field.params.get("dim")
        if collection_dim and int(collection_dim) != int(self.dimension):
            raise ValueError(
                f"维度不匹配：集合 '{self.collection_name}' 为 {collection_dim} 维，"
                f"当前模型输出 {self.dimension} 维。请先重建集合。"
            )

        print(f"[VectorStore] Encoding {len(chunks)} child chunks...")
        child_texts = [chunk["child_text"] for chunk in chunks]
        vectors = self.embedding_model.encode(
            child_texts,
            batch_size=ENCODE_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(child_texts) > 200
        )

        data = []
        for i, chunk in enumerate(chunks):
            data.append({
                "child_text": chunk["child_text"],
                "parent_text": chunk["parent_text"],
                "document_name": chunk["document_name"],
                "parent_index": int(chunk["parent_index"]),
                "child_index": int(chunk["child_index"]),
                "vector": vectors[i].tolist()
            })

        inserted = 0
        for start in range(0, len(data), INSERT_BATCH_SIZE):
            batch = data[start:start + INSERT_BATCH_SIZE]
            self.collection.insert(batch)
            inserted += len(batch)
            print(f"[VectorStore] Inserted {inserted}/{len(data)}")

        self.collection.flush()
        self._create_index()
        self.load()
        print(f"[VectorStore] Successfully inserted {inserted} chunks")
        return inserted

    # ---------- 检索 ----------

    def search(self, query: str, top_k: int = 5,
               filter_document: Optional[str] = None) -> List[Dict[str, Any]]:
        self._connect()
        self._ensure_model()
        self._ensure_collection()
        if self.collection is None:
            raise RuntimeError(
                f"Collection '{self.collection_name}' not found in database '{self.db_name}'"
            )

        query_vector = self.embedding_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )

        filter_expr = None
        if filter_document:
            filter_expr = f'document_name == "{filter_document}"'

        results = self.collection.search(
            data=query_vector.tolist(),
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {}},
            limit=top_k,
            output_fields=[
                "child_text", "parent_text", "document_name",
                "parent_index", "child_index"
            ],
            expr=filter_expr
        )

        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "dense_score": hit.distance,
                    "child_text": hit.entity.get("child_text"),
                    "parent_text": hit.entity.get("parent_text"),
                    "document_name": hit.entity.get("document_name"),
                    "parent_index": hit.entity.get("parent_index"),
                    "child_index": hit.entity.get("child_index"),
                    "source": "dense"
                })

        return formatted_results

    def query_all(self, fields: Optional[List[str]] = None,
                  limit: int = 16384) -> List[Dict[str, Any]]:
        """拉取全量标量字段（供内存 BM25 建索引使用）。"""
        self._connect()
        self._ensure_collection()
        if self.collection is None:
            return []

        fields = fields or ["child_text", "parent_text", "document_name",
                            "parent_index", "child_index"]
        return self.collection.query(expr="", output_fields=fields, limit=limit)

    def get_collection_stats(self):
        self._connect()
        if utility.has_collection(self.collection_name, using=self.conn_alias):
            c = Collection(self.collection_name, using=self.conn_alias)
            return {"row_count": c.num_entities, "collection": self.collection_name}
        return {"row_count": 0, "collection": self.collection_name}

    def has_collection(self):
        self._connect()
        return utility.has_collection(self.collection_name, using=self.conn_alias)

    def get_schema_fields(self) -> List[str]:
        """返回线上集合的字段名，用于启动自检。"""
        self._connect()
        self._ensure_collection()
        if self.collection is None:
            return []
        return [f.name for f in self.collection.schema.fields]

    def __del__(self):
        try:
            if connections.has_connection(self.conn_alias):
                connections.disconnect(self.conn_alias)
        except Exception:
            pass
