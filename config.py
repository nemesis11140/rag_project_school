import os
import configparser
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_INI = os.path.join(BASE_DIR, "config.ini")

# .env 里的密钥优先于 config.ini（config.ini 会被提交进 git，不能放密码）
sys.path.insert(0, BASE_DIR)
from src.common.env import load_dotenv  # noqa: E402

load_dotenv(os.path.join(BASE_DIR, ".env"))

config = configparser.ConfigParser()
config.read(CONFIG_INI, encoding='utf-8')


def secret(section: str, key: str, env_var: str, fallback: str = "") -> str:
    """读取敏感配置：真实环境变量 > .env > config.ini > fallback。"""
    value = os.environ.get(env_var)
    if value:
        return value
    try:
        return config.get(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return fallback


def setting(section: str, key: str, env_var: str, fallback: str = "") -> str:
    """读取普通配置，允许用环境变量覆盖（容器部署靠这个改 host）。"""
    value = os.environ.get(env_var)
    if value:
        return value
    try:
        return config.get(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return fallback


def setting_int(section: str, key: str, env_var: str, fallback: int = 0) -> int:
    raw = setting(section, key, env_var)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return fallback


MYSQL_CONFIG = {
    "host": setting("mysql", "host", "MYSQL_HOST", "localhost"),
    "port": setting_int("mysql", "port", "MYSQL_PORT", 3306),
    "user": setting("mysql", "user", "MYSQL_USER", "root"),
    "password": secret("mysql", "password", "MYSQL_PASSWORD"),
    "charset": config.get("mysql", "charset")
}

DB_NAME = config.get("database", "db_name")
TABLE_NAME = config.get("database", "table_name")

DATA_DIR = os.path.join(BASE_DIR, config.get("data", "data_dir"))
MODEL_DIR = os.path.join(BASE_DIR, config.get("model", "model_dir"))

def _json_setting(section: str, key: str, fallback: dict = None) -> dict:
    """读取 JSON 形式的配置项，解析失败时返回 fallback 而不是让整个服务起不来。"""
    raw = config.get(section, key, fallback="").strip()
    if not raw:
        return fallback if fallback is not None else {}
    try:
        import json

        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except Exception as e:
        print(f"[config] {section}.{key} 不是合法 JSON，已忽略: {e}")
        return fallback if fallback is not None else {}


LLM_CONFIG = {
    "api_url": config.get("llm", "api_url"),
    "api_key": secret("llm", "api_key", "LLM_API_KEY"),
    "model_name": config.get("llm", "model_name"),
    "temperature": config.getfloat("llm", "temperature"),
    "max_tokens": config.getint("llm", "max_tokens"),
    # 透传给服务商的额外请求参数（如关闭推理模型的思考过程）
    "extra_body": _json_setting("llm", "extra_body")
}

QA_GENERATION_CONFIG = {
    "num_pairs": config.getint("qa_generation", "num_pairs"),
    "min_line_length": config.getint("qa_generation", "min_line_length"),
    "max_question_length": config.getint("qa_generation", "max_question_length"),
    "max_answer_length": config.getint("qa_generation", "max_answer_length")
}

REDIS_CONFIG = {
    "host": setting("redis", "host", "REDIS_HOST", "localhost"),
    "port": setting_int("redis", "port", "REDIS_PORT", 6379),
    "password": secret("redis", "password", "REDIS_PASSWORD"),
    "db": setting_int("redis", "db", "REDIS_DB", 1),
    "qa_original_key": config.get("redis", "qa_original_key"),
    "qa_tokenized_key": config.get("redis", "qa_tokenized_key"),
    "qa_index_key": config.get("redis", "qa_index_key", fallback="qa_index")
}

MILVUS_CONFIG = {
    "host": setting("milvus", "host", "MILVUS_HOST", "localhost"),
    "port": setting_int("milvus", "port", "MILVUS_PORT", 19530),
    "database_name": config.get("milvus", "database_name"),
    "collection_name": config.get("milvus", "collection_name")
}

DOCUMENT_CONFIG = {
    "min_page_chars": config.getint("document", "min_page_chars", fallback=20),
    "ocr_fallback": config.getboolean("document", "ocr_fallback", fallback=True),
    "ocr_lang": config.get("document", "ocr_lang", fallback="ch"),
    "ocr_dpi": config.getint("document", "ocr_dpi", fallback=200),
}

CHUNKING_CONFIG = {
    "parent_chunk_size": config.getint("chunking", "parent_chunk_size"),
    "parent_chunk_overlap": config.getint("chunking", "parent_chunk_overlap"),
    "child_chunk_size": config.getint("chunking", "child_chunk_size"),
    "child_chunk_overlap": config.getint("chunking", "child_chunk_overlap"),
    "min_child_chunk_size": config.getint("chunking", "min_child_chunk_size", fallback=20)
}

EMBEDDING_CONFIG = {
    "model_name": config.get("embedding", "model_name"),
    "model_path": os.path.join(MODEL_DIR, config.get("embedding", "model_path")),
    "device": config.get("embedding", "device")
}

INTENT_MODEL_CONFIG = {
    "model_path": os.path.join(MODEL_DIR, config.get("intent_model", "model_path")),
    "base_model_path": os.path.join(MODEL_DIR, config.get("intent_model", "base_model_path")),
    "backend": config.get("intent_model", "backend", fallback="auto"),
    "device": config.get("intent_model", "device", fallback="cpu"),
    "confidence_threshold": config.getfloat("intent_model", "confidence_threshold", fallback=0.6)
}

RERANKER_CONFIG = {
    "model_path": os.path.join(MODEL_DIR, config.get("reranker", "model_path")),
    "device": config.get("reranker", "device", fallback="cpu"),
    "batch_size": config.getint("reranker", "batch_size", fallback=16),
    "max_length": config.getint("reranker", "max_length", fallback=512),
    "max_docs": config.getint("reranker", "max_docs", fallback=8)
}

SYSTEM_CONFIG = {
    "bm25_threshold": config.getfloat("system", "bm25_threshold"),
    "keyword_overlap_threshold": config.getfloat("system", "keyword_overlap_threshold"),
    "faq_candidate_count": config.getint("system", "faq_candidate_count", fallback=5),
    "faq_semantic_min_confidence": config.getfloat("system", "faq_semantic_min_confidence", fallback=0.9),
    "faq_semantic_timeout": config.getint("system", "faq_semantic_timeout", fallback=15),
    "default_top_k": config.getint("system", "default_top_k"),
    "recall_multiplier": config.getint("system", "recall_multiplier", fallback=3),
    "rewrite_max_queries": config.getint("system", "rewrite_max_queries", fallback=2),
    "context_max_chars": config.getint("system", "context_max_chars", fallback=4000)
}
