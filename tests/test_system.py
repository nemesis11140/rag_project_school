"""系统回归测试。

重点覆盖本次修复过的坑，避免回归：
  - 递归切割器必须终止（旧实现是死循环）
  - 切割结果长度受控
  - 同一份特征函数（训练/推理一致）
  - 高频问答 CSV 原问题 100% 自匹配
  - Milvus schema 与代码一致、dense 检索真的能返回结果
  - query 归一化生效

运行： python tests/test_system.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Callable, List, Tuple

RESULTS: List[Tuple[str, bool, str]] = []


def check(name: str, fn: Callable[[], str]):
    print(f"\n{'=' * 70}\n[测试] {name}\n{'=' * 70}")
    try:
        detail = fn()
        RESULTS.append((name, True, detail or ""))
        print(f"  ✅ 通过 {detail or ''}")
    except AssertionError as e:
        RESULTS.append((name, False, str(e)))
        print(f"  ❌ 断言失败: {e}")
    except Exception as e:
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  ❌ 异常: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------

def test_config():
    from config import MILVUS_CONFIG, REDIS_CONFIG, SYSTEM_CONFIG

    assert MILVUS_CONFIG["database_name"] == "rag_project_school", MILVUS_CONFIG
    assert REDIS_CONFIG["db"] == 1, REDIS_CONFIG
    assert SYSTEM_CONFIG["bm25_threshold"] > 0
    return (f"milvus={MILVUS_CONFIG['database_name']} redis_db={REDIS_CONFIG['db']} "
            f"threshold={SYSTEM_CONFIG['bm25_threshold']}")


def test_chunker_terminates():
    """回归：旧实现 start = end - overlap 永不退出，会挂死整个服务。"""
    from src.knowledge_base.text_chunker import ParentChildChunker, RecursiveTextSplitter

    chunker = ParentChildChunker()
    cases = [
        "短句。",
        "签到与退票。学生应当按时参加各项活动。",
        "第一段。" + "很长的正文内容" * 200,
        "无标点" * 500,
        "",
        "   ",
    ]

    for text in cases:
        started = time.time()
        chunker.split_text(text, "t.pdf")
        elapsed = time.time() - started
        assert elapsed < 5, f"输入长度 {len(text)} 用时 {elapsed:.1f}s，疑似死循环"

    splitter = RecursiveTextSplitter(50, 10)
    for text in cases:
        chunks = splitter.split_text(text)
        assert all(len(c) <= 50 for c in chunks), f"存在超过 chunk_size 的块: {[len(c) for c in chunks]}"

    # overlap 必须小于 chunk_size，否则构造就应报错
    try:
        RecursiveTextSplitter(100, 100)
    except ValueError:
        pass
    else:
        raise AssertionError("chunk_overlap >= chunk_size 应当抛 ValueError")

    return "6 组边界输入均正常终止，块长受控"


def test_chunker_output_shape():
    from src.knowledge_base.text_chunker import ParentChildChunker

    chunker = ParentChildChunker()
    text = "第一条 为规范学生管理。\n" * 200
    chunks = chunker.split_text(text, "doc.pdf")

    assert chunks, "未产出任何 chunk"
    for chunk in chunks:
        assert chunk["child_text"].strip()
        assert chunk["parent_text"].strip()
        assert chunk["child_text"] in chunk["parent_text"], "子块必须是父块的子串"
        assert chunk["document_name"] == "doc.pdf"

    parents = {c["parent_index"] for c in chunks}
    return f"父块 {len(parents)} / 子块 {len(chunks)}"


def test_document_loader():
    from config import DATA_DIR
    from src.knowledge_base.document_loader import DocumentLoader

    loader = DocumentLoader(ocr_fallback=False)
    pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    assert pdfs, f"{DATA_DIR} 下没有 PDF"

    details = []
    for name in pdfs:
        text, pages = loader.load_document(os.path.join(DATA_DIR, name))
        assert len(text) > 10000, f"{name} 只抽到 {len(text)} 字，预处理可能过度删除"
        details.append(f"{name}:{len(pages)}页/{len(text)}字")
    return " ".join(details)


def test_normalization():
    from src.common.text_utils import normalize, tokenize

    assert normalize("如何申请奖学金？") == normalize("怎么申请奖学金？"), "如何/怎么 未归一"
    assert "丢失" in normalize("我弄丢了学生证"), "丢了/弄丢 未归一"
    assert normalize("一卡通怎么补办") == normalize("校园卡怎么补办"), "一卡通 未归一"
    assert normalize("A。B") == "AB", "标点未清洗"

    # 不能过度归一：学费多少钱 不应被改写成别的词
    assert normalize("学费多少钱") == "学费多少", normalize("学费多少钱")

    q_tokens = tokenize(normalize("如何申请奖学金"), drop_question_words=True)
    assert set(q_tokens) == {"申请", "奖学金"}, q_tokens
    return f"归一化正常，示例 token={q_tokens}"


def test_intent_features_shared():
    """回归：训练/推理词表不一致会导致特征漂移、误判。"""
    import inspect

    from src.common import intent_features
    from src.rag_qa import intent_classifier

    assert "from ..common.intent_features import" in inspect.getsource(intent_classifier), \
        "推理侧必须从公共模块引特征"

    train_src = open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "train_intent_sklearn.py"), encoding="utf-8"
    ).read()
    assert "from src.common.intent_features import" in train_src, "训练脚本必须共用特征模块"
    assert "def extract_features" not in train_src, "训练脚本不应再自带一份 extract_features"

    feats = intent_features.extract_features("学费要交多少？")
    assert len(feats) == 4 and feats[0] > 0, feats
    return f"特征维度 {len(feats)}，学费类问题专业词命中 {int(feats[0])}"


def test_intent_classifier():
    from src.common.intent_features import INTENT_CHITCHAT, INTENT_CONSULT
    from src.rag_qa.intent_classifier import IntentClassifier

    classifier = IntentClassifier()
    cases = [
        ("今天天气怎么样？", INTENT_CHITCHAT),
        ("怎么做红烧肉？", INTENT_CHITCHAT),
        ("食堂中午几点开门？", INTENT_CONSULT),
        ("学费要交多少？", INTENT_CONSULT),
        ("图书馆的借书期限是多久？", INTENT_CONSULT),
        ("学生证丢了去哪补办？", INTENT_CONSULT),
    ]

    wrong = [(q, e, classifier.classify(q)) for q, e in cases if classifier.classify(q) != e]
    assert not wrong, f"误判: {wrong}"
    return f"backend={classifier.backend}，{len(cases)}/{len(cases)} 正确"


def test_mysql_data():
    from src.mysql_qa import MySQLManager

    pairs = MySQLManager().get_all_qa_pairs()
    assert len(pairs) >= 100, f"高频问答只有 {len(pairs)} 条"
    assert all(p.get("id") is not None for p in pairs), "缺少主键 id"
    return f"{len(pairs)} 条问答对"


def test_conversation_store():
    """会话列表和消息正文必须真正写入 MySQL，并按登录用户隔离。"""
    import uuid

    from src.auth import AuthManager
    from src.mysql_qa import ConversationStore

    auth = AuthManager()
    auth.ensure_tables()
    store = ConversationStore()
    store.ensure_tables()
    suffix = uuid.uuid4().hex[:12]
    user = auth.register(f"test_{suffix}", "TestPassword123!")
    other_user = auth.register(f"other_{suffix}", "TestPassword123!")
    conversation = store.create_conversation(user["id"])
    conversation_id = conversation["id"]

    try:
        assert store.get_conversation(other_user["id"], conversation_id) is None, \
            "其他用户不应看到该会话"
        updated = store.append_exchange(
            user["id"], conversation_id,
            "转专业需要什么条件？", "请按学院要求提交申请。"
        )
        assert updated and updated["message_count"] == 2, updated
        assert updated["title"].startswith("转专业"), updated

        messages = store.get_messages(user["id"], conversation_id)
        assert messages and [item["role"] for item in messages] == ["user", "assistant"]
        assert messages[0]["content"] == "转专业需要什么条件？"
        assert store.list_conversations(user["id"])[0]["id"] == conversation_id
        return f"会话 {conversation_id} 的 2 条消息写入、读取和用户隔离正常"
    finally:
        auth.delete_user(user["id"])
        auth.delete_user(other_user["id"])


def test_auth_manager():
    import uuid

    from src.auth import AuthManager

    auth = AuthManager()
    auth.ensure_tables()
    username = f"auth_{uuid.uuid4().hex[:12]}"
    password = "TestPassword123!"
    user = auth.register(username, password)
    try:
        assert auth.authenticate(username, password)["id"] == user["id"]
        assert auth.authenticate(username, "wrong-password") is None
        token = auth.create_session(user["id"])
        assert auth.get_user_by_token(token)["id"] == user["id"]
        auth.delete_session(token)
        assert auth.get_user_by_token(token) is None
        return "注册、密码哈希校验、登录会话和退出正常"
    finally:
        auth.delete_user(user["id"])


def test_password_reset():
    import uuid

    from src.auth import AuthManager

    auth = AuthManager()
    auth.ensure_tables()
    suffix = uuid.uuid4().hex[:12]
    username = f"reset_{suffix}"
    old_password = "OldPassword123!"
    new_password = "NewPassword456!"
    user = auth.register(username, old_password)
    other_user = None
    try:
        other_user = auth.register(f"other_reset_{suffix}", old_password)
        tokens = [auth.create_session(user["id"]) for _ in range(2)]
        other_token = auth.create_session(other_user["id"])

        try:
            auth.reset_password(username, "short")
        except ValueError:
            pass
        else:
            raise AssertionError("新密码不足 8 位时应拒绝重置")
        assert auth.authenticate(username, old_password)["id"] == user["id"]
        assert all(auth.get_user_by_token(token) for token in tokens)

        assert auth.reset_password(username, new_password)
        assert auth.authenticate(username, old_password) is None
        assert auth.authenticate(username, new_password)["id"] == user["id"]
        assert all(auth.get_user_by_token(token) is None for token in tokens)
        assert auth.authenticate(other_user["username"], old_password)["id"] == other_user["id"]
        assert auth.get_user_by_token(other_token)["id"] == other_user["id"]
        assert not auth.reset_password(f"missing_{suffix}", new_password)
        assert auth.authenticate(f"missing_{suffix}", new_password) is None
        return "存在性检查、新旧密码校验、旧会话失效及其他账户隔离正常"
    finally:
        auth.delete_user(user["id"])
        if other_user:
            auth.delete_user(other_user["id"])


def test_qa_self_match():
    """验收标准：CSV 原问题必须 100% 命中自己。"""
    from src.mysql_qa import BM25Retriever, MySQLManager

    pairs = MySQLManager().get_all_qa_pairs()
    retriever = BM25Retriever()
    retriever.load_questions([p["question"] for p in pairs], ids=[p["id"] for p in pairs])

    failures = []
    for pair in pairs:
        top = retriever.search_topk(pair["question"], top_k=1)
        if not top or top[0]["question"] != pair["question"]:
            failures.append((pair["question"], top[0]["question"] if top else None))

    # 分词完全相同的问句（如「缓考怎么申请？」/「可以申请缓考吗？」）BM25 会打平，
    # 靠字符 n-gram tiebreak 区分，这里要求严格命中自己
    assert not failures, f"自匹配失败 {len(failures)} 条: {failures[:5]}"
    return f"{len(pairs)} 条全部命中自己"


def test_qa_thresholds():
    """改写问法要召回，跨话题问法要放行——两边都不能塌。"""
    from config import SYSTEM_CONFIG
    from src.mysql_qa import BM25Retriever, MySQLManager

    pairs = MySQLManager().get_all_qa_pairs()
    retriever = BM25Retriever()
    retriever.load_questions([p["question"] for p in pairs], ids=[p["id"] for p in pairs])

    score_th = SYSTEM_CONFIG["bm25_threshold"]
    overlap_th = SYSTEM_CONFIG["keyword_overlap_threshold"]

    def accepted(query):
        top = retriever.search_topk(query, top_k=1)
        if not top:
            return None
        top = top[0]
        return top if (top["score"] >= score_th and top["keyword_overlap"] >= overlap_th) else None

    # 应当命中的改写问法（归一化 / 同义词 生效）
    should_hit = {
        "学费要交多少啊？": "学费要交多少啊？",
        "如何申请奖学金": "怎么申请奖学金？",
        "奖学金咋申请": "怎么申请奖学金？",
        "学费多少钱": "学费要交多少啊？",
        "我弄丢了学生证怎么办": "学生证丢了怎么办？",
        "宿舍门禁是几点": "宿舍门禁时间是几点？",
    }
    for query, expected in should_hit.items():
        top = accepted(query)
        assert top is not None, f"'{query}' 应当命中高频库，实际被放行到 RAG"
        assert top["question"] == expected, f"'{query}' 命中错误: {top['question']} != {expected}"

    # 应当放行的跨话题问法：库里没有对应问题，不能硬答
    should_miss = [
        "宿舍几点熄灯",          # 区分词"熄灯"库里不存在，最接近的是"宿舍什么时候开门"
        "转专业需要什么条件",
        "图书馆借书期限",
        "今天天气怎么样",
    ]
    for query in should_miss:
        top = accepted(query)
        assert top is None, (
            f"'{query}' 不该命中高频库，实际命中 '{top['question']}' "
            f"(score={top['score']:.2f}, ov={top['keyword_overlap']:.2f})"
        )

    return f"命中 {len(should_hit)}/{len(should_hit)}，正确放行 {len(should_miss)}/{len(should_miss)}"


def test_milvus_schema_and_dense():
    """回归：schema 与代码不一致会让 dense 检索静默失效。"""
    from pymilvus import Collection, connections, utility

    from config import MILVUS_CONFIG
    from src.knowledge_base.vector_store import VectorStoreManager

    store = VectorStoreManager()
    store._connect()

    if not utility.has_collection(MILVUS_CONFIG["collection_name"], using=store.conn_alias):
        raise AssertionError(
            f"集合不存在，请先运行: python scripts/build_knowledge_base.py"
        )

    collection = Collection(MILVUS_CONFIG["collection_name"], using=store.conn_alias)
    fields = {f.name: f for f in collection.schema.fields}

    required = {"child_text", "parent_text", "document_name",
                "parent_index", "child_index", "vector"}
    missing = required - set(fields)
    assert not missing, f"缺少字段 {sorted(missing)}（现有 {sorted(fields)}）"

    store.load_embedding_model()
    dim = fields["vector"].params.get("dim")
    assert int(dim) == int(store.dimension), \
        f"维度不匹配：集合 {dim} vs 模型 {store.dimension}，dense 检索必然失败"

    assert collection.num_entities > 0, "集合为空"

    # 真正跑一次向量检索
    results = store.search("奖学金怎么申请", top_k=3)
    assert results, "dense 检索没有返回任何结果"
    assert results[0].get("child_text"), "检索结果缺少 child_text"
    assert results[0].get("parent_text"), "检索结果缺少 parent_text"

    return (f"{collection.num_entities} 条，dim={dim}，"
            f"top1 相似度={results[0]['dense_score']:.3f}")


def test_hybrid_retrieval():
    from src.rag_qa.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()
    retriever._ensure_initialized()
    assert retriever.dense_available, "dense 路不可用（embedding 模型未加载）"

    results = retriever.retrieve(["奖学金怎么申请", "奖学金申请条件"], top_k=5)
    assert results, "混合检索无结果"

    for doc in results:
        assert doc.get("child_text"), "缺少 child_text"
        assert doc.get("parent_text"), "缺少 parent_text（父块必须被带回来）"
        assert doc.get("rrf_score", 0) > 0, "缺少 RRF 分数"

    parents = [(d["document_name"], d["parent_index"]) for d in results]
    assert len(parents) == len(set(parents)), f"父块未去重: {parents}"

    return f"{len(results)} 条，父块去重正常，top1 rrf={results[0]['rrf_score']:.5f}"


def test_query_rewriter():
    from src.rag_qa.query_rewriter import QueryRewriter

    rewriter = QueryRewriter()

    result = rewriter.rewrite("怎么申请奖学金？")
    assert result["queries"][0] == "怎么申请奖学金？", "原问题必须排在第一位"
    assert len(result["queries"]) <= 2, f"query 数量应受控，实际 {len(result['queries'])}"

    multi = rewriter.rewrite("转专业、休学都是什么流程？")
    assert multi["strategy"] == "sub_query", multi

    conditional = rewriter.rewrite("宿舍能不能养宠物？")
    assert conditional["strategy"] == "backtrack", conditional

    long_count = len(rewriter.rewrite("关于学校各项规章制度的具体要求有哪些")["queries"])
    assert long_count <= 2

    history = [
        {"role": "user", "content": "转专业怎么申请？"},
        {"role": "assistant", "content": "需要按学院要求提交申请。"},
    ]
    contextual = rewriter.contextualize("那需要什么材料？", history)
    assert "转专业怎么申请" in contextual, contextual
    assert "那需要什么材料" in contextual, contextual
    assert rewriter.is_follow_up("那需要什么材料？", history)
    assert not rewriter.is_follow_up("奖学金怎么申请？", history), "独立新问题不应继承旧主题"

    return "策略选择、query 数量与多轮追问补全正常"


def test_rrf_fusion():
    from src.rag_qa.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()

    def doc(name, parent, child, source):
        return {"document_name": name, "parent_index": parent, "child_index": child,
                "child_text": f"text{child}", "parent_text": "p", "source": source}

    dense = [doc("a", 0, 0, "dense"), doc("a", 0, 1, "dense")]
    sparse = [doc("a", 0, 1, "sparse"), doc("a", 0, 2, "sparse")]

    fused = retriever._rrf_fuse([dense, sparse])
    top = fused[0]
    assert top["child_index"] == 1, "两路都命中的文档应当排第一"
    assert top["source"] == "hybrid", top

    deduped = retriever._deduplicate_by_parent(fused)
    assert len({(d["document_name"], d["parent_index"]) for d in deduped}) == len(deduped)
    assert len(deduped) == 1, "同一父块只应保留一条"
    return "两路命中优先 + 父子去重正常"


def test_redis_cache():
    from src.mysql_qa import RedisCache

    cache = RedisCache()
    stats = cache.get_cache_stats()
    assert stats.get("indexed_questions_count", 0) > 0, \
        f"Redis 问题索引为空，请运行 scripts/init_redis.py（{stats}）"

    entry = cache.get_qa_index_entry("怎么申请奖学金")
    assert entry and entry.get("id") is not None, f"索引查询失败: {entry}"
    return f"索引 {stats['indexed_questions_count']} 条，命中 id={entry['id']}"


# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("RAG 系统回归测试")
    print("=" * 70)

    check("配置加载", test_config)
    check("递归切割器终止性（回归）", test_chunker_terminates)
    check("父子分块结构", test_chunker_output_shape)
    check("文档读取与预处理", test_document_loader)
    check("query 归一化", test_normalization)
    check("意图特征训练/推理一致（回归）", test_intent_features_shared)
    check("意图识别", test_intent_classifier)
    check("MySQL 高频问答", test_mysql_data)
    check("用户注册与登录", test_auth_manager)
    check("忘记密码与会话失效", test_password_reset)
    check("MySQL 对话记录", test_conversation_store)
    check("Redis 问题索引", test_redis_cache)
    check("高频问答自匹配 100%（回归）", test_qa_self_match)
    check("高频问答阈值判定", test_qa_thresholds)
    check("Milvus schema 与 dense 检索（回归）", test_milvus_schema_and_dense)
    check("混合检索双路召回", test_hybrid_retrieval)
    check("RRF 融合与父块去重", test_rrf_fusion)
    check("检索策略选择与改写", test_query_rewriter)

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, detail in RESULTS:
        print(f"  {'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail and not ok else ""))
    print(f"\n  通过 {passed}/{len(RESULTS)}")
    print("=" * 70)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
