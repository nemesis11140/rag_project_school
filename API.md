# RAG 智能问答系统 - API 接口文档

## 基础信息

| 项目 | 内容 |
|------|------|
| 服务地址 | http://localhost:5000 |
| API 前缀 | /api |
| 数据格式 | JSON |
| 认证方式 | `HttpOnly` 登录 Cookie |
| 交互式文档 | http://localhost:5000/docs |

---

## 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/health | 健康检查 |
| POST | /api/warmup | 预热模型与索引 |
| GET | /api/documents | 列出 data/ 下的 PDF |
| POST | /api/auth/register | 注册账号并登录 |
| POST | /api/auth/login | 登录 |
| POST | /api/auth/reset-password | 按用户名重置密码（演示流程） |
| POST | /api/auth/reset-password | 按用户名重置密码（演示流程） |
| POST | /api/auth/logout | 退出登录 |
| GET | /api/auth/me | 获取当前用户 |
| GET / POST | /api/conversations | 查询 / 新建对话 |
| GET | /api/conversations/{id}/messages | 查询对话消息 |
| DELETE | /api/conversations/{id} | 删除对话 |
| POST | /api/query | 问答主接口 |
| POST | /api/process-document | 单文档增量入库 |
| GET | /api/stats | 系统统计 |
| POST | /api/refresh-cache | 重建 Redis 缓存 |

---

## 用户认证

注册和登录请求：

```json
{
  "username": "student_01",
  "password": "至少八个字符"
}
```

密码使用 PBKDF2-SHA256 加随机盐后保存，数据库不保存明文密码。登录成功后服务端
通过 `HttpOnly` Cookie 维持 7 天登录状态。会话表使用 `user_id` 关联账号，用户只能
查询、写入或删除自己的会话和消息。

未登录或登录状态过期时，受保护接口返回 `401`。

### 忘记密码

登录窗口提供“忘记密码？”入口，输入已注册的用户名、新密码和确认新密码即可提交。前端检查两次密码一致，后端检查账户存在性及新密码长度（8～128 个字符），不要求原密码。

```text
POST /api/auth/reset-password
```

```json
{
  "username": "student_01",
  "new_password": "NewPassword123!"
}
```

成功响应为 `{"success": true, "message": "密码已重置，请使用新密码登录"}`。用户名格式或密码长度不合法返回 `400`，账户不存在返回 `404`。密码更新与该账户所有旧登录会话的失效在同一事务中完成；历史对话保留，重置成功后需要用新密码重新登录。

此接口按演示需求仅检查账户存在性，不验证账户归属；知道用户名的人也能重置密码。正式对外部署前应增加验证码或其他身份验证。

### 忘记密码

登录窗口提供“忘记密码？”入口，输入已注册的用户名、新密码和确认新密码即可提交。前端检查两次密码一致，后端检查账户存在性及新密码长度（8～128 个字符），不要求原密码。

```text
POST /api/auth/reset-password
```

```json
{
  "username": "student_01",
  "new_password": "NewPassword123!"
}
```

成功响应为 `{"success": true, "message": "密码已重置，请使用新密码登录"}`。用户名格式或密码长度不合法返回 `400`，账户不存在返回 `404`。密码更新与该账户所有旧登录会话的失效在同一事务中完成；历史对话保留，重置成功后需要用新密码重新登录。

此接口按演示需求仅检查账户存在性，不验证账户归属；知道用户名的人也能重置密码。正式对外部署前应增加验证码或其他身份验证。

---

### 1. 健康检查

```
GET /api/health
```

```json
{
  "status": "ok",
  "system_ready": true,
  "models_ready": true,
  "models": { "embedding": true, "intent": "bert", "reranker": true },
  "warmup": "done",
  "error": null,
  "problem": null
}
```

| 字段 | 说明 |
|------|------|
| `status` | `starting`（检查或预热中）/ `ok`（就绪）/ `degraded`（初始化失败） |
| `system_ready` | 组件对象是否建好（毫秒级）。**不代表模型已加载** |
| `models_ready` | 三个模型是否全部就绪。前端看这个字段 |
| `models.embedding` | bge-m3 是否已加载 |
| `models.intent` | 意图识别用的后端：`bert` / `sklearn` / `rules`（降级兜底也算就绪）/ `null`（未加载） |
| `models.reranker` | 精排模型是否已加载 |
| `warmup` | 预热状态：`pending` / `running` / `done` / `failed` |
| `problem` | 失败时包含 `code`、`component`、`message`、`hint`，区分知识库缺失、数据库连接与模型文件问题；正常时为 `null` |

服务启动时会在后台自动预热，所以 `models_ready` 通常在前 20~35s 内从
`false` 翻到 `true`，具体耗时取决于服务器。只有知识库非空且预热成功才会就绪；启动失败时查询返回 `503` 和可操作的提示。修复数据库或挂载后可再次预热，不再把 Milvus 数据库缺失统称为模型加载失败。

---

### 2. 预热

```
POST /api/warmup
```

加载链路上全部懒加载的模型并各跑一次前向：bge-m3 + BM25 索引、意图 BERT、
精排模型。幂等——已预热过会立即返回（`elapsed_ms` 仍是首次的耗时）。
服务启动和前端页面打开时都会自动调用，一般不需要手动调。

```json
{
  "success": true,
  "message": "预热完成",
  "elapsed_ms": 21350.4,
  "timings": {
    "retrieval_ms": 8346.0,
    "intent_ms": 57.4,
    "rerank_ms": 4950.1,
    "total_ms": 13353.5
  }
}
```

模型加载失败时返回 500，detail 为失败原因（如 Milvus 未启动）。

---

### 3. 文档列表

```
GET /api/documents
```

```json
{
  "documents": [
    { "name": "2025学生手册.pdf", "path": "...", "size_mb": 6.34 }
  ]
}
```

---

### 4. 问答接口

```
POST /api/query
```

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| query | string | 是 | - | 用户问题 |
| conversation_id | integer | 否 | null | MySQL 会话 ID；网页端始终传入 |
| document | string | 否 | null | 限定在某个文档内检索 |
| top_k | int | 否 | 5 | 返回的参考文档数 |
| chat_history | array | 否 | null | `[{"role":"user","content":"..."}]`，最多取最近 12 条（约 6 轮） |
| chat_count | int | 否 | 0 | 已发生的对话轮数 |

传入 `conversation_id` 后，服务端会直接从 MySQL 读取最近 12 条消息作为上下文，
并在回答成功后把本轮用户问题和助手回答写回 MySQL。`chat_history` 仅用于兼容旧的
API 调用方式，网页端不再提交或保存消息正文。

网页端根据登录账号加载数据。账号保存在 `users` 表，登录会话保存在
`user_sessions` 表；会话列表、标题和消息正文保存在 `chat_conversations`、
`chat_messages` 表中。

**响应**

```json
{
  "query": "奖学金怎么申请？",
  "source": "mysql",
  "answer": "……",
  "question": "怎么申请奖学金？",
  "score": 7.767,
  "keyword_overlap": 1.0,
  "method": "bm25_retrieval"
}
```

RAG 链路的响应：

```json
{
  "query": "转专业的流程是什么？",
  "source": "rag",
  "intent": "专业咨询",
  "answer": "……",
  "strategy": "direct",
  "rewritten_queries": ["转专业的流程是什么？"],
  "retrieved_docs": [ { "child_text": "...", "parent_text": "...", "document_name": "2025学生手册.pdf" } ],
  "timings": {
    "intent_ms": 12.3, "rewrite_ms": 0.1, "retrieval_ms": 210.5,
    "rerank_ms": 480.2, "generate_ms": 1520.8, "total_ms": 2224.0
  },
  "method": "rag_pipeline"
}
```

**`source` 取值**

| 值 | 含义 |
|----|------|
| `redis` | 命中已验证的原始问题映射，答案仍从 MySQL 获取 |
| `mysql` | 原文精确匹配，或经问题语义等价校验通过，答案取自 MySQL |
| `rag` | 走向量检索 + 精排 + LLM 生成 |
| `chitchat_fallback` | 闲聊回答，或直接介绍助手的身份与能力 |

请求先识别意图，只有校园咨询参与 Redis 和 MySQL 高频问答匹配。“你有什么用”“你能做什么”“你是谁”等明确询问助手的问题直接返回能力介绍，不检索校园知识，也不调用大模型。

高频问答的 `method` 为 `exact_match`（只做排版清洗的原文匹配）、`semantic_verified`（语义等价校验通过）或 `semantic_cache_hit`（复用已校验的问题映射）。BM25 分数和关键词覆盖率只筛选候选，不能直接决定返回答案。语义比较保留用户原话，检查对象、事项、否定、条件及问题范围；服务异常、超时或不确定时转 RAG。旧的 `qa_answer:*` 答案缓存与归一化索引不再用于直接返回答案。

校验复用现有 LLM 配置。`[system]` 中 `faq_candidate_count` 默认 5，`faq_semantic_timeout` 默认 15 秒，`faq_semantic_min_confidence` 默认 0.9（模型判断的门槛，不是统计准确率）。问答库、模型或校验规则变化会使旧问题映射失效。

离线回归：`python -m unittest discover -s tests -p 'test_faq_semantics.py'`。真实模型验证：`python scripts/evaluate_faq_semantics.py --live`，使用 20 组同义、跨对象、否定、条件与多问题案例，不访问数据库，结果保存到 `runtime/faq-semantics-evaluation.json`。

**`strategy` 取值**：`direct`（原问题）/ `hypothetical`（假设问题）/
`sub_query`（子问题）/ `backtrack`（回溯问题）。

---

### 5. 单文档入库

```
POST /api/process-document
{ "document_name": "2025学生手册.pdf" }
```

**注意**：这是**增量**写入，不会清空集合。整体重建请用
`python scripts/build_knowledge_base.py`。

---

### 6. 系统统计

```
GET /api/stats
```

```json
{
  "mysql_qa": {
    "bm25_threshold": 3.0,
    "keyword_overlap_threshold": 0.6,
    "indexed_questions": 200,
    "indexed_questions_count": 200,
    "cached_answers_count": 18
  },
  "knowledge_base": { "row_count": 676, "collection": "rag_knowledge_base" }
}
```

---

### 7. 刷新缓存

```
POST /api/refresh-cache
```

清空答案缓存与问题索引，重新从 MySQL 加载并重建 Redis 索引。
**更新 high_frequency_qa 表后调用此接口**（或重跑 `scripts/init_redis.py`）。

---

### 错误响应

| 状态码 | 场景 |
|--------|------|
| 400 | query / document_name 为空 |
| 500 | 初始化失败、查询或文档处理异常 |

```json
{ "detail": "查询失败: ..." }
```
