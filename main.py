"""RAG 智能问答系统 - FastAPI 后端服务。

端点一律使用同步 def：链路里有 torch 推理、MySQL/Redis/Milvus 访问、
HTTP 调用 LLM，这些都是阻塞操作。写成 async def 会独占事件循环，
并发时全部串行；写成 def 时 FastAPI 会自动丢到线程池执行。
"""
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.common.startup import StartupIssue

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("rag")

unified_system = None
_init_lock = threading.Lock()
conversation_store = None
_conversation_store_lock = threading.Lock()
auth_manager = None
_auth_manager_lock = threading.Lock()


def get_system():
    """线程安全的懒加载，避免并发首请求各初始化一份模型。"""
    global unified_system
    if unified_system is None:
        with _init_lock:
            if unified_system is None:
                logger.info("初始化系统组件...")
                from src.unified_query_system import UnifiedQuerySystem
                unified_system = UnifiedQuerySystem()
                logger.info("系统初始化完成")
    return unified_system


def get_conversation_store():
    """线程安全地初始化 MySQL 会话存储，并确保数据表存在。"""
    global conversation_store
    if conversation_store is None:
        with _conversation_store_lock:
            if conversation_store is None:
                from src.mysql_qa import ConversationStore
                store = ConversationStore()
                store.ensure_tables()
                conversation_store = store
    return conversation_store


def get_auth_manager():
    """线程安全地初始化 MySQL 用户认证存储。"""
    global auth_manager
    if auth_manager is None:
        with _auth_manager_lock:
            if auth_manager is None:
                from src.auth import AuthManager
                manager = AuthManager()
                manager.ensure_tables()
                auth_manager = manager
    return auth_manager


def require_user(request: Request) -> Dict[str, Any]:
    """读取登录 Cookie；未登录或会话过期时返回 401。"""
    token = request.cookies.get("rag_session")
    user = get_auth_manager().get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    return user


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        "rag_session",
        token,
        max_age=7 * 24 * 60 * 60,
        httponly=True,
        samesite="lax",
        path="/",
    )


# 预热状态。模型加载要十几秒，前端和运维都需要知道"现在到哪一步了"。
_warmup: Dict[str, Any] = {"status": "pending", "elapsed_ms": None,
                           "timings": None, "error": None, "problem": None}
_warmup_lock = threading.Lock()


def run_warmup() -> Dict[str, Any]:
    """加载全部模型。已预热过就直接返回，不会重复加载。

    整个过程串行化在一把锁里：预热线程和用户点的"预热"按钮撞上时，
    后到者等前一个跑完拿到结果，而不是各加载一份模型。
    """
    with _warmup_lock:
        if _warmup["status"] == "done":
            return dict(_warmup)

        _warmup["status"] = "running"
        _warmup["error"] = None
        _warmup["problem"] = None
        started = time.time()
        try:
            try:
                get_auth_manager()
                get_conversation_store()
            except Exception as error:
                raise StartupIssue("mysql_unavailable", "mysql", "MySQL 初始化失败",
                                   "检查 MySQL 容器状态、连接配置和数据库权限。") from error
            system = get_system()
            result = system.warmup()
            _warmup["timings"] = result["timings"]
            _warmup["status"] = "done"
            logger.info("预热完成: %s", result["timings"])
        except Exception as e:
            _warmup["status"] = "failed"
            issue = e if isinstance(e, StartupIssue) else StartupIssue(
                "startup_failed", "system", "系统初始化失败",
                "请查看应用容器日志中的预热异常。")
            _warmup["problem"] = issue.as_dict()
            _warmup["error"] = issue.message
            logger.exception("预热失败")
        finally:
            _warmup["elapsed_ms"] = round((time.time() - started) * 1000, 1)
        return dict(_warmup)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 数据库检查纳入预热状态。依赖未就绪时仍能展示主页和准确的故障原因。
    # 后台预热：不阻塞 uvicorn 启动，也不占用事件循环。
    # 用线程而不是 await，是因为加载模型是纯阻塞操作，
    # 放事件循环里会把启动和所有请求一起卡住。
    threading.Thread(target=run_warmup, name="preheat", daemon=True).start()
    yield


app = FastAPI(title="RAG 智能问答系统", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[int] = None
    document: Optional[str] = None
    top_k: int = 5
    chat_history: Optional[List[ChatMessage]] = None
    chat_count: int = 0


class DocumentRequest(BaseModel):
    document_name: str


class ConversationCreate(BaseModel):
    title: str = "新对话"


class AuthCredentials(BaseModel):
    username: str
    password: str


class PasswordResetRequest(BaseModel):
    username: str
    new_password: str


@app.get("/api/health")
def health_check():
    """组件状态 + 模型状态。

    system_ready 只说明组件对象建好了（毫秒级），models_ready 才代表
    模型全部加载完、第一个问题不用再等——前端看的是后者。
    """
    models = unified_system.get_model_status() if unified_system else {}
    models_ready = _warmup["status"] == "done" and bool(unified_system) and all([
        models.get("embedding"),
        models.get("intent"),      # 降级成 rules 也算就绪，那是终态
        models.get("reranker"),
    ])
    return {
        "status": "ok" if models_ready else ("degraded" if _warmup["status"] == "failed" else "starting"),
        "system_ready": unified_system is not None,
        "models_ready": models_ready,
        "models": models,
        "warmup": _warmup["status"],
        "error": _warmup["error"],
        "problem": _warmup["problem"],
    }


@app.post("/api/warmup")
def warmup():
    """显式预热：加载 bge-m3 + BM25 索引 + 意图 BERT + 精排模型，并各跑一次前向。"""
    state = run_warmup()
    if state["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"预热失败: {state['error']}")
    return {
        "success": True,
        "message": "预热完成",
        "elapsed_ms": state["elapsed_ms"],
        "timings": state["timings"],
    }


@app.get("/api/documents")
def get_documents():
    from config import DATA_DIR

    documents = []
    if os.path.exists(DATA_DIR):
        for filename in sorted(os.listdir(DATA_DIR)):
            if filename.lower().endswith(".pdf"):
                path = os.path.join(DATA_DIR, filename)
                documents.append({"name": filename, "path": path,
                                  "size_mb": round(os.path.getsize(path) / 1024 / 1024, 2)})
    return {"documents": documents}


@app.post("/api/auth/register")
def register(payload: AuthCredentials, response: Response):
    try:
        user = get_auth_manager().register(payload.username, payload.password)
    except ValueError as exc:
        status_code = 409 if str(exc) == "用户名已存在" else 400
        raise HTTPException(status_code=status_code, detail=str(exc))
    token = get_auth_manager().create_session(user["id"])
    set_session_cookie(response, token)
    return {"user": user}


@app.post("/api/auth/login")
def login(payload: AuthCredentials, response: Response):
    user = get_auth_manager().authenticate(payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = get_auth_manager().create_session(user["id"])
    set_session_cookie(response, token)
    return {"user": user}


@app.post("/api/auth/reset-password")
def reset_password(payload: PasswordResetRequest):
    try:
        updated = get_auth_manager().reset_password(payload.username, payload.new_password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=404, detail="账户不存在，请检查用户名")
    return {"success": True, "message": "密码已重置，请使用新密码登录"}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    get_auth_manager().delete_session(request.cookies.get("rag_session"))
    response.delete_cookie("rag_session", path="/")
    return {"success": True}


@app.get("/api/auth/me")
def current_user(request: Request):
    return {"user": require_user(request)}


@app.get("/api/conversations")
def list_conversations(request: Request):
    user = require_user(request)
    items = get_conversation_store().list_conversations(user["id"])
    return {"conversations": items}


@app.post("/api/conversations")
def create_conversation(request: Request, payload: ConversationCreate):
    user = require_user(request)
    conversation = get_conversation_store().create_conversation(
        user["id"], payload.title
    )
    return {"conversation": conversation}


@app.get("/api/conversations/{conversation_id}/messages")
def get_conversation_messages(conversation_id: int, request: Request):
    user = require_user(request)
    messages = get_conversation_store().get_messages(
        user["id"], conversation_id
    )
    if messages is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"conversation_id": conversation_id, "messages": messages}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: int, request: Request):
    user = require_user(request)
    deleted = get_conversation_store().delete_conversation(
        user["id"], conversation_id
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"success": True}


@app.post("/api/query")
def query_system(payload: QueryRequest, request: Request):
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="查询不能为空")

    try:
        user = require_user(request)
        if _warmup["status"] == "failed":
            problem = _warmup["problem"] or {}
            raise HTTPException(status_code=503, detail=" ".join(filter(None, [
                problem.get("message", "系统尚未就绪"), problem.get("hint")
            ])))
        system = get_system()
        chat_history = None
        chat_count = payload.chat_count
        store = get_conversation_store()

        if payload.conversation_id is not None:
            stored_messages = store.get_messages(
                user["id"], payload.conversation_id, limit=12
            )
            if stored_messages is None:
                raise HTTPException(status_code=404, detail="对话不存在")
            chat_history = [
                {"role": message["role"], "content": message["content"]}
                for message in stored_messages
            ]
            chat_count = sum(1 for message in chat_history if message["role"] == "user")
        elif payload.chat_history:
            # 保留旧 API 调用方式；网页端只提交 conversation_id。
            chat_history = [
                {"role": message.role, "content": message.content.strip()}
                for message in payload.chat_history[-12:]
                if message.content.strip()
            ]

        result = system.query(
            payload.query.strip(),
            top_k=payload.top_k,
            document=payload.document,
            chat_history=chat_history,
            chat_count=chat_count,
        )

        if payload.conversation_id is not None:
            conversation = store.append_exchange(
                user["id"],
                payload.conversation_id,
                payload.query.strip(),
                result["answer"],
            )
            if conversation is None:
                raise HTTPException(status_code=404, detail="对话不存在")
            result["conversation_id"] = payload.conversation_id
            result["conversation"] = conversation

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("查询失败")
        raise HTTPException(status_code=500, detail=f"查询失败: {e}")


@app.post("/api/process-document")
def process_document(request: DocumentRequest):
    if not request.document_name:
        raise HTTPException(status_code=400, detail="文档名称不能为空")

    try:
        result = get_system().add_document(request.document_name)
        return {"success": True, "message": "文档处理完成", "result": result}
    except Exception as e:
        logger.exception("文档处理失败")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
def get_stats():
    return get_system().get_stats()


@app.post("/api/refresh-cache")
def refresh_cache():
    get_system().refresh_cache()
    return {"success": True, "message": "缓存已刷新"}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(BASE_DIR, "templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(
            content=f.read(),
            headers={"Cache-Control": "no-store"},
        )


app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


if __name__ == "__main__":
    import uvicorn

    # 必须同时监听 IPv4 与 IPv6。
    # 只绑 0.0.0.0 时，客户端把 localhost 解析到 ::1，每次请求都要先等
    # IPv6 连接超时再回退到 IPv4（实测 Python requests 2063ms vs 4ms，
    # curl 约 213ms），前端用 localhost 打开也会中招。
    # 注意不能只绑 "::"：Windows 下 Python 默认 IPV6_V6ONLY=1，
    # 那样反而完全不接受 IPv4 连接（实测 127.0.0.1 连接超时）。
    HOSTS = ["0.0.0.0", "::"]

    print("=" * 60)
    print("RAG 智能问答系统 - FastAPI 服务")
    print("  服务地址: http://localhost:5000")
    print("  API文档:  http://localhost:5000/docs")
    print("=" * 60)

    uvicorn.run(app, host=HOSTS, port=5000, log_level="info")
