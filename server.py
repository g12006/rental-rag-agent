"""
FastAPI 服务

来源：ragbasic-master/medical/backend/api.py
    FastAPI + CORSMiddleware + Pydantic 请求体 + HTTPException + uvicorn
在 ragbasic 的两个接口之外，补充了人工审核恢复接口。
"""
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

import config
from agent.graph import graph, new_config, snapshot
from ragbase import store
from ragbase.chunkers import STRATEGIES, build_chunks

app = FastAPI(title="租房平台智能客服 Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载前端页面（web/index.html）
WEB_DIR = Path(__file__).parent / "web"
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

# 建索引时加锁，避免并发重建（medical/backend/main.py 的 _db_lock 思路）
_db_lock = threading.Lock()


class AskRequest(BaseModel):
    question: str
    thread_id: str = "web"
    user_id: str = "anonymous"


class ResumeRequest(BaseModel):
    thread_id: str = "web"
    user_id: str = "anonymous"
    approved: bool = True
    content: str = ""


class RebuildRequest(BaseModel):
    strategy: str = config.CHUNK_STRATEGY


@app.post("/api/rebuild-index")
def rebuild_index(payload: RebuildRequest) -> dict:
    if payload.strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"未知分块策略：{payload.strategy}")
    try:
        with _db_lock:
            chunks = build_chunks(payload.strategy)
            count = store.build_index(chunks)
        return {"ok": True, "message": f"索引构建完成，写入 {count} 条数据", "count": count}
    except Exception as e:
        print(f"构建索引失败：{e}")
        raise HTTPException(status_code=500, detail=f"索引构建失败：{e}")


@app.post("/api/ask")
def ask_question(payload: AskRequest) -> dict:
    q = payload.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="问题不能为空")

    cfg = new_config(payload.thread_id, payload.user_id)
    try:
        from langchain_core.messages import HumanMessage
        result = graph.invoke({"messages": [HumanMessage(content=q)]}, config=cfg)
    except Exception as e:
        print(f"查询失败：{e}")
        raise HTTPException(status_code=500, detail=f"查询失败：{e}")

    # 命中人工审核中断
    if result.get("__interrupt__"):
        return {
            "answer": "",
            "interrupted": True,
            "review": result["__interrupt__"][0].value,
            "thread_id": payload.thread_id,
        }

    data = snapshot(result)
    data["thread_id"] = payload.thread_id
    return data


@app.post("/api/resume")
def resume(payload: ResumeRequest) -> dict:
    """人工审核后恢复执行（LangGraph interrupt + Command(resume=...)）"""
    from langgraph.types import Command

    cfg = new_config(payload.thread_id, payload.user_id)
    result = graph.invoke(Command(resume={
        "approved": "true" if payload.approved else "false",
        "content": payload.content,
    }), config=cfg)

    data = snapshot(result)
    data["thread_id"] = payload.thread_id
    return data


@app.get("/api/stats")
def stats() -> dict:
    return {
        "document_path": str(config.DOCUMENT_PATH),
        "source_filter": config.DOC_SOURCE_FILTER,
        "chunk_strategy": config.CHUNK_STRATEGY,
        "vector_backend": config.VECTOR_BACKEND,
        "chat_model": config.CHAT_MODEL,
        "human_review": config.HUMAN_REVIEW,
    }


if __name__ == "__main__":
    uvicorn.run(app="server:app", host=config.API_HOST, port=config.API_PORT, reload=False)
