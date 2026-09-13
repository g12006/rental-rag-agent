"""
嵌入模型 + 向量库

来源：
    ragbasic-master/milvus.py        Milvus Lite 的建库 / 建集合 / 插入 / 检索
    ragbasic-master/similarity.py    HuggingFaceEmbedding + similarity + SimilarityMode
    LangChain/test30.py、test31.py   OpenAIEmbeddings + InMemoryVectorStore

两套"纯血"组合，互不混搭，通过 config.VECTOR_BACKEND 切换：
    milvus : BGE-M3(1024 维)  + Milvus Lite       —— 对应 ragbasic
    memory : OpenAIEmbeddings + InMemoryVectorStore —— 对应 LangChain 课堂代码

统一对外返回 [{"id", "text", "score", "metadata"}]，score 为余弦相似度（越大越相似）。
"""
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

# 嵌入模型单例（BGE-M3 加载很慢，只加载一次）
# 注意：LangGraph 会在多线程里跑节点，单例必须加锁，否则并发下会重复加载（实测加载了两次，多花 40 秒 + 多占一份显存/内存）
_embed_model = None
_embed_lock = threading.Lock()


def get_embed_model():
    """BGE-M3 本地嵌入模型（ragbasic-master/similarity.py）"""
    global _embed_model
    if _embed_model is None:
        with _embed_lock:
            if _embed_model is None:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                print(f"[embed] 正在加载 BGE-M3：{config.model_path}（首次约 20-40 秒）")
                _embed_model = HuggingFaceEmbedding(model_name=config.model_path)
    return _embed_model


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """批量文本向量化（medical/backend/main.py 的 EMBED_BATCH_SIZE 思路）"""
    model = get_embed_model()
    vectors: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        vectors.extend(model.get_text_embedding_batch(texts=batch))
        done = min(start + batch_size, len(texts))
        print(f"[embed] 已向量化 {done}/{len(texts)}")
    return vectors


def embed_query(text: str) -> List[float]:
    return get_embed_model().get_text_embedding(text)


def embedding_similarity(sentence1: str, sentence2: str, mode=None) -> float:
    """两段文本的相似度（similarity.py）"""
    if mode is None:
        from llama_index.core.base.embeddings.base import SimilarityMode
        mode = SimilarityMode.DEFAULT
    model = get_embed_model()
    v1 = model.get_text_embedding(sentence1)
    v2 = model.get_text_embedding(sentence2)
    return model.similarity(v1, v2, mode=mode)


# ============================================================ 后端 A：Milvus Lite
def _resolve_db_path() -> Path:
    """
    Milvus Lite 的库路径必须是纯 ASCII。
    faiss 在 Windows 用 C 的 fopen 写索引文件，非 ASCII 路径会报
    "could not open ...autoindex.idx for writing: No such file or directory"。
    """
    directory = Path(config.MILVUS_DB_DIR)
    if not str(directory).isascii():
        print(
            f"[store][警告] MILVUS_DB_DIR={directory} 含非 ASCII 字符，"
            f"Milvus Lite 无法写索引，已改用 {config.MILVUS_DB_FALLBACK}"
        )
        directory = Path(config.MILVUS_DB_FALLBACK)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{config.DB_NAME}.db"


class MilvusBackend:
    name = "milvus"

    def __init__(self):
        from pymilvus import MilvusClient
        self.db_path = _resolve_db_path()
        self.collection = config.COLLECTION
        self.dimension = config.embed_dim
        print(f"[store] Milvus 库文件：{self.db_path} (backend id={id(self)})")
        self.client = MilvusClient(str(self.db_path))

    def reset(self) -> None:
        if self.client.has_collection(self.collection):
            self.client.drop_collection(collection_name=self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            dimension=self.dimension,
            metric_type="COSINE",
        )

    def insert_rows(self, rows: List[Dict[str, Any]], batch_size: int = 200) -> int:
        total = len(rows)
        for start in range(0, total, batch_size):
            self.client.insert(collection_name=self.collection, data=rows[start:start + batch_size])
            if start + batch_size < total:
                time.sleep(0.2)
        return total

    def search(self, vector: List[float], top_k: int) -> List[dict]:
        self.client.load_collection(collection_name=self.collection)
        results = self.client.search(
            collection_name=self.collection,
            data=[vector],
            limit=top_k,
            output_fields=["id", "text"],
        )
        hits = results[0] if results else []
        return [
            {
                "id": h.get("id", h.get("entity", {}).get("id")),
                "text": h.get("entity", {}).get("text", ""),
                "score": h.get("distance", 0.0),   # COSINE：越大越相似
                "metadata": h.get("entity", {}),
            }
            for h in hits
        ]

    def count(self) -> int:
        if not self.client.has_collection(self.collection):
            return 0
        return self.client.get_collection_stats(self.collection).get("row_count", 0)


# ============================================================ 后端 B：InMemoryVectorStore
class MemoryBackend:
    """
    OpenAIEmbeddings + InMemoryVectorStore（LangChain/test30.py、test31.py）
    注意：需要 OPENAI_API_KEY，因此采用懒加载，避免选到该后端时才报缺少密钥。
    """

    name = "memory"

    def __init__(self):
        from langchain_core.documents import Document as LCDocument

        self._LCDocument = LCDocument
        self._embeddings = None
        self.store = None

    def _ensure(self):
        if self.store is None:
            from langchain_core.vectorstores import InMemoryVectorStore
            from langchain_openai import OpenAIEmbeddings

            self._embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            self.store = InMemoryVectorStore(embedding=self._embeddings)
        return self.store

    def reset(self) -> None:
        self.store = None
        self._ensure()

    def insert_rows(self, rows: List[Dict[str, Any]], batch_size: int = 200) -> int:
        self._ensure()
        docs = [
            self._LCDocument(
                page_content=r["text"],
                metadata={k: v for k, v in r.items() if k not in ("id", "vector", "text")},
            )
            for r in rows
        ]
        self.store.add_documents(docs)
        return len(docs)

    def search(self, query: str, top_k: int) -> List[dict]:
        self._ensure()
        results = self.store.similarity_search_with_score(query, k=top_k)
        return [
            {
                "id": i + 1,
                "text": doc.page_content,
                "score": float(score),      # 余弦相似度，越大越相似
                "metadata": doc.metadata,
            }
            for i, (doc, score) in enumerate(results)
        ]

    def count(self) -> int:
        return 0 if self.store is None else len(self.store.store)


_backend = None
_backend_lock = threading.Lock()


def get_backend():
    """按配置返回向量库后端（进程内单例，加锁防并发重复构造）"""
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                if config.VECTOR_BACKEND == "memory":
                    _backend = MemoryBackend()
                else:
                    _backend = MilvusBackend()
                print(f"[store] 向量后端 = {_backend.name} (id={id(_backend)})")
    return _backend


def build_index(chunks, reset: bool = True) -> int:
    """
    分块 -> 向量化 -> 入库

    :param chunks: llama_index Document 列表（来自 ragbase.chunkers）
    """
    if not chunks:
        raise ValueError("分块结果为空，请检查知识库目录与过滤条件")

    backend = get_backend()
    if reset:
        backend.reset()

    texts = [c.text for c in chunks]
    metas = [c.metadata for c in chunks]

    if isinstance(backend, MilvusBackend):
        vectors = embed_texts(texts)
        rows = [
            {"id": i + 1, "vector": v, "text": t, **{k: str(val) for k, val in m.items()}}
            for i, (t, v, m) in enumerate(zip(texts, vectors, metas))
        ]
    else:
        # memory 后端自带嵌入，不需要外部向量
        rows = [{"id": i + 1, "text": t, **m} for i, (t, m) in enumerate(zip(texts, metas))]

    inserted = backend.insert_rows(rows)
    print(f"[store] 入库完成：{inserted} 条")
    return inserted


def search(question: str, top_k: int = None) -> List[dict]:
    """检索，统一返回 [{"id","text","score","metadata"}]"""
    backend = get_backend()
    top_k = top_k or config.TOP_K

    if isinstance(backend, MilvusBackend):
        return backend.search(embed_query(question), top_k)

    # memory 后端：用 LangChain 自带的相似度检索（含 score）
    results = backend.store.similarity_search_with_score(question, k=top_k)
    return [
        {
            "id": i + 1,
            "text": doc.page_content,
            "score": float(score),      # 余弦相似度，越大越相似
            "metadata": doc.metadata,
        }
        for i, (doc, score) in enumerate(results)
    ]


def count() -> int:
    return get_backend().count()


def drop() -> None:
    get_backend().reset()
