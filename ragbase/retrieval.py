"""
检索 -> 过滤 -> 上下文压缩 -> 提示词增强

来源：
    ragbasic-master/ragPromptEnhancer.py
        retrieve        向量检索
        filter_chunks   按相似度过滤
        compress_context 按字符数压缩
        build_prompt    提示词模板拼装
"""
from typing import Dict, List, Tuple

import config
from ragbase import store

# 提示词模板：与 ragPromptEnhancer.PROMPT_TEMPLATE 同构，补充了"保留原文表述"的要求
PROMPT_TEMPLATE = """【角色】你是租房平台的智能客服，只基于文档回答问题。
【任务】根据文档回答用户的问题，禁止编造文档以外的信息。
【上下文】
{context}
【用户的问题】
{question}
【输出要求】
1. 准确、简洁，能分条就分条；
2. 当文档中确实没有相关内容时，直接回答“文档中不存在”；
3. 涉及押金、合同、违约等条款时，尽量保留文档原文表述；
4. 不要使用文档以外的知识去回答。
"""


def retrieve(question: str, top_k: int = None) -> List[dict]:
    """向量检索"""
    return store.search(question, top_k=top_k)


def filter_chunks(hits: List[dict], min_score: float = None) -> List[dict]:
    """
    按相似度过滤。

    注意：ragbasic-master/ragPromptEnhancer.py 里写作
        if hit.get("distance") > max_distance: continue
    但 Milvus 的 COSINE 度量下 distance 越大越相似，方向是反的，
    本实现统一成 score（余弦相似度）后按 "越大越相似" 过滤。
    """
    min_score = config.MIN_SCORE if min_score is None else min_score
    kept = []
    for hit in hits:
        if hit.get("score", 0.0) < min_score:
            continue
        text = (hit.get("text") or "").strip()
        if len(text) >= 10:          # 太短的片段直接舍弃
            kept.append(hit)
    return kept


def compress_context(texts: List[str], max_chars: int = None) -> str:
    """按字符上限拼接上下文，去重 + 截断（compress_context）"""
    max_chars = config.MAX_CHARS if max_chars is None else max_chars
    seen = set()
    parts: List[str] = []
    total = 0

    for text in texts:
        if text in seen:
            continue
        seen.add(text)

        sep_len = 2 if parts else 0
        if total + sep_len + len(text) > max_chars:
            remain = max_chars - total - sep_len
            if remain > 20:          # 保留 20 字以上才截断，避免语义断裂
                parts.append(text[:remain] + "....")
            break

        parts.append(text)
        total += sep_len + len(text)

    return "\n\n".join(parts)


def build_prompt(question: str, context: str) -> str:
    if not context:
        context = "(无相关的文档片段)"
    return PROMPT_TEMPLATE.format(context=context, question=question)


def get_context(
    question: str,
    top_k: int = None,
    min_score: float = None,
    max_chars: int = None,
) -> Tuple[str, List[Dict]]:
    """
    检索 + 过滤 + 压缩，返回 (上下文字符串, 来源列表)
    """
    hits = retrieve(question, top_k=top_k)
    kept = filter_chunks(hits, min_score=min_score)

    sources = [
        {
            "rank": i,
            "id": h.get("id"),
            "score": round(float(h.get("score", 0.0)), 4),
            "text": h.get("text", ""),
            "source": h.get("metadata", {}).get("file_path")
                      or h.get("metadata", {}).get("source_file_path", ""),
        }
        for i, h in enumerate(kept, 1)
    ]

    context = compress_context([h.get("text", "") for h in kept], max_chars=max_chars)
    return context, sources


def get_prompt(question: str, **kwargs) -> str:
    """一步拿到增强后的提示词（ragPromptEnhancer.get_prompt）"""
    context, _ = get_context(question, **kwargs)
    return build_prompt(question, context)


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "押金怎么退？"
    print(get_prompt(q))
