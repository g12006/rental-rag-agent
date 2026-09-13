"""
文本分块：6 种策略 + LangChain 分割器

来源：
    ragbasic-master/fixedSizeChunk.py   固定大小分块
    ragbasic-master/sentenceChunk.py    句子分块
    ragbasic-master/recursiveChunk.py   递归分块（TokenTextSplitter）
    ragbasic-master/semanticChunk.py    语义分块（SemanticSplitterNodeParser）
    ragbasic-master/structureChunk.py   结构分块（pdf / ppt / csv / txt 分流）
    ragbasic-master/llmChunk.py         LLM 分块
    LangChain/test28.py                 RecursiveCharacterTextSplitter
"""
import csv
import re
from typing import Callable, List, Optional

from llama_index.core.schema import Document

import config
from ragbase.ingest import load_clean_documents

# PDF / Markdown 页内次级结构切分点：编号问题（1.）或项目符号（• ◦ -）前
# 来自 ragbasic-master/structureChunk.py
_SUBSECTION_RE = re.compile(r"(?=\n(?:\d+\.|•|◦|\-)\s*)")

# 单个分块的最大字符数（structureChunk.py 的 max_chunk_chars）
MAX_CHUNK_CHARS = 1500


# ============================================================ 1. 固定大小分块
def fixed_size_chunk_documents(
    input_dir,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    chunks: List[Document] = []
    step = chunk_size - chunk_overlap
    documents = load_clean_documents(input_dir, config.DOC_SOURCE_FILTER)

    for d in documents:
        text = d.text
        path = d.metadata.get("file_path")
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            m = dict(d.metadata)
            m.update({
                "source_file_path": path,
                "chunk_index": idx,
                "chunk_start": start,
                "chunk_end": end,
                "chunk_strategy": "fixed",
            })
            chunks.append(Document(text=text[start:end], metadata=m))
            if end >= len(text):
                break
            start += step
            idx += 1
    return chunks


# ============================================================ 2. 句子分块
def sentence_chunk_documents(
    input_dir,
    max_sentences: int = 5,
) -> List[Document]:
    chunks: List[Document] = []
    documents = load_clean_documents(input_dir, config.DOC_SOURCE_FILTER)

    for d in documents:
        text = d.text
        if not text:
            continue
        path = d.metadata.get("file_path")

        sents = [
            s.strip()
            for s in re.split(r"(?<=[。！？!?\.])\s+", text)
            if s.strip()
        ]
        # 没有空白分隔时，按句末标点直接切
        if len(sents) == 1:
            sents = [
                s.strip()
                for s in re.split(r"(?<=[。！？!?\.])", text)
                if s.strip()
            ]

        for i in range(0, len(sents), max_sentences):
            part = " ".join(sents[i:i + max_sentences]).strip()
            if not part:
                continue
            m = dict(d.metadata)
            m.update({
                "source_file_path": path,
                "chunk_index": i // max_sentences,
                "chunk_start": i,
                "chunk_end": min(i + max_sentences, len(sents)),
                "chunk_strategy": "sentence",
            })
            chunks.append(Document(text=part, metadata=m))
    return chunks


# ============================================================ 3. 递归分块
def recursive_chunk_documents(input_dir) -> List[Document]:
    """分隔符优先级：段落 -> 换行 -> 中英文句末 -> 分号 -> 逗号 -> 空格 -> 单字符"""
    from llama_index.core.node_parser import TokenTextSplitter

    separators = [
        "\n\n", "\n",
        "。", "！", "？",
        ". ", "! ", "? ",
        "；", ";",
        "，", ",",
        " ", "",
    ]
    splitter = TokenTextSplitter(
        separator=separators[0],
        backup_separators=separators[1:],
    )

    documents = load_clean_documents(input_dir, config.DOC_SOURCE_FILTER)
    chunks: List[Document] = []

    for d in documents:
        if not d.text:
            continue
        path = d.metadata.get("file_path")
        nodes = splitter.get_nodes_from_documents(
            [Document(text=d.text, metadata={"file_path": path})]
        )
        for idx, n in enumerate(nodes):
            m = dict(n.metadata)
            m.update({
                "source_file_path": path,
                "chunk_index": idx,
                "chunk_strategy": "recursive",
            })
            chunks.append(Document(text=n.text, metadata=m))
    return chunks


# ============================================================ 4. 语义分块
def semantic_chunk_documents(input_dir) -> List[Document]:
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from ragbase.store import get_embed_model

    documents = load_clean_documents(input_dir, config.DOC_SOURCE_FILTER)
    splitter = SemanticSplitterNodeParser(embed_model=get_embed_model())

    chunks: List[Document] = []
    for d in documents:
        if not d.text:
            continue
        path = d.metadata.get("file_path")
        nodes = splitter.get_nodes_from_documents(
            [Document(text=d.text, metadata=d.metadata)]
        )
        for idx, n in enumerate(nodes):
            m = dict(n.metadata)
            m.update({
                "source_file_path": path,
                "chunk_index": idx,
                "chunk_strategy": "semantic",
            })
            chunks.append(Document(text=n.text, metadata=m))
    return chunks


# ============================================================ 5. 结构分块
def _extract_section_text(section: dict) -> str:
    return section.get("content")


def _chunk_ppt_document(d: Document) -> List[Document]:
    sections = d.metadata.get("text_sections") or []
    slide_title = d.metadata.get("title")
    texts = [t for t in (_extract_section_text(s) for s in sections) if t]

    full_text = "\n".join(texts)
    if len(full_text) <= MAX_CHUNK_CHARS:
        return [Document(text=full_text, metadata={"chunk_index": 0, "chunk_strategy": "structure"})]

    chunk_text = full_text
    if slide_title and slide_title not in chunk_text:
        chunk_text = f"{slide_title}\n{chunk_text}"
    return [Document(text=chunk_text, metadata={"section_index": 0, "chunk_strategy": "structure"})]


def _merge_parts(parts: List[str], max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """
    把过碎的片段按 max_chars 合并。

    为什么需要：_SUBSECTION_RE 会在每一个 "- " 列表项前切一刀，
    租房 Q&A 通篇是 "- **回答1：**" 这种列表，直接切会得到 244 个 100 字左右的碎块，
    语义不完整、检索效果差。因此这里做一次相邻合并（仍沿用同一个切分正则）。
    """
    merged: List[str] = []
    buf = ""
    for p in parts:
        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= max_chars:
            buf = f"{buf}\n{p}"
        else:
            merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)
    return merged


def _chunk_pdf_document(d: Document) -> List[Document]:
    """按页内次级结构（编号 / 项目符号）切分，并对过碎片段做合并"""
    text = d.text
    if not text:
        return []
    base_meta = {"page_label": d.metadata.get("page_label"), "chunk_strategy": "structure"}

    if len(text) <= MAX_CHUNK_CHARS:
        meta = dict(base_meta)
        meta["chunk_index"] = 0
        return [Document(text=text, metadata=meta)]

    parts = [p.strip() for p in _SUBSECTION_RE.split(text) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts = _merge_parts(parts)

    chunks = []
    for idx, part in enumerate(parts):
        meta = dict(base_meta)
        meta["chunk_index"] = idx
        chunks.append(Document(text=part, metadata=meta))
    return chunks


def _read_csv_header(path: str) -> str:
    with open(path, newline="", encoding="utf-8-sig") as f:
        row = next(csv.reader(f), None)
        return ",".join(row)


def _chunk_csv_document(d: Document) -> List[Document]:
    if not d.text:
        return []
    lines = [line.strip() for line in d.text.splitlines() if line.strip()]
    header = _read_csv_header(d.metadata.get("file_path"))

    chunks = []
    for idx, line in enumerate(lines):
        meta = {"chunk_index": idx, "chunk_strategy": "structure"}
        chunks.append(Document(text=f"{header}\n{line}", metadata=meta))
    return chunks


def _chunk_single_document(d: Document) -> List[Document]:
    if d.metadata.get("text_sections"):
        return _chunk_ppt_document(d)
    if d.metadata.get("page_label"):
        return _chunk_pdf_document(d)
    if str(d.metadata.get("file_name", "")).endswith(".csv"):
        return _chunk_csv_document(d)
    if str(d.metadata.get("file_name", "")).endswith((".md", ".markdown")):
        # 扩展点：原 structureChunk.py 对 md 会走到最后的 `return [d]`（整篇不分块）。
        # 租房 Q&A 通篇是 "- **回答1：**" 这类列表项，
        # 因此这里复用同一个 _SUBSECTION_RE，让 md 也能按项目符号切分。
        return _chunk_pdf_document(d)
    return [d]


def structure_chunk_documents(input_dir) -> List[Document]:
    documents = load_clean_documents(input_dir, config.DOC_SOURCE_FILTER)
    chunks: List[Document] = []
    for d in documents:
        chunks.extend(_chunk_single_document(d))
    return chunks


# ============================================================ 6. LLM 分块
CHUNK_DELIMITER = "===CHUNK==="

# 单次喂给 LLM 的字符上限：超过就先切窗口（否则输出会被 max_tokens 截断）
LLM_CHUNK_WINDOW = 6000

LLM_CHUNK_SYSTEM_PROMPT = f"""你是RAG知识库文档分块助手。请把用户提供的文档切分为多个语义完整、适合向量检索的分块。
要求：
1. 每个分块围绕一个独立的主题
2. 尽量不要拆分段落，不要拆分句子
3. 保持原始文本的语义不变
4. 各个分块之间用{CHUNK_DELIMITER}分隔开
5. 只输出分块的正文，不要额外的信息
"""


def _parse_llm_chunks(response: str) -> List[str]:
    parts = re.split(rf"\n?{CHUNK_DELIMITER}\n?", response.strip())
    chunks = [p.strip() for p in parts if p.strip()]
    return chunks or [response.strip()]


def _split_for_llm(text: str, max_chars: int = LLM_CHUNK_WINDOW) -> List[str]:
    """
    把长文档先切成窗口，再逐窗口交给 LLM。

    为什么需要：整篇 24,709 字符一次性喂进去，模型要吐回几乎等长的文本，
    会被 max_tokens 截断 —— 实测只返回 72% 的内容，知识库直接少掉近三成。
    这里沿用 _SUBSECTION_RE 先按项目符号切，再用 _merge_parts 合并到 max_chars，
    每个窗口单独调一次 LLM，输出就不会超长了。
    """
    if len(text) <= max_chars:
        return [text]
    parts = [p.strip() for p in _SUBSECTION_RE.split(text) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return _merge_parts(parts, max_chars)


def llm_chunk_documents(
    input_dir,
    max_chunk_chars: int = 500,
    chat_fn: Optional[Callable[[str, str], str]] = None,
) -> List[Document]:
    """
    :param chat_fn: (user_prompt, system_prompt) -> str；默认使用 agent.model.chat_text
    """
    if chat_fn is None:
        from agent.model import chat_text
        chat_fn = chat_text

    documents = load_clean_documents(input_dir, config.DOC_SOURCE_FILTER)
    chunks: List[Document] = []
    idx = 0

    for d in documents:
        if not d.text:
            continue
        for window in _split_for_llm(d.text):
            user_prompt = (
                f"请将以下文档分块，每个块不超过{max_chunk_chars}字符：\n\n{window}"
            )
            for chunk_text in _parse_llm_chunks(
                chat_fn(user_prompt, LLM_CHUNK_SYSTEM_PROMPT)
            ):
                chunks.append(Document(
                    text=chunk_text,
                    metadata={
                        "source_file_path": d.metadata.get("file_path"),
                        "chunk_index": idx,
                        "chunk_strategy": "llm",
                    },
                ))
                idx += 1
    return chunks


# ============================================================ 7. LangChain 分割器
def langchain_split_documents(
    input_dir,
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    LangChain 侧的文本分割器（LangChain/test28.py、test30.py）
    与上面的 6 种策略并存，用于对比不同分割器对检索效果的影响。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    documents = load_clean_documents(input_dir, config.DOC_SOURCE_FILTER)
    chunks: List[Document] = []
    idx = 0
    for d in documents:
        if not d.text:
            continue
        for piece in splitter.split_text(d.text):
            m = dict(d.metadata)
            m.update({
                "source_file_path": d.metadata.get("file_path"),
                "chunk_index": idx,
                "chunk_strategy": "langchain_recursive",
            })
            chunks.append(Document(text=piece, metadata=m))
            idx += 1
    return chunks


# ============================================================ 统一入口
STRATEGIES = {
    "fixed": fixed_size_chunk_documents,
    "sentence": sentence_chunk_documents,
    "recursive": recursive_chunk_documents,
    "semantic": semantic_chunk_documents,
    "structure": structure_chunk_documents,
    "llm": llm_chunk_documents,
    "langchain": langchain_split_documents,
}


def build_chunks(strategy: str = None, input_dir=None, **kwargs) -> List[Document]:
    """按策略名构建分块"""
    strategy = strategy or config.CHUNK_STRATEGY
    if strategy not in STRATEGIES:
        raise ValueError(f"未知分块策略 {strategy}，可选：{list(STRATEGIES)}")
    input_dir = input_dir or config.DOCUMENT_PATH
    chunks = STRATEGIES[strategy](input_dir, **kwargs)
    print(f"[chunk] 策略={strategy} 目录={input_dir} -> {len(chunks)} 个分块")
    return chunks


if __name__ == "__main__":
    for name in STRATEGIES:
        try:
            result = build_chunks(name)
            avg = sum(len(c.text) for c in result) // max(len(result), 1)
            print(f"  平均 {avg} 字符\n")
        except Exception as e:
            print(f"  {name} 失败：{type(e).__name__}: {e}\n")
