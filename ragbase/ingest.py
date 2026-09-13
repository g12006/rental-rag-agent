"""
文档加载 + 数据清洗

来源：
    ragbasic-master/util.py        —— parse_all_formats（SimpleDirectoryReader）
    ragbasic-master/dataClean.py   —— clean_text / clean_ppt / clean_doc
"""
import re
import unicodedata
from pathlib import Path
from typing import List

from llama_index.core import SimpleDirectoryReader, Document

# ---------------------- 预编译正则：Markdown 标记清理规则 ----------------------
_MD_HEADING_BOLD = re.compile(r"^#\s*\*\*(.+?)\*\*\s*$", re.MULTILINE)
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_LEADING_HASH = re.compile(r"^#\s+", re.MULTILINE)
_INLINE_HASH_WRAP = re.compile(r"#\s*(.+?)\s*#")
_TRAILING_HASH = re.compile(r"#\s*$")

# ---------------------- 预编译正则：PPT 解析噪声 ----------------------
_PPTX_TITLE_LINE = re.compile(r"^Title:\s*.+\s*$", re.MULTILINE)
_PPTX_SEPARATOR = re.compile(r"^-{3,}\s*$", re.MULTILINE)
_PPTX_SPEAKER_NOTES = re.compile(r"^\[Speaker Notes\]:\s*", re.MULTILINE)

# ---------------------- 预编译正则：通用脏字符 ----------------------
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH = re.compile(r"[\ufeff\u200b\u200c\u200d]")
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")


def parse_all_formats(input_dir) -> List[Document]:
    """加载目录下所有格式的文件（pdf / pptx / csv / txt / md ...）"""
    reader = SimpleDirectoryReader(input_dir=str(input_dir))
    return reader.load_data()


def clean_ppt(text: str) -> str:
    """清理 PPT 解析产生的结构噪声（标题行、分割线、备注前缀、# 号）"""
    text = _PPTX_TITLE_LINE.sub("", text)
    text = _PPTX_SEPARATOR.sub("", text)
    text = _PPTX_SPEAKER_NOTES.sub("", text)
    text = _MD_HEADING_BOLD.sub(r"\1", text)
    text = _MD_BOLD.sub(r"\1", text)

    while True:
        cleaned = _INLINE_HASH_WRAP.sub(r"\1", text)
        if cleaned == text:
            break
        text = cleaned

    text = _LEADING_HASH.sub("", text)
    text = _TRAILING_HASH.sub("", text)
    text = re.sub(r"\s+#\s+", "", text)
    return text.replace("#", "")


def clean_text(text: str, source_suffix: str = "") -> str:
    """对单段文本执行清洗：隐形字符 -> 全角归一 -> 控制字符 -> 换行 -> 空行压缩"""
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("\ufeff", "")
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if source_suffix.lower() in {".pptx", ".ppt", ".pptm"}:
        text = clean_ppt(text)

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = _MULTI_BLANK_LINES.sub("\n\n", "\n".join(lines))
    return text.strip()


def clean_doc(doc: Document) -> Document:
    """构建清洗后的新 Document 对象（保留原 metadata）"""
    suffix = Path(doc.metadata.get("file_path", "")).suffix
    return Document(text=clean_text(doc.text, suffix), metadata=doc.metadata)


def load_clean_documents(input_dir, source_filter: str = "") -> List[Document]:
    """
    加载 -> 过滤 -> 清洗，返回干净的 Document 列表

    :param input_dir:     文档目录（或单个文件路径）
    :param source_filter: 只保留 file_path 中包含该关键词的文档；为空表示全部
    """
    docs = parse_all_formats(input_dir)

    if source_filter:
        docs = [
            d for d in docs
            if source_filter in str(d.metadata.get("file_path", ""))
        ]

    cleaned = []
    for d in docs:
        new_doc = clean_doc(d)
        if new_doc.text:
            cleaned.append(new_doc)
    return cleaned


if __name__ == "__main__":
    import config

    documents = load_clean_documents(config.DOCUMENT_PATH, config.DOC_SOURCE_FILTER)
    print(f"清洗后共 {len(documents)} 篇文档")
    for i, d in enumerate(documents, 1):
        path = d.metadata.get("file_path")
        print(f"\n第 {i} 篇 | {path}")
        print(f"字符数：{len(d.text)}")
        print(d.text[:300])
