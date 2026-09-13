"""
把 RAG 检索能力封装成 LangChain 工具

来源：
    LangChain/test5.py  @tool + Annotated 参数注解
    LangChain/test6.py  StructuredTool.from_function + args_schema + content_and_artifact
"""
from typing import List, Tuple

from langchain_core.tools import StructuredTool, tool
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from ragbase import retrieval

import config


@tool
def search_rental_knowledge(
    query: Annotated[str, ..., "用于在租房知识库中做向量检索的查询语句，应保留关键业务词"],
) -> str:
    """检索租房平台知识库（房源、押金、合同、签约、项目架构等），返回最相关的文档片段。"""
    context, _ = retrieval.get_context(query, top_k=config.TOP_K)
    return context or "(未检索到相关文档片段)"


class SearchInput(BaseModel):
    """search_with_sources 的入参"""

    query: str = Field(description="用于在租房知识库中检索的查询语句")
    top_k: int = Field(default=3, description="返回的片段条数")


def _search_with_sources(query: str, top_k: int = 3) -> Tuple[str, List[dict]]:
    """
    content_and_artifact：
        content  -> 喂给 LLM 的上下文字符串
        artifact -> 保留给下游（API / 前端）展示的来源明细
    """
    return retrieval.get_context(query, top_k=top_k)


search_with_sources_tool = StructuredTool.from_function(
    func=_search_with_sources,
    name="search_with_sources",
    description="检索租房知识库并同时返回来源明细，适用于需要标注出处的回答",
    args_schema=SearchInput,
    response_format="content_and_artifact",
)


# 供 ToolNode / bind_tools 使用
TOOLS = [search_rental_knowledge]
