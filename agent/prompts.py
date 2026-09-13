"""
提示词模板 / 少样本示例 / 结构化输出的 schema

来源：
    LangChain/test19.py  ChatPromptTemplate + MessagesPlaceholder
    LangChain/test21.py  FewShotChatMessagePromptTemplate
    LangChain/test9.py   with_structured_output（Pydantic）
    LangChain/test11.py  Optional + description 的信息提取 schema
    LangGraph/test3.py   GRADE_PROMPT + GradeDocuments
    LangGraph/test9.py   Route（路由模式）
"""
from typing import List, Literal, Optional

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from pydantic import BaseModel, Field


# ============================================================ 结构化输出 schema
class Route(BaseModel):
    """用户问题的路由决策（LangGraph/test9.py）"""

    decision: Literal["rental_qa", "small_talk", "human_handoff"] = Field(
        description=(
            "rental_qa：与租房平台业务相关的问题（房源、押金、合同、签约、微服务架构等）；"
            "small_talk：打招呼、闲聊、与租房无关的通用问题；"
            "human_handoff：投诉、退款、纠纷等需要人工客服介入的敏感问题"
        )
    )


class GradeDocuments(BaseModel):
    """检索片段与问题的相关性评分（LangGraph/test3.py）"""

    score: str = Field(description="相关性评分：相关为 'yes'，不相关为 'no'")


class UserProfile(BaseModel):
    """从对话中提取的租房偏好，用于跨会话记忆（LangGraph/test14.py 的 Person）"""

    city: Optional[str] = Field(default=None, description="用户提到的城市")
    budget: Optional[str] = Field(default=None, description="用户提到的租金预算")
    room_type: Optional[str] = Field(default=None, description="用户提到的户型，如两室一厅")
    name: Optional[str] = Field(default=None, description="用户的称呼")


# ============================================================ 提示词
ROUTER_SYSTEM = (
    "你是租房平台客服系统的意图分类器，只根据用户最后一条消息判断意图。\n\n"
    "三类取值的判定标准：\n"
    "1. rental_qa：与租房平台业务相关的问题。包括项目介绍、技术栈、架构设计、"
    "房源搜索/发布、缓存、消息队列、即时通信、部署运维，以及任何关于「这个项目」的提问。\n"
    "2. small_talk：打招呼、感谢、纯闲聊，与租房和本项目都无关。\n"
    "3. human_handoff：投诉、退款、纠纷、法律维权等需要人工客服介入的敏感问题。\n\n"
    "示例：\n"
    "  介绍一下这个项目 -> rental_qa\n"
    "  项目用了哪些技术栈 -> rental_qa\n"
    "  房源搜索是怎么实现的 -> rental_qa\n"
    "  你好 -> small_talk\n"
    "  谢谢 -> small_talk\n"
    "  我要投诉你们 -> human_handoff\n"
)

GRADE_PROMPT = (
    "你是一个评分员，评估检索到的文档与用户问题的相关性。\n"
    "以下是检索到的文档：\n\n{context}\n\n"
    "以下是用户的问题：{question}\n"
    "如果文档包含与用户问题相关的关键字或语义，则将其评为相关。\n"
    "给出一个二元分数 'yes' 或 'no'，以表明该文档是否与问题相关。"
)

REWRITE_PROMPT = (
    "查看输入并尝试推断潜在的语义意图/含义。\n"
    "这是最初的问题：\n ------- \n{question}\n ------- \n"
    "提出一个改进后的问题："
)

PROFILE_SYSTEM = (
    "你是一个提取信息的专家，只从文本中提取与租房相关的用户偏好信息。"
    "如果某个属性没有提到，返回 null。"
)

ANSWER_SYSTEM = (
    "你是租房平台的智能客服。\n"
    "严格基于给定的【上下文】回答用户问题，不要编造上下文中没有的信息；"
    "如果上下文不足以回答，直接说“文档中不存在”。\n"
    "回答要准确、简洁，能分条就分条。\n"
    "如果用户有历史偏好，回答时尽量贴合。\n\n"
    "【历史偏好】\n{user_profile}\n\n"
    "【上下文】\n{context}"
)

# 少样本示例：教模型"答不出来要直说"，避免幻觉
ANSWER_EXAMPLES = [
    {
        "q": "你们平台支持押一付三吗？",
        "a": "文档中不存在。上下文中未提及支付方式相关条款，建议联系人工客服确认。",
    },
    {
        "q": "介绍一下这个项目。",
        "a": "本项目是**基于脚手架的微服务在线租房系统**，业务模型对标贝壳、安居客、闲鱼等应用。"
            "后端采用 Spring Cloud 微服务架构，前端包含 Web 管理端与微信小程序端。",
    },
]

_example_prompt = ChatPromptTemplate([("human", "{q}"), ("ai", "{a}")])

answer_few_shot = FewShotChatMessagePromptTemplate(
    examples=ANSWER_EXAMPLES,
    example_prompt=_example_prompt,
)

answer_prompt = ChatPromptTemplate(
    [
        ("system", ANSWER_SYSTEM),
        answer_few_shot,
        ("human", "{question}"),
    ]
)

SMALL_TALK_SYSTEM = (
    "你是租房平台的智能客服助手。用户现在只是打招呼或闲聊，"
    "请用一句话友好回应，并说明你可以解答租房平台相关的问题。"
)

HANDOFF_SYSTEM = (
    "用户的问题需要人工客服介入。请用一句话说明已转接人工客服，"
    "并提示用户稍等，不要编造任何解决方案。"
)
