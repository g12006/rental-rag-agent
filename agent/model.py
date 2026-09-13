"""
聊天模型初始化

来源：
    LangChain/test2.py  ChatOpenAI(model / api_key / base_url / temperature / max_tokens)
    LangChain/test3.py  init_chat_model + configurable_fields（可配置模型 / 模型模拟器）
"""
from functools import lru_cache

import config


@lru_cache(maxsize=1)
def get_chat_model():
    """DeepSeek 聊天模型（OpenAI 兼容协议）"""
    from langchain_openai import ChatOpenAI

    if not config.DEEPSEEK_API_KEY:
        raise ValueError(
            "未配置 DEEPSEEK_API_KEY，请在 rental_rag_agent/.env 中填写，"
            "或设置环境变量 DEEPSEEK_API_KEY"
        )

    return ChatOpenAI(
        model=config.CHAT_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )


def get_configurable_model():
    """
    可配置模型：真正的模型 / 参数在 .invoke(config=...) 时才确定。
    对应 LangChain/test3.py 的 configurable_fields + config_prefix 用法。
    """
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=config.CHAT_MODEL,
        model_provider="openai",
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        configurable_fields=("model", "model_provider", "max_tokens", "temperature"),
        config_prefix="first",
    )


def structured(schema, **kwargs):
    """
    结构化输出。

    ⚠️ DeepSeek 不支持 OpenAI 的 `response_format: json_schema`，
    `with_structured_output(schema)` 的默认路径会 400：
        "This response_format type is unavailable now"
    实测只有 method="function_calling" 可用（json_mode 也会 400）。
    """
    return get_chat_model().with_structured_output(schema, method="function_calling", **kwargs)


def chat_text(user_prompt: str, system_prompt: str = "") -> str:
    """一次性对话，返回文本。供 llmChunk 等纯文本场景使用"""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=user_prompt))
    return get_chat_model().invoke(messages).content
