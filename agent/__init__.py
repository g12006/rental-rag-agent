"""
agent —— 组件层（LangChain）+ 编排层（LangGraph）

    model.py   聊天模型初始化
    prompts.py 提示词模板 / 少样本 / 结构化输出 schema
    tools.py   @tool 与 StructuredTool 封装检索能力
    graph.py   LangGraph 主图：路由 -> 检索 -> 评分 -> 重写 -> 生成 -> 人工审核
"""
