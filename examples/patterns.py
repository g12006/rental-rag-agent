"""
补充工作流模式（主图之外的独立演示）

    python examples/patterns.py parallel          并行分支（LangGraph/test8.py）
    python examples/patterns.py orchestrator      协调者-工作者 Send（LangGraph/test10.py）
    python examples/patterns.py subgraph          子图（LangGraph/test29.py、test30.py）
"""
import operator
import sys
from typing import Annotated, TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send


# ============================================================ 1. 并行分支
class AnalyzeState(TypedDict):
    topic: str
    price: str
    location: str
    contract: str
    report: str


def price_task(state: AnalyzeState):
    return {"price": "同小区均价 3200-3800 元/月，押一付三为主流"}


def location_task(state: AnalyzeState):
    return {"location": "地铁 3 号线沿线房源去化最快，通勤 30 分钟内占比 62%"}


def contract_task(state: AnalyzeState):
    return {"contract": "标准合同含押金退还条款，退租需提前 30 天书面通知"}


def combine(state: AnalyzeState):
    report = (
        f"【{state['topic']}】\n"
        f"价格：{state['price']}\n"
        f"地段：{state['location']}\n"
        f"合同：{state['contract']}\n"
    )
    return {"report": report}


def demo_parallel():
    wf = StateGraph(AnalyzeState)
    wf.add_node(price_task)
    wf.add_node(location_task)
    wf.add_node(contract_task)
    wf.add_node(combine)

    for n in ("price_task", "location_task", "contract_task"):
        wf.add_edge(START, n)
        wf.add_edge(n, "combine")
    wf.add_edge("combine", END)

    result = wf.compile().invoke({"topic": "租房市场三维度分析"})
    print(result["report"])


# ============================================================ 2. 协调者-工作者
class WorkState(TypedDict):
    topic: str
    sections: list
    completed_sections: Annotated[list, operator.add]
    final_report: str


def demo_orchestrator():
    from langchain_core.messages import HumanMessage
    from pydantic import BaseModel

    from agent.model import get_chat_model, structured

    class Section(BaseModel):
        name: str
        description: str

    class Sections(BaseModel):
        sections: list[Section]

    model = get_chat_model()
    planner = structured(Sections)

    def orchestrator(state: WorkState):
        result = planner.invoke([
            HumanMessage(content=f"为「{state['topic']}」制定报告大纲，3 到 4 个章节")
        ])
        return {"sections": result.sections}

    def worker(state: WorkState):
        section = state["section"]
        r = model.invoke([HumanMessage(content=f"编写章节：{section.name}，要求：{section.description}")])
        return {"completed_sections": [f"## {section.name}\n{r.content}"]}

    def synthesizer(state: WorkState):
        return {"final_report": "\n\n---\n\n".join(state["completed_sections"])}

    def assign_workers(state: WorkState):
        return [Send("worker", {"section": s}) for s in state["sections"]]

    b = StateGraph(WorkState)
    b.add_node(orchestrator)
    b.add_node(worker)
    b.add_node(synthesizer)
    b.add_edge(START, "orchestrator")
    b.add_conditional_edges("orchestrator", assign_workers)
    b.add_edge("worker", "synthesizer")
    b.add_edge("synthesizer", END)

    result = b.compile().invoke({"topic": "租房平台押金纠纷处理指南"})
    print(result["final_report"])


# ============================================================ 3. 子图
class SubState(TypedDict):
    sub_result: str
    question: str


class ParentState(TypedDict):
    question: str
    answer: str


def demo_subgraph():
    def sub_node_1(state: SubState):
        return {"sub_result": f"检索到与「{state['question']}」相关的 3 条政策"}

    def sub_node_2(state: SubState):
        return {"question": state["question"] + "（已校对）"}

    sub = StateGraph(SubState)
    sub.add_sequence([sub_node_1, sub_node_2])
    sub.add_edge(START, "sub_node_1")
    sub_graph = sub.compile()

    def parent_node(state: ParentState):
        r = sub_graph.invoke({"question": state["question"], "sub_result": ""})
        return {"answer": f"{r['sub_result']}｜问题：{r['question']}"}

    b = StateGraph(ParentState)
    b.add_node(parent_node)
    b.add_edge(START, "parent_node")
    b.add_edge("parent_node", END)

    print(b.compile().invoke({"question": "押金多久退"})["answer"])

    # 另一种：把编译好的子图直接当成主图的节点（test30）
    b2 = StateGraph(ParentState)
    b2.add_node("parent_node", parent_node)
    b2.add_edge(START, "parent_node")
    b2.add_edge("parent_node", END)
    print("(子图作为节点的写法见 LangGraph/test30.py，本文件演示节点内调用子图)")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "parallel"
    {"parallel": demo_parallel,
     "orchestrator": demo_orchestrator,
     "subgraph": demo_subgraph}[name]()
