"""
记忆与运行时上下文的独立演示（不依赖 LLM，可直接运行）

    python examples/memory_demo.py runtime     静态运行时上下文（LangGraph/test22.py）
    python examples/memory_demo.py travel      时间旅行：历史快照 + 改状态重放（test11/test21.py）
    python examples/memory_demo.py store       跨会话记忆 store（LangGraph/test13.py、test14.py）
"""
import sys
import uuid
from dataclasses import dataclass
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


# ============================================================ 1. 静态运行时上下文
@dataclass
class ContextSchema:
    user_id: str
    language: str = "zh"


class CtxState(TypedDict):
    messages: list
    user_name: str


def demo_runtime():
    def node(state: CtxState, runtime: Runtime[ContextSchema]):
        greeting = "你好" if runtime.context.language == "zh" else "hello"
        user_name = state.get("user_name", "Guest")
        return {"messages": [f"{greeting}，{user_name}（user_id={runtime.context.user_id}）！"]}

    b = StateGraph(CtxState, context_schema=ContextSchema)
    b.add_node(node)
    b.add_edge(START, "node")
    b.add_edge("node", END)

    g = b.compile()
    print(g.invoke(
        {"messages": [], "user_name": "小明"},
        context=ContextSchema(user_id="u-1", language="zh"),
    ))


# ============================================================ 2. 时间旅行
class TravelState(TypedDict):
    topic: str
    joke: str


def demo_travel():
    def generate_topic(state: TravelState):
        return {"topic": "押一付三"}

    def generate_joke(state: TravelState):
        return {"joke": f"关于「{state['topic']}」的笑话"}

    b = StateGraph(TravelState)
    b.add_sequence([generate_topic, generate_joke])
    b.add_edge(START, "generate_topic")
    b.add_edge("generate_joke", END)

    g = b.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "1"}}
    print("首次执行：", g.invoke({}, config=cfg))

    states = list(g.get_state_history(cfg))
    print(f"\n共 {len(states)} 个历史快照（最新的在最前）")
    for i, s in enumerate(states):
        print(f"  [{i}] next={s.next} values={s.values}")

    # states[1]：generate_topic 已执行完、generate_joke 尚未执行的那个快照
    target = states[1]
    new_cfg = g.update_state(target.config, values={"topic": "租房押金"})
    print("\n把 topic 改成「租房押金」后重放：", g.invoke(None, config=new_cfg))


# ============================================================ 3. 跨会话记忆
def demo_store():
    store = InMemoryStore()
    user_id = "user_123"
    namespace = (user_id, "prefs")

    store.put(namespace, str(uuid.uuid4()), {"city": "西安", "budget": "3000 以内"})
    store.put(namespace, str(uuid.uuid4()), {"room_type": "两室一厅"})

    print(f"namespace={namespace} 中的记忆：")
    for mem in store.search(namespace, limit=5):
        print("  ", mem.dict() if hasattr(mem, "dict") else mem)

    print(f"\n换个用户 user_456 查不到：{store.search(('user_456', 'prefs'))}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "runtime"
    {"runtime": demo_runtime,
     "travel": demo_travel,
     "store": demo_store}[name]()
