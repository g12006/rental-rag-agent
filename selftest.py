"""项目自检：语法 + 导入 + 分块 + 离线示例（不需要 API Key）"""
import subprocess
import sys
import traceback

PY = sys.executable
HERE = __file__.rsplit("\\", 1)[0]


def run(desc, fn):
    print(f"\n{'=' * 60}\n{desc}\n{'=' * 60}")
    try:
        fn()
        print(f"[OK] {desc}")
        return True
    except Exception:
        traceback.print_exc()
        print(f"[FAIL] {desc}")
        return False


def t_ingest():
    from ragbase.ingest import load_clean_documents
    import config
    docs = load_clean_documents(config.DOCUMENT_PATH, config.DOC_SOURCE_FILTER)
    assert docs, "清洗后没有文档"
    print(f"文档数={len(docs)} 首篇字符数={len(docs[0].text)}")
    print("预览:", docs[0].text[:120].replace("\n", " "))


def t_chunkers():
    from ragbase.chunkers import build_chunks
    for s in ["fixed", "sentence", "structure", "langchain"]:
        chunks = build_chunks(s)
        avg = sum(len(c.text) for c in chunks) // max(len(chunks), 1)
        print(f"  {s:12s} -> {len(chunks):4d} 块，平均 {avg} 字符")


def t_recursive_token():
    from ragbase.chunkers import build_chunks
    chunks = build_chunks("recursive")
    print(f"  recursive(TokenTextSplitter) -> {len(chunks)} 块")


def t_graph_import():
    from agent.graph import graph
    nodes = set(graph.get_graph().nodes)
    print("图节点:", sorted(n for n in nodes if n not in ("__start__", "__end__")))


def t_prompts():
    from agent.prompts import answer_prompt, Route, GradeDocuments, UserProfile
    msgs = answer_prompt.invoke({
        "context": "押金在退租后 7 个工作日内退还。",
        "user_profile": "(暂无历史偏好)",
        "question": "押金多久退？",
    })
    print(f"提示词消息数={len(msgs.to_messages())}（含少样本 2 组示例 -> 应为 1+4+1=6）")
    for m in msgs.to_messages():
        print(f"   [{m.type}] {m.content[:60]}")
    print("schema:", Route.model_fields.keys(), GradeDocuments.model_fields.keys())


def t_tools():
    from agent.tools import TOOLS, search_rental_knowledge, search_with_sources_tool
    print("工具:", [t.name for t in TOOLS], search_with_sources_tool.name)
    print("工具参数:", search_rental_knowledge.args)


def t_memory_backend_import():
    import os
    os.environ["VECTOR_BACKEND"] = "memory"
    import importlib
    import config
    importlib.reload(config)
    from ragbase.store import get_backend
    b = get_backend()
    print("后端切换成功:", b.name)


def t_examples():
    for mod, arg in [
        ("examples/memory_demo.py", "runtime"),
        ("examples/memory_demo.py", "store"),
        ("examples/memory_demo.py", "travel"),
        ("examples/patterns.py", "parallel"),
        ("examples/patterns.py", "subgraph"),
    ]:
        r = subprocess.run([PY, f"{HERE}\\{mod}", arg],
                           capture_output=True, text=True, encoding="utf-8", errors="ignore")
        print(f"--- {mod} {arg} -> rc={r.returncode}")
        print((r.stdout or "")[:400])
        if r.returncode != 0:
            print((r.stderr or "")[-800:])
            raise RuntimeError(f"{mod} {arg} 失败")


if __name__ == "__main__":
    results = [
        run("1. 文档加载 + 清洗", t_ingest),
        run("2. 分块策略（不含需模型的）", t_chunkers),
        run("3. 递归分块 TokenTextSplitter", t_recursive_token),
        run("4. 主图导入与节点", t_graph_import),
        run("5. 提示词模板 + 少样本", t_prompts),
        run("6. 工具定义", t_tools),
        run("7. 向量后端切换", t_memory_backend_import),
        run("8. 离线示例脚本", t_examples),
    ]
    print("\n" + "=" * 60)
    print(f"通过 {sum(results)}/{len(results)}")
