"""
命令行入口：建索引 / 单次提问 / 多轮对话

流式输出对应 LangGraph/test26.py（stream_mode="messages"）
状态更新打印对应 LangGraph/test24.py（stream_mode="updates"）
"""
import argparse
import sys

from langchain_core.messages import HumanMessage
from langgraph.types import Command

import config
from agent.graph import graph, new_config, snapshot
from ragbase import store
from ragbase.chunkers import STRATEGIES, build_chunks


def cmd_build(args):
    chunks = build_chunks(args.strategy, **({"chunk_size": args.size, "chunk_overlap": args.overlap}
                                            if args.strategy == "fixed" else {}))
    if not chunks:
        print("分块结果为空，请检查 DOCUMENT_PATH / DOC_SOURCE_FILTER")
        return
    n = store.build_index(chunks)
    print(f"\n索引构建完成：{n} 条向量（策略={args.strategy}）")


def cmd_stats(args):
    print(f"知识库目录    : {config.DOCUMENT_PATH}")
    print(f"来源过滤      : {config.DOC_SOURCE_FILTER or '(全部)'}")
    print(f"分块策略      : {config.CHUNK_STRATEGY}  可选：{list(STRATEGIES)}")
    print(f"向量后端      : {config.VECTOR_BACKEND}")
    print(f"聊天模型      : {config.CHAT_MODEL} @ {config.DEEPSEEK_BASE_URL}")


def _print_sources(sources):
    if not sources:
        return
    print("\n--- 引用来源 ---")
    for s in sources:
        print(f"[{s['rank']}] score={s['score']}  {s['source']}")
        print(f"    {s['text'][:120]}")


def _invoke_with_review(payload, cfg):
    """支持 interrupt 人工审核的一次调用"""
    result = graph.invoke(payload, config=cfg)
    while result.get("__interrupt__"):
        info = result["__interrupt__"][0].value
        print(f"\n=== 需要人工审核 ===\n{info.get('instruction','')}\n")
        print(info.get("content", ""))
        choice = input("\n直接回车=批准；或输入改写后的内容：").strip()
        if choice:
            resume = {"approved": "false", "content": choice}
        else:
            resume = {"approved": "true"}
        result = graph.invoke(Command(resume=resume), config=cfg)
    return result


def cmd_ask(args):
    cfg = new_config(args.thread, args.user)
    result = _invoke_with_review({"messages": [HumanMessage(content=args.question)]}, cfg)
    print("\n=== 回答 ===")
    print(result["messages"][-1].content)
    _print_sources(result.get("sources", []))
    info = snapshot(result)
    print(f"\n[元信息] 路由={info['route']} 重写={info['rewrite_count']} "
          f"LLM调用={info['llm_calls']}")


def cmd_chat(args):
    cfg = new_config(args.thread, args.user)
    print("租房客服已就绪。输入 q 退出。\n")
    while True:
        question = input("你：").strip()
        if question.lower() in ("q", "quit", "退出"):
            print("已退出。")
            break
        if not question:
            continue

        print("客服：", end="", flush=True)
        payload = {"messages": [HumanMessage(content=question)]}

        if config.HUMAN_REVIEW:
            result = _invoke_with_review(payload, cfg)
            print(result["messages"][-1].content)
        else:
            # 流式输出 LLM token
            for chunk in graph.stream(payload, config=cfg, stream_mode="messages"):
                token, metadata = chunk
                if getattr(token, "content", None):
                    print(token.content, end="", flush=True)
            print()
            state = graph.get_state(cfg)
            _print_sources(state.values.get("sources", []))


def main():
    parser = argparse.ArgumentParser(description="租房平台智能客服 Agent")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("build", help="构建向量索引")
    p.add_argument("--strategy", default=config.CHUNK_STRATEGY, choices=list(STRATEGIES))
    p.add_argument("--size", type=int, default=500)
    p.add_argument("--overlap", type=int, default=50)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("ask", help="单次提问")
    p.add_argument("question")
    p.add_argument("--thread", default="cli")
    p.add_argument("--user", default="anonymous")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="多轮对话（流式）")
    p.add_argument("--thread", default="cli")
    p.add_argument("--user", default="anonymous")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("stats", help="查看当前配置")
    p.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
