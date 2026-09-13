"""
检索阈值校准工具

打印每个 query 的 top-k 相似度得分，以及经过 filter_chunks + compress_context 之后
实际进入提示词的上下文。换知识库 / 换嵌入模型后用它重新校准 config.MIN_SCORE。

    python probe_scores.py
"""
import config
from ragbase import retrieval, store

QUERIES = [
    "押金怎么退",
    "介绍一下这个项目",
    "项目用了哪些技术栈",
    "房源搜索是怎么实现的",
    "今天天气怎么样",
]

if __name__ == "__main__":
    print(f"MIN_SCORE = {config.MIN_SCORE}\n")

    for q in QUERIES:
        hits = store.search(q, top_k=config.TOP_K)
        print(f"=== {q}")
        for h in hits:
            mark = "保留" if h["score"] >= config.MIN_SCORE else "丢弃"
            print(f"  [{mark}] score={h['score']:.4f}  {h['text'][:60]}")

        context, sources = retrieval.get_context(q)
        print(f"  -> 保留 {len(sources)} 条，上下文 {len(context)} 字符")
        if not context:
            print("  -> 上下文为空，模型将回答「文档中不存在」")
        print()
