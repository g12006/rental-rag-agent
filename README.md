# 租房平台智能客服 Agent

把两份学习资料融合成一个可运行的综合项目：

- **`ragbasic-master`**（LlamaIndex 系 RAG 底座）→ 数据层 + 服务层
- **`LangChain学习资料`**（LangChain 组件 + LangGraph 编排）→ 组件层 + 编排层

> 约束：**只使用这两份资料里出现过的知识点**。凡是资料里没有的东西（ReAct 预构建 Agent、Rerank、HyDE、多路召回等）一律不引入。

---

## 1. 快速开始

```bash
conda activate RAG                       # 你的 RAG 环境（Python 3.12）
cd rental_rag_agent
pip install langchain-text-splitters milvus-lite     # 环境里缺的两个包

cp .env.example .env                     # 填入 DEEPSEEK_API_KEY

python cli.py stats                      # 看当前配置
python cli.py build                      # 构建向量索引（首次会加载 BGE-M3，约 1-2 分钟）
python cli.py ask "介绍一下这个项目"       # 单次提问
python cli.py chat                       # 多轮对话（流式输出）

python server.py                         # 启动 FastAPI，默认 127.0.0.1:8080
```

只跑通数据层（不需要 API Key）：

```bash
python ragbase/ingest.py                 # 加载 + 清洗
python ragbase/chunkers.py               # 对比 7 种分块策略
python ragbase/retrieval.py "押金怎么退"   # 看增强后的提示词
python examples/patterns.py parallel     # 离线工作流示例
python examples/memory_demo.py travel    # 时间旅行示例
```

---

## 2. 目录结构

```
rental_rag_agent/
├── config.py                 全局配置（对齐 ragbasic/config.py 风格）
├── .env.example
├── requirements.txt
│
├── ragbase/                  【数据层】RAG 底座 —— 全部来自 ragbasic-master
│   ├── ingest.py             SimpleDirectoryReader 加载 + 正则清洗
│   ├── chunkers.py           7 种分块策略 + LangChain 分割器
│   ├── store.py              BGE-M3 嵌入 + Milvus Lite / InMemoryVectorStore
│   └── retrieval.py          检索 → 过滤 → 上下文压缩 → 提示词增强
│
├── agent/                    【组件层 + 编排层】—— 全部来自 LangChain/LangGraph 资料
│   ├── model.py              ChatOpenAI / init_chat_model 可配置模型
│   ├── prompts.py            ChatPromptTemplate + 少样本 + 结构化输出 schema
│   ├── tools.py              @tool / StructuredTool 封装检索能力
│   └── graph.py              LangGraph 主图
│
├── server.py                 【服务层】FastAPI（对齐 ragbasic/medical/backend/api.py）
├── cli.py                    命令行：build / ask / chat / stats
└── examples/
    ├── patterns.py           并行 / 协调者-工作者(Send) / 子图
    └── memory_demo.py        Runtime 上下文 / 时间旅行 / store 跨会话记忆
```

---

## 3. 主图结构

```
                          ┌──────────── START ────────────┐
                          │                               │
                          ▼                               │
                   sync_profile  ← store 跨会话记忆        │
                          │                               │
                          ▼                               │
                   route_intent  ← with_structured_output  │
                    │      │      │                       │
       small_talk ──┘      │      └── human_handoff       │
          │                │                  │           │
     chat_direct           │              handoff         │
          │                │                  │           │
          ▼                ▼                  ▼           │
         END      generate_query_or_respond ← bind_tools   │
                          │                               │
              tools_condition（LangGraph 内置）            │
                    │              │                      │
                 tools          __end__                   │
                    │              │                      │
                    ▼              ▼                      │
               retrieve        END                        │
              (ToolNode)                                  │
                    │                                     │
                    ▼                                     │
            collect_sources                               │
                    │                                     │
            grade_documents ── yes ──> generate_answer ───┘
                    │                        │
                    no                       ├─> human_review（interrupt，可选）
                    │                        ├─> summarize_conversation（消息过多时）
                    ▼                        └─> END
            rewrite_question ──> 回到 generate_query_or_respond
```

---

## 4. 知识点映射表

### 4.1 数据层 —— `ragbasic-master`

| # | 知识点 | 出处 | 落地位置 |
|---|---|---|---|
| 1 | `SimpleDirectoryReader` 多格式加载 | `util.py:12`、`dataParse.py:9` | `ragbase/ingest.py` |
| 2 | 正则清洗（零宽字符 / NFKC / 控制字符 / 空行压缩 / PPT 噪声） | `dataClean.py:44-75` | `ragbase/ingest.py` |
| 3 | 固定大小分块（`step = size - overlap` 滑动） | `fixedSizeChunk.py:8-40` | `ragbase/chunkers.py` |
| 4 | 句子分块 + 正则断句回退 | `sentenceChunk.py:23-50` | 同上 |
| 5 | 递归分块 `TokenTextSplitter` + 分隔符优先级列表 | `recursiveChunk.py:19-32` | 同上 |
| 6 | 语义分块 `SemanticSplitterNodeParser` | `semanticChunk.py:45` | 同上 |
| 7 | 结构分块：pdf / ppt / csv / txt 分流 + csv 表头拼接 | `structureChunk.py:122-135` | 同上 |
| 8 | LLM 分块 + `===CHUNK===` 分隔符解析 | `llmChunk.py:10,23-29` | 同上 |
| 9 | `HuggingFaceEmbedding`(BGE-M3) + `similarity` + `SimilarityMode` | `similarity.py`、`semanticChunk.py:15` | `ragbase/store.py` |
| 10 | Milvus Lite：建库 / 建集合 / 插入 / 检索 / 按 id 取 | `milvus.py:19-65` | `ragbase/store.py` |
| 11 | 相关性过滤 `filter_chunks` | `ragPromptEnhancer.py:100` | `ragbase/retrieval.py` |
| 12 | 上下文压缩 `compress_context` | `ragPromptEnhancer.py:72` | 同上 |
| 13 | 提示词模板（角色/任务/上下文/问题/输出要求） | `ragPromptEnhancer.py:14-24,114` | 同上 |
| 14 | 批量嵌入 + 分批入库 + 索引重建 | `medical/backend/main.py:130-188` | `ragbase/store.py` |

### 4.2 组件层 —— `LangChain/` 课堂代码

| # | 知识点 | 出处 | 落地位置 |
|---|---|---|---|
| 15 | `ChatOpenAI`（model / api_key / base_url / temperature / max_tokens） | `test2.py`、`test7.py` | `agent/model.py` |
| 16 | `init_chat_model` + `configurable_fields`（可配置模型） | `test3.py` | `agent/model.py` |
| 17 | 消息类型 `SystemMessage` / `HumanMessage` / `AIMessage` / `ToolMessage` | `test1.py`、`test7.py` | 全局 |
| 18 | `bind_tools` 工具绑定 | `test7.py` | `agent/graph.py` |
| 19 | `@tool` + `Annotated` 参数注解 | `test5.py` | `agent/tools.py` |
| 20 | `StructuredTool.from_function` + `args_schema` + `content_and_artifact` | `test6.py` | `agent/tools.py` |
| 21 | `with_structured_output`（Pydantic schema） | `test9/10/11.py` | `agent/prompts.py`、`graph.py` |
| 22 | `ChatPromptTemplate` + `MessagesPlaceholder` | `test19.py` | `agent/prompts.py` |
| 23 | 少样本 `FewShotChatMessagePromptTemplate` | `test21.py` | `agent/prompts.py` |
| 24 | 文本分割器 `RecursiveCharacterTextSplitter` | `test28.py`、`test30.py` | `ragbase/chunkers.py` |
| 25 | `OpenAIEmbeddings` + `InMemoryVectorStore`（备用向量后端） | `test30.py`、`test31.py` | `ragbase/store.py` |
| 26 | LCEL `\|` 组装链 + `StrOutputParser` | `test1.py`、`test15.py` | `agent/graph.py`（`answer_prompt \| model`） |
| 27 | `trim_messages` / `filter_messages` 消息管理 | `test17.py`、`test18.py` | `agent/graph.py` |

### 4.3 编排层 —— `LangGraph/` 课堂代码

| # | 知识点 | 出处 | 落地位置 |
|---|---|---|---|
| 28 | `StateGraph` + `TypedDict` State + `Annotated[list, operator.add]` | `test1.py` | `agent/graph.py` |
| 29 | `add_node` / `add_edge` / `add_sequence` / `START` / `END` / `compile` | `test1.py`、`test6.py` | 主图 |
| 30 | `add_conditional_edges`（path + path_map） | `test1.py` | 路由 / 评分 / 后处理 |
| 31 | `MessagesState` + `ToolNode` + `tools_condition` | `test3.py` | 检索分支 |
| 32 | Agentic RAG 四节点：决策 → 检索 → 评分 → 生成 | `test3.py` | **主图骨架** |
| 33 | `with_structured_output(GradeDocuments)` 相关性评分 | `test3.py` | `grade_documents` |
| 34 | 问题重写 `rewrite_question` 回环 | `test3.py` | `rewrite_question` |
| 35 | 路由模式（结构化决策 + 条件边） | `test9.py` | `route_intent` |
| 36 | `InMemorySaver` + `thread_id` 线程级持久化 | `test11.py`、`test12.py` | `builder.compile` |
| 37 | `InMemoryStore` 跨会话记忆：namespace + put/search + `BaseStore` 注入 | `test13.py`、`test14.py` | `sync_profile` |
| 38 | `RemoveMessage` 消息裁剪 + 历史摘要 | `test15.py` | `summarize_conversation` |
| 39 | `interrupt` + `Command(resume)` 人工审核 | `test16-18.py` | `human_review` |
| 40 | 流式 `stream_mode="messages"` | `test26.py` | `cli.py` |
| 41 | 并行分支（多 START 边 + 汇总） | `test8.py` | `examples/patterns.py` |
| 42 | 协调者-工作者 `Send` 动态分发 | `test10.py` | `examples/patterns.py` |
| 43 | 子图（节点内调用 / 作为节点） | `test29.py`、`test30.py` | `examples/patterns.py` |
| 44 | 时间旅行 `get_state_history` / `update_state` / 重放 | `test11.py`、`test21.py` | `examples/memory_demo.py` |
| 45 | `context_schema` + `Runtime` 静态运行时上下文 | `test22.py`、`test23.py` | `examples/memory_demo.py` |

### 4.4 服务层 —— `ragbasic/medical/backend`

| # | 知识点 | 出处 | 落地位置 |
|---|---|---|---|
| 46 | FastAPI + `CORSMiddleware` + Pydantic 请求体 + `HTTPException` | `api.py` | `server.py` |
| 47 | `uvicorn.run` | `api.py` | `server.py` |
| 48 | 线程锁 + 重建索引 + 检索 + 生成 + sources 打分返回 | `main.py:227-261` | `server.py` |

---

## 5. 数据量实测

知识库：`脚手架级微服务租房平台Q&A.md`（54,167 bytes / 25,322 字符 / 864 行）
清洗后：**24,709 字符 / 545 行**，共 326 句。

| 分块策略 | 块数 | 平均每块 | 向量体积（1024 维 fp32） |
|---|---|---|---|
| `fixed`（500/50） | 55 | 498 字符 | ~225 KB |
| `sentence`（max_sentences=5）**默认** | 66 | 360 字符 | ~270 KB |
| `sentence`（max_sentences=1） | 326 | 76 字符 | ~1.3 MB |
| `recursive`（TokenTextSplitter） | 21 | — | ~86 KB |
| `structure`（合并后） | 18 | 1,371 字符 | ~74 KB |
| `langchain`（RecursiveCharacter ~1000） | 27 | 930 字符 | ~110 KB |
| `semantic`（SemanticSplitterNodeParser） | **6** | 4,118 字符（min 1,590 / max 8,476） | ~25 KB |
| `llm`（切窗口后） | **76** | 323 字符 | ~310 KB |

两个需模型的策略实测补充：

- **`semantic` 耗时 471 秒**（CPU，326 句两两算相似度），且块偏大（平均 4,118 字符，
  远超 `MAX_CHARS=1500`，检索时会被 `compress_context` 截断）。
  语义分块质量好但**不适合频繁重建**，本项目默认仍用 `sentence`。
- **`llm` 耗时 68 秒 / 5 次调用**，覆盖率 **100%**（24,615 / 24,709）。
  修 bug 前只有 72%（见第 7 节第 4 行）。

资源：BGE-M3 fp32 约 **2.3 GB** 内存 + torch 运行时约 0.5 GB。首次建库 = 模型加载 20-40s + 嵌入 10-30s（66 块）。

---

## 6. 阈值校准：`MIN_SCORE` 是怎么定成 0.50 的

`ragbasic` 用的是 `max_distance=0.5`，但换知识库 / 换嵌入模型后这个数不能照抄。
在本文档（BGE-M3 + 1024 维 + sentence 分块）上实测的 top-5 得分区间：

| 查询 | top1 得分 | 是否相关 |
|---|---|---|
| `项目用了哪些技术栈` | 0.65 | 相关 |
| `介绍一下这个项目` | 0.628 | 相关 |
| `房源搜索是怎么实现的` | 0.595 | 相关 |
| `押金怎么退` | 0.454 | **文档里没有**（应回答"文档中不存在"） |
| `今天天气怎么样` | 0.423 | 无关 |

相关与无关的**分水岭在 0.5 附近**，因此 `MIN_SCORE=0.50`。
低于该阈值的片段会被 `filter_chunks` 丢掉，上下文为空时模型回答"文档中不存在"——这正是想要的行为。

换知识库后重新校准：

```bash
python probe_scores.py      # 改一下里面的 QUERIES 列表
```

---

## 7. 对原始资料的 4 处修正 / 扩展

| 位置 | 原资料 | 本项目 | 原因 |
|---|---|---|---|
| `ragPromptEnhancer.py:104` | `if hit.get("distance") > max_distance: continue` | 统一成 `score`，按 `score < MIN_SCORE` 过滤 | Milvus COSINE 度量下 distance **越大越相似**，原写法方向反了，会把最相关的结果过滤掉 |
| `structureChunk.py` | md 文档会掉到最后的 `return [d]`，整篇不分块 | 复用同一个 `_SUBSECTION_RE` 让 md 也能按项目符号切分，并合并过碎片段 | 租房 Q&A 通篇是 `- **回答1：**` 列表，不分块 = 24,709 字一个向量，检索必废；不合并 = 244 个 100 字碎块，语义不完整 |
| `llmChunk.py` | 直接使用 `util.chat` | `chat_fn` 参数注入，默认走 `agent.model.chat_text` | 避免数据层反向依赖 Agent 层造成循环导入 |
| `llmChunk.py` | 整篇文档一次性喂给 LLM | 先按 `_SUBSECTION_RE` + `_merge_parts` 切成 ≤6,000 字符的窗口，逐窗口调用 | 24,709 字符一次性进去，模型要吐回等长文本，会被 `max_tokens` 截断 —— **实测只返回 72%，知识库直接少掉近三成内容**；切窗口后覆盖率 100% |

---

## 8. 双向量后端

`config.VECTOR_BACKEND` 切换，两套组合都是"纯血"、互不混搭：

| 后端 | 嵌入 | 向量库 | 来源 | 依赖 |
|---|---|---|---|---|
| `milvus`（默认） | BGE-M3（1024 维） | Milvus Lite 本地文件 | `ragbasic-master` | 本地模型，无外网 |
| `memory` | `OpenAIEmbeddings` | `InMemoryVectorStore` | `LangChain/test30、31` | 需要 `OPENAI_API_KEY` |

---

## 9. 配置项

| 变量 | 默认 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | 必填 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | |
| `EMBED_MODEL_PATH` | `D:\models\BAAI\bge-m3` | 本地 BGE-M3 |
| `DOCUMENT_PATH` | LangChain 资料的 markdown 目录 | 知识库目录 |
| `DOC_SOURCE_FILTER` | `租房` | 只保留路径含该关键词的文档 |
| `VECTOR_BACKEND` | `milvus` | `milvus` / `memory` |
| `CHUNK_STRATEGY` | `sentence` | 7 种可选 |
| `TOP_K` / `MIN_SCORE` / `MAX_CHARS` | `5` / `0.50` / `1500` | 检索与上下文压缩（`MIN_SCORE` 校准见第 6 节） |
| `MAX_REWRITE` | `2` | 问题重写上限，防死循环 |
| `MAX_MESSAGES` | `10` | 触发历史摘要的消息条数 |
| `HUMAN_REVIEW` | `false` | 开启后每次生成答案都进人工审核 |
| `MILVUS_DB_DIR` | `C:\ragdata` | **必须纯 ASCII**，见下面排错 |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8080` | |

---

## 10. 排错：Milvus Lite 在中文路径下跨进程检索失败

### 现象
`cli.py build` 成功，但另起进程检索时报：

```
MilvusException: Error in __cdecl faiss::FileIOWriter::FileIOWriter(const char *):
could not open ...\collections\rental_qa\partitions\_default\indexes\
data_000001_000066.vector.autoindex.idx for writing: No such file or directory
```

### 根因（已实测确认）
Milvus Lite 底层用 faiss 写索引文件，faiss 在 Windows 上走 C 的 `fopen`，
**无法处理非 ASCII 路径**。本项目位于 `C:\Users\龚祎\...`，用户名含中文，因此索引写不进去。

| 库路径 | 同进程建库+检索 | 跨进程检索 |
|---|---|---|
| ASCII（`C:\Windows\Temp\...`） | 成功 | **成功** |
| 含中文（`C:\Users\龚祎\...`） | 成功 | **失败** |

### 处理
`config.MILVUS_DB_DIR` 默认指向 `C:\ragdata`（纯 ASCII），并在检测到非 ASCII 时自动兜底 + 告警。
需要换目录就改 `.env` 里的 `MILVUS_DB_DIR`，**保证路径全是英文**。

> 顺带一提：`ragbasic-master` 把库建在项目目录下（`config.base_path / f"{DB_NAME}.db"`），
> 在中文用户名的 Windows 机器上会踩同一个坑。

---

## 11. 已知限制

- `langchain_classic` 未安装 → `create_retriever_tool`（`LangGraph/test3.py`）不可用，改用 `StructuredTool.from_function` 自行封装，效果等价。
- `init_embeddings` 在当前 `langchain-core 1.6.2` 不存在 → store 的语义搜索（`test13.py`）不可用，退化为按 namespace + limit 精确取，这也是 `test14.py` 的主用法。
- Redis / Pinecone / Postgres 三套向量库在资料里有（`test26/32/33/34`、`test11/13`），但依赖 `192.168.100.238` 等远程机器，本机不可达，未接入。
- `semantic` / `llm` 两种分块需要加载模型或调用 LLM，未在自检中默认执行。

---

## 12. 端到端实测记录（DeepSeek + BGE-M3 + Milvus Lite）

```
python selftest.py          # 8/8 通过
python cli.py ask "介绍一下这个项目"
python server.py            # 127.0.0.1:8080 ，前端在 /
```

### 12.1 CLI 单次问答

```
路线：sync_profile -> route_intent -> generate_query_or_respond -> retrieve
      -> collect_sources -> grade_documents -> generate_answer
路由=rental_qa  重写=0  LLM调用=5   命中 5 条来源（score 0.65 ~ 0.73）  耗时约 33s
```

5 次 LLM 调用分别是：偏好提取 1 + 意图路由 1 + 工具调用决策 1 + 相关性评分 1 + 生成答案 1。

### 12.2 HTTP 接口（三种意图各一发）

| 问题 | route | 来源数 | 说明 |
|------|-------|--------|------|
| 项目用了哪些技术栈？ | `rental_qa` | 5 | 正常走 RAG，答案带引用编号 |
| 你好呀 | `small_talk` | 0 | 直接闲聊，不检索，LLM 调用降到 3 |
| 我在西安，预算3000以内，想租两室一厅 | `rental_qa` | 0 | 偏好写入 store 成功（`city=西安；budget=3000以内；room_type=两室一厅`）；知识库无房源数据 → 触发问题重写 → 达到 `MAX_REWRITE` 上限后如实回答"知识库里没有" |

第三条最能说明编排的价值：**检索不到 → 改写问题 → 再检索 → 仍无结果 → 不编造**。

### 12.3 人工审核链路（interrupt → Command(resume)）

启动：`HUMAN_REVIEW=true python server.py`

| 步骤 | 请求 | 实测结果 |
|------|------|----------|
| 提问 | `POST /api/ask` | 返回 `interrupted=true`、`answer` 为空、`review.instruction="请审核客服回复，可批准或改写"` + 待审核原文 |
| 改写提交 | `POST /api/resume {approved:false, content:"【人工改写版】..."}` | `interrupted=false`，最终 `answer` **等于**改写内容 |
| 直接批准 | `POST /api/resume {approved:true, content:""}` | `interrupted=false`，最终 `answer` **原样保留** |

要点：`interrupt()` 把图挂起在 `human_review` 节点，state 由 checkpointer 保住；
`Command(resume=...)` 的返回值就是 `interrupt()` 的返回值，节点据此决定 goto。
thread_id 必须前后一致，否则找不到挂起的 checkpoint。

### 12.4 前端页面

浏览器打开 `http://127.0.0.1:8080/` 即可，支持：多轮对话、显示引用来源与相似度、
底部展示路由/重写次数/LLM 调用次数、一键重建索引（切换分块策略）、
以及 `HUMAN_REVIEW=true` 时的"审核 / 改写回复"面板（文本框 + 批准 / 提交改写两个按钮）。

---

## 13. 两处实现取舍说明

1. **`grade_documents` 用 `Command` 而不是 `add_conditional_edges`**
   课堂代码 `LangGraph/test3.py` 里评分节点返回节点名、由条件边跳转。
   本项目改成节点内 `return Command(update={...}, goto=...)`：同样是路由，
   但还能把评分结果和 LLM 调用次数写回 state（条件边函数拿不到 state 写权限）。

2. **统一 `snapshot()` 出口**
   LangGraph 里没被写过的 state channel 读出来是 `None` 而不是"键不存在"，
   直接 `result.get("rewrite_count", 0)` 会拿到 `None`。
   所以 `agent/graph.py` 里加了 `snapshot()` 统一兜底，CLI / API 共用它。

---

## 14. 配套讲稿

按「💡 概念 → 📝 代码 → 📊 输出 → ⚠️ 注意 → ✅ 掌握标准 → 🎯 面试追问」6 要素逐讲拆解：

| 讲 | 主题 | 文件 |
|---|---|---|
| 1 | 数据层①：文档加载与清洗 | [docs/lecture-01-ingest.md](docs/lecture-01-ingest.md) |
| 2 | 数据层②：7 种分块策略 | 待续 |
| 3-11 | 检索 / 组件层 / 编排层 / 服务层 / 面试串讲 | 待续 |

第 1 讲配套可运行示例：`python examples/clean_demo.py`（不需要 API Key、不需要向量库）。
