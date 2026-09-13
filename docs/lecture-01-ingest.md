# 第 1 讲 · 数据层①：文档加载与清洗

> 对应文件：`ragbasic/ingest.py`
> 资料出处：`ragbasic-master/util.py`（parse_all_formats）+ `ragbasic-master/dataClean.py`（clean_text / clean_ppt / clean_doc）

**保姆式说明**：本讲所有代码都不需要 API Key、不需要向量库，直接 `python examples/clean_demo.py` 就能看到结果。

---

## 模块 1.1 · 文档加载：`SimpleDirectoryReader`

### 💡 概念解释

`SimpleDirectoryReader` 是 LlamaIndex 的**统一入口加载器**：你给它一个目录，它按文件后缀自动分派对应的解析器（pdf / docx / pptx / csv / md / txt…），返回一个 `List[Document]`。

`Document` 是 LlamaIndex 的最小数据单元，只有两个东西：

```python
Document(text="正文内容", metadata={"file_path": "...", "page_label": "3", ...})
```

C++ 视角：`Document` 就是一个 `struct { std::string text; std::unordered_map<std::string, Any> metadata; }`，
`SimpleDirectoryReader` 是一个按扩展名查函数表的工厂。

**为什么要这一层**：RAG 的第一公里是"把各种格式的脏数据变成统一结构"。没有它，你就要自己写 6 个解析器 + 6 套异常处理。

### 📝 完整可运行代码

```python
"""跑：python -c "from ragbase.ingest import parse_all_formats; ..."（见下方命令）"""
from ragbase.ingest import parse_all_formats

docs = parse_all_formats(r"C:\Users\龚祎\Desktop\学习资料\LangChain学习资料\课堂代码\PythonProject\Docs\markdown")

print(f"加载了 {len(docs)} 篇文档")
for d in docs:
    print(f"  文件：{d.metadata.get('file_path')}")
    print(f"  字符数：{len(d.text)}")
    print(f"  metadata keys：{list(d.metadata.keys())}")
```

实际在项目里直接跑：

```bash
cd rental_rag_agent
python ragbase/ingest.py
```

### 📊 输出结果

```
清洗后共 1 篇文档

第 1 篇 | C:\Users\龚祎\Desktop\学习资料\LangChain学习资料\课堂代码\PythonProject\Docs\markdown
字符数：24709
# 通用问题 ### **为什么做这个项目?** - **回答1:(出于兴趣爱好开发)** 大学期间,我和同学在外合租过一段时间...
```

（原始文件 25,322 字符 → 清洗后 24,709 字符，减少 613 字符，全部是不可见字符和多余空行）

### ⚠️ 注意事项

1. **`load_data()` 是饿汉式全量加载**——10 万文件会直接吃满内存。大批量的要分批调或用 `num_files_limit` 限制。
2. **`metadata` 的字段因解析器而异**：PDF 有 `page_label`，PPTX 有 `text_sections` / `title`，纯文本只有 `file_path`。后面 `structure` 分块策略就是靠 `metadata` 里的这些字段判断"这是 PDF 还是 PPT"的。
3. **路径里有中文不影响 Python 层**（`open()` 没问题），但会影响底层 C 扩展落盘（见 README 第 10 节 Milvus 的坑）。

### ✅ 掌握标准

- [ ] 能说出 `Document` 的两个字段分别是什么
- [ ] 能独立写一个"加载目录 → 打印每篇字符数"的脚本
- [ ] 能解释为什么不同格式的 `metadata` 字段不同

### 🎯 面试追问（大厂深度）

**Q1：`SimpleDirectoryReader` 底层是怎么做到"按后缀自动选解析器"的？**

> 源码级：它内部维护 `self.file_reader` 或默认的 `default_file_reader_cls` 字典，key 是扩展名（含点），value 是 reader 类。
> `load_data()` 遍历目录下所有文件 → 取 `Path(f).suffix` → 查表 → 找不到就跳过（或者按 `required_exts` / `exclude` 过滤）。
> 追问：**这个设计叫什么模式？** → 策略模式 + 工厂，和 C++ 里的 `std::unordered_map<std::string, std::unique_ptr<Parser>>` 一样。

**Q2：它为什么用 `fsspec` 而不是直接 `open()`？**

> 因为要支持**统一资源标识符**：`s3://`、`hdfs://`、`https://` 都能塞进同一个接口。
> fsspec 提供类文件对象的抽象，reader 只管 `f.read()`，不关心底层是本地磁盘还是对象存储。
> 追问：**这对 RAG 工程意味着什么？** → 知识库放 OSS/S3 上时，代码不用改一行。

**Q3：`Document` 和 `Node` 有什么区别？为什么 LlamaIndex 要分两层？**

> `Document` = 一整个原始文档，没有结构关系。
> `Node` = 从 Document 切出来的块，**带 `relationships` 指针**（`SOURCE` 指向原 Document，`PREVIOUS`/`NEXT` 指向相邻块，`PARENT` 指向父块）。
> 追问：**这个指针有什么用？** → 检索命中一个 Node 后，能顺着 `PREVIOUS/NEXT` 把上下文窗口扩大（ sentence window retrieval ），顺着 `SOURCE` 回溯原文出处。**这是 RAG 能给出"引用来源"的基础**。本项目 `retrieval.py` 返回的 `sources` 就是靠 chunk 的 metadata 做到的。

---

## 模块 1.2 · 文本清洗 5 步：`clean_text`

### 💡 概念解释

从 Word / PDF / 网页复制出来的文本里藏着大量**肉眼看不见但模型看得见**的字符。它们会：

- 让 BGE 的分词器把「押金规则」切成奇怪的 token
- 让同一句话的 query 和 doc 向量不一致 → **检索不到**
- 让 `len(text)` 虚高 → 分块位置错乱

`clean_text` 用 5 步把它们清掉：

| 步骤 | 做什么 | 正则 / 函数 | 为什么 |
|---|---|---|---|
| ① | 去零宽字符 | `_ZERO_WIDTH = [\ufeff\u200b\u200c\u200d]` | BOM、零宽空格、零宽连接符 |
| ② | Unicode 归一 | `unicodedata.normalize("NFKC", text)` | 全角 ＡＢＣ１２３ → 半角 ABC123 |
| ③ | 去控制字符 | `_CONTROL_CHARS = [\x00-\x08\x0b\x0c\x0e-\x1f\x7f]` | 响铃符、换页符等 |
| ④ | 换行统一 | `.replace("\r\n","\n").replace("\r","\n")` | Windows/Mac/Unix 三套换行 |
| ⑤ | 压缩空行 | 去首尾空格 + 丢空行 | 连续空行没有语义 |

### 📝 完整可运行代码

`examples/clean_demo.py`（已验证可跑）：

```python
from ragbase.ingest import _CONTROL_CHARS, _ZERO_WIDTH, clean_text

DIRTY = (
    "\ufeff### **押金\ufeff规则**\u200b\r\n\r\n\r\n"
    "ＡＢＣ１２３ 元\x07\n\n\n"
    "第二条：退租需提前 30 天\n"
)

t = DIRTY
t = _ZERO_WIDTH.sub("", t)                            # ①
t = unicodedata.normalize("NFKC", t)                  # ②
t = _CONTROL_CHARS.sub("", t)                         # ③
t = t.replace("\r\n", "\n").replace("\r", "\n")       # ④
lines = [line.strip() for line in t.split("\n") if line.strip()]
t = "\n".join(lines)                                  # ⑤
```

跑：

```bash
python examples/clean_demo.py
```

### 📊 输出结果

```
原始             长度= 48  '\ufeff### **押金\ufeff规则**\u200b\r\n\r\n\r\nＡＢＣ１２３ 元\x07\n\n\n第二条：退租需提前 30 天\n'

① 零宽字符         长度= 45  '### **押金规则**\r\n\r\n\r\nＡＢＣ１２３ 元\x07\n\n\n第二条：退租需提前 30 天\n'
② NFKC 归一      长度= 45  '### **押金规则**\r\n\r\n\r\nABC123 元\x07\n\n\n第二条:退租需提前 30 天\n'
③ 控制字符         长度= 44  '### **押金规则**\r\n\r\n\r\nABC123 元\n\n\n第二条:退租需提前 30 天\n'
④ 换行统一         长度= 41  '### **押金规则**\n\n\nABC123 元\n\n\n第二条:退租需提前 30 天\n'
⑤ 压缩空行         长度= 36  '### **押金规则**\nABC123 元\n第二条:退租需提前 30 天'

clean_text     长度= 36  '### **押金规则**\nABC123 元\n第二条:退租需提前 30 天'

长度：48 -> 36（少了 12 个字符）
```

注意 ② 那一行：`第二条：` 变成了 `第二条:`——全角冒号被转成了半角。

### ⚠️ 注意事项

1. **顺序不能反**：必须先删零宽字符再 NFKC。因为 `\u200b`(ZWSP) 和 `\ufeff`(BOM) **在 Unicode 里没有兼容分解映射**，`normalize("NFKC", "\u200b")` 返回的还是 `\u200b`。先删后归一才干净。
2. **`NFKC` 是有损不可逆的**：它会把 `①` → `1`、`Ａ` → `A`、`：` → `:`、全角空格 → 半角空格。**所以 query 和 doc 必须做同样的处理**，否则一个全角一个半角，向量就对不上了。本项目 `retrieval.py` 检索时对 query 不做 NFKC（DeepSeek 传过来的 query 本来就是干净的），但如果你的 query 来自 OCR 或老旧系统，**必须也走一遍 `clean_text`**。
3. **控制字符正则为什么写成 `[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]` 而不是 `[\x00-\x1f]`？**
   因为要**保留 `\t`(0x09)、`\n`(0x0a)、`\r`(0x0d)** 这三个有排版意义的字符，只删 `\x0b`(垂直制表) 和 `\x0c`(换页)。这是源码级细节，面试官很爱问。
4. **`\n{3,}` 替换成 `\n\n` 而不是 `\n`**：保留段落边界。段落是语义单元，后面 `RecursiveCharacterTextSplitter` 的第一优先级分隔符就是 `\n\n`。全压成一个 `\n` 会让分块质量下降。
5. **PPT 是特殊分支**：`.pptx` 会额外走 `clean_ppt`，干掉 `Title:` 行、`---` 分割线、`[Speaker Notes]:` 前缀这些解析噪声。

### ✅ 掌握标准

- [ ] 能默写 5 步的顺序，并说出"为什么零宽要在 NFKC 前"
- [ ] 能解释控制字符正则里"跳过 09/0a/0d"的原因
- [ ] 看到 `ＡＢＣ１２３` 能反应出要用 NFKC
- [ ] 能在自己项目里复用这套清洗

### 🎯 面试追问（大厂深度）

**Q1：零宽字符到底是怎么让检索失效的？说到底层。**

> 三层影响：
> ① **分词层**：BGE-M3 用 BERT 式 WordPiece tokenizer。`押金\u200b规则` 会被切成 `押金` + `[UNK]`/`##▁` 之类的碎片，而 query `押金规则` 切出来是完整 token。token 序列不同 → embedding 不同。
> ② **向量层**：零宽字符占了 position embedding 的位置，稀释了真实语义的注意力权重。
> ③ **匹配层**：用户**永远打不出**零宽字符，所以 query 侧天然是干净的，doc 侧脏 → 两侧永远不在同一个向量空间。
> 追问：**那我在 query 侧也加零宽字符行不行？** → 不行，你不知道原文插在哪里、插了几个。正解是 doc 侧清干净。

**Q2：NFKC / NFC / NFD / NFKC 四者区别？为什么 RAG 选 NFKC？**

> - `NFD` 规范分解：`é` → `e` + `´`（变音符号拆出来）
> - `NFC` 规范合成：`e` + `´` → `é`
> - `NFKC` **兼容**合成：在规范合成基础上，还把"兼容字符"折叠成标准形式（全角→半角、① →1、㍿→株式会社、罗马数字→ASCII）
> - `NFKD` 兼容分解
>
> **选 NFKC 的原因**：RAG 要的是"同一种写法的文本有同一个向量"。兼容归一能把「ＡＢＣ」和「ABC」统一，召回率直接上去。
> 追问：**NFKC 有什么代价？** → 有损、不可逆（全角引号 `“` 会变成 `"`），如果业务依赖原文精确还原（比如合同），清洗后的文本不能回写数据库。

**Q3：为什么清洗后还要 `if new_doc.text` 过滤空文档？**

> ① 空字符串送进 embedding 模型，BGE 会返回**全零向量或报错**（取决于实现），入库后 COSINE 相似度是 0 或 NaN，污染检索。
> ② 更现实的场景：PDF 扫描件（纯图片）解析出来就是空文本，不过滤会有大量空 Document。

**Q4：清洗放在"解析后"还是"解析前"？**

> 必须**解析后**。因为零宽/控制字符是**解析产物**的一部分（PDF 解析器会把排版指令转成这些字符）；在原始二进制上做清洗会破坏文件结构。
> 唯一例外是**编码层**的 BOM，可以在读取时用 `encoding="utf-8-sig"` 直接吃掉——本项目 `_ZERO_WIDTH` 里保留 `\ufeff` 就是为了兜住没用 `utf-8-sig` 的路径。

---

## 模块 1.3 · 过滤与装配：`load_clean_documents`

### 💡 概念解释

把「加载 → 过滤 → 清洗」三步串成一条流水线，是数据层对外的唯一入口：

```python
def load_clean_documents(input_dir, source_filter: str = "") -> List[Document]:
```

- `input_dir`：目录或单个文件路径
- `source_filter`：**只保留路径里包含该关键词的文档**（本项目填 `"租房"`，从一堆 Docs 里挑出那一份 Q&A）

### 📝 完整可运行代码

```python
import config
from ragbase.ingest import load_clean_documents

# 只加载路径里含"租房"的文档，并清洗
docs = load_clean_documents(config.DOCUMENT_PATH, config.DOC_SOURCE_FILTER)
print(f"命中 {len(docs)} 篇，共 {sum(len(d.text) for d in docs)} 字符")
```

### 📊 输出结果

```
命中 1 篇，共 24709 字符
```

### ⚠️ 注意事项

1. **`source_filter` 用的是子串匹配 `in str(file_path)`**，两个坑：
   - Windows 路径分隔符是 `\`，且大小写不敏感，跨平台会不一致
   - 关键词太短会误伤（比如填 `"a"` 会命中几乎所有路径）
   生产环境应该换成 `Path.suffix` 白名单 + 显式文件清单。
2. **`clean_doc` 新建 Document 而不是改原对象**：避免副作用，清洗前的数据还能留着对比。
3. **`metadata` 是浅拷贝**：本函数直接把原 `metadata` 传给新 Document（`Document(text=..., metadata=doc.metadata)`），两个对象共享同一个 dict。改一个会影响另一个——本项目没有修改 metadata 的地方所以没暴露问题，但你要加字段时应该用 `dict(doc.metadata)`。

### ✅ 掌握标准

- [ ] 能说清 `load_clean_documents` 的三步
- [ ] 知道 `source_filter` 是子串匹配及其隐患
- [ ] 知道返回的是**新 Document 列表**，原对象没被改动

### 🎯 面试追问（大厂深度）

**Q1：为什么要做 `source_filter`，而不是换个目录？**

> 这是**课程资料目录**的现实约束：`Docs/markdown` 下混着多份文档，本项目只要租房那份。
> 工程上更好的做法是**配置化数据源清单**（YAML 里写清楚要哪些文件 + 各自的元数据），而不是靠文件名子串猜。
> 追问：**如果两份文档内容冲突怎么办？** → 给 Document 加 `metadata["source_of_truth"]` 字段，检索时按优先级重排；或者在 prompt 里让模型说明"存在两种说法"。

**Q2：`Document` 的 `metadata` 之后会流到哪里？**

> 一路带到向量库：本项目 `chunkers.py` 里每次 `m = dict(d.metadata)` 再塞 `chunk_index` / `chunk_strategy` / `source_file_path`，
> 最后 `store.py` 写入 Milvus 时 metadata 会变成**标量字段**，检索结果 `sources` 里就能带出 `rank` / `score` / `text`。
> 追问：**为什么 metadata 要写进向量库而不是只放内存？** → 因为向量库是唯一持久化层；重启进程后还能按 metadata 过滤（比如"只搜 2024 年之后的文档"）。

---

## 本讲小结

| 步骤 | 输入 | 输出 | 关键函数 |
|---|---|---|---|
| 加载 | 目录 | `List[Document]`（脏） | `SimpleDirectoryReader.load_data()` |
| 过滤 | 文档列表 | 命中的文档 | `source_filter in file_path` |
| 清洗 | 脏文本 | 干净文本 | `clean_text`（5 步） |

**下一讲**：数据层② —— 7 种分块策略逐个拆解（为什么默认选 `sentence`，`structure` 策略怎么修的 bug）。

## 动手验证清单

```bash
cd rental_rag_agent
python ragbase/ingest.py          # 看清洗后的文档
python examples/clean_demo.py     # 看 5 步清洗每一步的效果
python selftest.py                # 第 1 项就是"文档加载 + 清洗"
```
