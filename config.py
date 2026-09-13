"""
全局配置
风格对齐 ragbasic-master/config.py：路径 / 模型 / 密钥集中在一处，由 .env 覆盖默认值。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录
base_path = Path(__file__).parent

# 解析 .env（优先级：环境变量 > .env > 代码默认值）
load_dotenv(base_path / ".env")

# ---------------------- 嵌入模型（ragbasic: BGE-M3 本地） ----------------------
# 对应 ragbasic-master/config.py: model_path
model_path = os.getenv("EMBED_MODEL_PATH", r"D:\models\BAAI\bge-m3")
# BGE-M3 输出 1024 维，与 ragbasic-master/vectorSearch.py 的 DIMENSION=1024 一致
embed_dim = int(os.getenv("EMBED_DIM", "1024"))

# ---------------------- 知识库 ----------------------
# 默认指向 LangChain 学习资料里的租房平台 Q&A 目录
DOCUMENT_PATH = Path(os.getenv(
    "DOCUMENT_PATH",
    r"C:\Users\龚祎\Desktop\学习资料\LangChain学习资料\课堂代码\PythonProject\Docs\markdown",
))
# 只保留文件路径包含该关键词的文档（空字符串 = 全部）
# 目录里还有 C++/Java/测试开发方向的 md，用 "租房" 过滤出租房知识库
DOC_SOURCE_FILTER = os.getenv("DOC_SOURCE_FILTER", "租房")

# ---------------------- 向量库后端：milvus / memory ----------------------
# milvus : BGE-M3 + Milvus Lite（ragbasic 原生组合，本地文件，无需服务）
# memory : OpenAIEmbeddings + InMemoryVectorStore（LangChain test30/31 组合，零额外依赖）
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "milvus").lower()
DB_NAME = os.getenv("DB_NAME", "rental_rag")
COLLECTION = os.getenv("COLLECTION", "rental_qa")
# Milvus Lite 的库目录（必须是纯 ASCII 路径）
# 原因：faiss 在 Windows 用 C 的 fopen 写索引文件，路径含中文会报
#   "could not open ...xxx.vector.autoindex.idx for writing: No such file or directory"
# 因此默认放到 C:\ragdata，而不是项目目录（项目在 C:\Users\龚祎\... 下）
MILVUS_DB_DIR = os.getenv("MILVUS_DB_DIR", r"C:\ragdata")
MILVUS_DB_FALLBACK = r"C:\ragdata"

# ---------------------- 大语言模型（DeepSeek，OpenAI 兼容协议） ----------------------
# 对应 LangChain/test2.py 的 ChatOpenAI(model / api_key / base_url)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-chat")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

# ---------------------- 检索参数（对齐 ragPromptEnhancer.get_prompt 默认值） ----------------------
TOP_K = int(os.getenv("TOP_K", "5"))
# 相关性下限：score（余弦相似度）低于该值的片段直接丢弃
# ragbasic 里写作 max_distance=0.5 并用 ">" 过滤，方向是反的（见 README 说明）
#
# 0.50 是在本知识库上实测校准出来的（见 README「阈值校准」）：
#   BGE-M3 下，相关查询的 top1 约 0.58-0.65，无关查询约 0.40-0.45
# 换知识库 / 换嵌入模型后请重新校准：python probe_scores.py
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.50"))
# 上下文压缩上限，对应 compress_context(max_chars=500)，这里放宽以容纳完整问答对
MAX_CHARS = int(os.getenv("MAX_CHARS", "1500"))

# ---------------------- 编排参数 ----------------------
# 分块策略：fixed / sentence / recursive / semantic / structure / llm / langchain
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "sentence")
# 问题重写次数上限，超过后强制进入生成节点，避免死循环
MAX_REWRITE = int(os.getenv("MAX_REWRITE", "2"))
# 触发历史消息摘要的消息条数上限
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "10"))
# 生成答案后是否进入人工审核（interrupt）
HUMAN_REVIEW = os.getenv("HUMAN_REVIEW", "false").lower() == "true"

# ---------------------- 服务 ----------------------
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8080"))
