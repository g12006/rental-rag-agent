"""
第 1 讲配套：数据清洗 5 步演示

运行：python examples/clean_demo.py
不需要 API Key，不需要向量库，纯正则 + unicodedata。
"""
import io
import sys
import unicodedata
from pathlib import Path

# 让脚本能被这样直接跑：python examples/clean_demo.py
# 直接跑时 sys.path[0] 是 examples/，项目根目录不在里面，需要手动加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from ragbase.ingest import _CONTROL_CHARS, _ZERO_WIDTH, clean_text

# 一段"脏"文本：BOM、零宽空格、全角字母数字、控制字符、CRLF、连续空行
DIRTY = (
    "\ufeff### **押金\ufeff规则**\u200b\r\n\r\n\r\n"
    "ＡＢＣ１２３ 元\x07\n\n\n"
    "第二条：退租需提前 30 天\n"
)


def show(step: str, text: str):
    """打印每一步的结果：repr 能看见肉眼不可见的字符"""
    print(f"{step:<14} 长度={len(text):>3}  {text!r}")


def main():
    print("=" * 72)
    print("原始文本")
    print("=" * 72)
    show("原始", DIRTY)

    print("\n" + "=" * 72)
    print("逐步执行 clean_text 内部的 5 步")
    print("=" * 72)

    t = DIRTY
    t = _ZERO_WIDTH.sub("", t)                 # ① 去零宽字符（BOM / ZWSP / ZWNJ / ZWJ）
    show("① 零宽字符", t)

    t = unicodedata.normalize("NFKC", t)      # ② 全角归一（ＡＢＣ１２３ -> ABC123）
    show("② NFKC 归一", t)

    t = _CONTROL_CHARS.sub("", t)             # ③ 去控制字符（\x07 响铃等）
    show("③ 控制字符", t)

    t = t.replace("\r\n", "\n").replace("\r", "\n")   # ④ 换行统一成 \n
    show("④ 换行统一", t)

    lines = [line.strip() for line in t.split("\n") if line.strip()]
    t = "\n".join(lines)                      # ⑤ 去首尾空白 + 丢掉空行
    show("⑤ 压缩空行", t)

    print("\n" + "=" * 72)
    print("直接调用 clean_text（一步到位，与上面 5 步等价）")
    print("=" * 72)
    result = clean_text(DIRTY)
    show("clean_text", result)
    print(f"\n长度：{len(DIRTY)} -> {len(result)}（少了 {len(DIRTY) - len(result)} 个字符）")

    print("\n[注意] NFKC 把全角冒号「：」也转成了半角「:」，见讲稿 ⚠️ 第 3 条")


if __name__ == "__main__":
    main()
