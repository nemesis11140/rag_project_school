"""公共工具层：不依赖任何业务包，供 knowledge_base / mysql_qa / rag_qa 共用。

这里刻意保持空 `__init__`：config.py 也会 import 本包（读 .env），
如果在 __init__ 里 eager import 子模块，会形成 config -> common -> config 的循环。
"""
