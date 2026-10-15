"""极简 .env 加载器。

只做一件事：把项目根目录下 .env 里的 KEY=VALUE 注入 os.environ。
不覆盖已存在的环境变量（真实环境变量优先级更高）。
用它替代 python-dotenv，避免为了读 5 行配置多装一个依赖。
"""
import os

_loaded = False


def load_dotenv(path: str = None) -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True

    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            ".env"
        )
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
